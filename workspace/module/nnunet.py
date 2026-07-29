"""
NNUnetSetup for inference.

Loads nnU-Net plans/dataset.json and builds the PlansManager/
ConfigurationManager/LabelManager, and the network architecture via
nnU-Net's own trainer static method. Training-only pieces from the
original per-experiment nnunet.py (loss/optimizer construction, boundary-
weight ramping, and their losses.py dependency) are intentionally not
ported here -- inference never calls them.

Sliding-window inference lives in PredictionOps (composed, not mixed in)
-- self.predictor. It shares NNUnetSetup's own pm/cm/lm/dataset_json by
construction, so callers never need to pass configuration_manager
themselves. Mask writing is NOT wrapped here (see inference_module.py,
which calls utils.write_submission_prediction directly to get its
returned path).

Static (authored once, shared by every experiment) rather than copied
per-experiment. The one experiment-specific behavior found in the SSL
family (ssl_small_patch_size overrides the plans' patch_size) is handled
generically via an optional cfg.patch_size_override field instead of
forking this file per branch.
"""

from batchgenerators.utilities.file_and_folder_operations import join, load_json

from nnunetv2.paths import nnUNet_preprocessed

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

from utils import keep_largest_component as _keep_largest_component


class PredictionOps:
    """
    Sliding-window inference.

    Constructed once by NNUnetSetup with its own pm/cm/lm/dataset_json --
    no dependency on the LightningModule's `self`. `network`/`device` are
    passed into run_prediction() per call since those live on the
    LightningModule (network is trained in place; device can change with
    `.to()`).
    """

    def __init__(
        self,
        plans_manager,
        configuration_manager,
        label_manager,
        dataset_json,
        trainer_name,
        configuration_name,
        postprocess_keep_largest_component=False,
        use_mirroring=True,
        tile_step_size=0.5,
    ):
        self.pm = plans_manager
        self.cm = configuration_manager
        self.lm = label_manager
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.configuration_name = configuration_name
        self.postprocess_keep_largest_component = postprocess_keep_largest_component
        self.use_mirroring = use_mirroring
        self.tile_step_size = tile_step_size

    def _unwrap_network(self, network):
        if hasattr(network, "module"):
            return network.module
        return network

    def _make_predictor(self, net, device):
        predictor = nnUNetPredictor(
            tile_step_size=self.tile_step_size,
            use_gaussian=True,
            use_mirroring=self.use_mirroring,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.plans_manager = self.pm
        predictor.configuration_manager = self.cm
        predictor.network = net
        predictor.dataset_json = self.dataset_json
        predictor.trainer_name = self.trainer_name
        predictor.allowed_mirroring_axes = tuple(range(len(self.cm.patch_size)))
        predictor.label_manager = self.lm
        predictor.list_of_parameters = [net.state_dict()]
        return predictor

    def _predict_logits(self, network, device, data):
        if self.cm.previous_stage_name is not None:
            raise RuntimeError(
                f"Configuration {self.configuration_name} is cascaded, "
                "but this module does not support cascaded inference."
            )
        net = self._unwrap_network(network)
        old_deep_supervision = net.decoder.deep_supervision
        net.decoder.deep_supervision = False
        predictor = self._make_predictor(net, device)
        try:
            logits = predictor.predict_sliding_window_return_logits(data)
        finally:
            net.decoder.deep_supervision = old_deep_supervision
            compute_gaussian.cache_clear()
        return logits

    def _restore_prediction_shape(self, logits, properties):
        predicted_segments, predicted_probs = convert_predicted_logits_to_segmentation_with_correct_shape(
            predicted_logits=logits.cpu(),
            plans_manager=self.pm,
            configuration_manager=self.cm,
            label_manager=self.lm,
            properties_dict=properties,
            return_probabilities=True,
        )
        return predicted_segments, predicted_probs

    # =========================================================================
    # Main prediction entry point
    # =========================================================================
    def run_prediction(self, network, device, batch, batch_idx=None):
        item = batch[0]
        logits = self._predict_logits(network=network, device=device, data=item["data"])
        predicted_segments, predicted_probs = self._restore_prediction_shape(logits=logits, properties=item["properties"])
        if self.postprocess_keep_largest_component:
            predicted_segments = _keep_largest_component(predicted_segments, self.lm.foreground_labels)
        item.update({"logits": logits, "predicted_segments": predicted_segments, "predicted_probs": predicted_probs})
        return item


class NNUnetSetup:
    def __init__(self, litmodule_cfg, enable_deep_supervision=True, trainer_name="NNUnetSetup"):
        self.cfg = litmodule_cfg
        self.enable_deep_supervision = enable_deep_supervision

        self.dataset_name = maybe_convert_to_dataset_name(litmodule_cfg.dataset_id)
        self.base = join(nnUNet_preprocessed, self.dataset_name)
        self.plans = load_json(join(self.base, litmodule_cfg.plans_identifier + ".json"))
        self.dataset_json = load_json(join(self.base, "dataset.json"))

        self.pm = PlansManager(self.plans)
        self.cm = self.pm.get_configuration(litmodule_cfg.configuration)
        self.lm = self.pm.get_label_manager(self.dataset_json)

        patch_size_override = getattr(litmodule_cfg, "patch_size_override", None)
        if patch_size_override is not None:
            self.cm.configuration["patch_size"] = list(patch_size_override)

        self.num_input_channels = determine_num_input_channels(self.pm, self.cm, self.dataset_json)

        self.predictor = PredictionOps(
            plans_manager=self.pm,
            configuration_manager=self.cm,
            label_manager=self.lm,
            dataset_json=self.dataset_json,
            trainer_name=trainer_name,
            configuration_name=litmodule_cfg.configuration,
            postprocess_keep_largest_component=litmodule_cfg.postprocess_keep_largest_component,
            use_mirroring=litmodule_cfg.tta_use_mirroring,
            tile_step_size=litmodule_cfg.tta_tile_step_size,
        )

    def build_network(self):
        return nnUNetTrainer.build_network_architecture(
            self.pm, self.cm, self.num_input_channels, self.lm.num_segmentation_heads, self.enable_deep_supervision
        )
