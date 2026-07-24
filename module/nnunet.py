"""
NNUnetSetup for SSL nnU-Net.

Loads nnU-Net plans/dataset.json and builds the PlansManager/
ConfigurationManager/LabelManager, wraps nnU-Net's own trainer static
methods (network architecture, loss, optimizer/scheduler) behind
lightweight shim objects, and builds the supervised loss variants
defined in losses.py.

Standalone component: holds only its own plan/config state and the
litmodule cfg. Does not touch the LightningModule's `self` -- network
and loss are built here but owned by the LightningModule (required
for DDP wrapping, `.to(device)`, checkpointing, and optimizer param
discovery, all of which need `network`/`loss` as direct LightningModule
attributes). `network`/`device` are likewise passed into
predictor.run_prediction() per call rather than stored, for the same
reason.

Sliding-window inference + writing case zips/submission files live in
PredictionOps (composed, not mixed in) -- self.predictor. It shares
NNUnetSetup's own pm/cm/lm/dataset_json by construction, so callers
never need to pass configuration_manager themselves.
"""

import numpy as np

from batchgenerators.utilities.file_and_folder_operations import join, load_json

from nnunetv2.paths import nnUNet_preprocessed

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

from .losses import BoundaryLoss, CompoundLoss, WeakStrongPseudoLabelLoss

from utils import (
    write_prediction_case_zip as _write_prediction_case_zip,
    write_submission_prediction as _write_submission_prediction,
    keep_largest_component as _keep_largest_component,
)


class PredictionOps:
    """
    Sliding-window inference plus writing case zips/submission files.

    Constructed once by NNUnetSetup with its own pm/cm/lm/dataset_json
    -- no dependency on the LightningModule's `self`. `network`/`device`
    are passed into run_prediction() per call since those live on the
    LightningModule (network is trained in place; device can change
    with `.to()`).
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

    def write_prediction_case_zip(self, prediction, zip_dir, include_gt, reset_direction=False, keep_temp_folder=False):
        _write_prediction_case_zip(
            prediction=prediction,
            zip_dir=zip_dir,
            configuration_manager=self.cm,
            include_gt=include_gt,
            keep_temp_folder=keep_temp_folder,
            reset_direction=reset_direction,
        )

    def write_submission_prediction(self, prediction, output_folder, convert_to_255, keep_classes, submission_output_format):
        _write_submission_prediction(
            prediction=prediction,
            output_folder=output_folder,
            configuration_manager=self.cm,
            convert_to_255=convert_to_255,
            keep_classes=keep_classes,
            output_format=submission_output_format,
        )


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

    def _make_trainer_shim(self, is_ddp):
        shim = type("S", (), {})()
        shim.configuration_manager = self.cm
        shim.label_manager = self.lm
        shim.enable_deep_supervision = self.enable_deep_supervision
        shim.is_ddp = is_ddp
        shim._get_deep_supervision_scales = lambda: nnUNetTrainer._get_deep_supervision_scales(shim)
        shim._do_i_compile = lambda: False
        return shim

    def build_network(self):
        return nnUNetTrainer.build_network_architecture(
            self.pm, self.cm, self.num_input_channels, self.lm.num_segmentation_heads, self.enable_deep_supervision
        )

    def build_loss(self, is_ddp):
        shim = self._make_trainer_shim(is_ddp)

        assert not self.lm.has_regions, (
            "CompoundLoss (Dice + class-balanced CE + optional BoundaryLoss) does "
            "not support region-based labels"
        )

        if self.cfg.use_boundary:
            boundary_cls = BoundaryLoss
            boundary_kwargs = {"do_bg": False}
        else:
            boundary_cls = None
            boundary_kwargs = None

        loss = CompoundLoss(
            batch_dice=self.cm.batch_dice,
            ddp=is_ddp,
            ignore_label=self.lm.ignore_label,
            boundary_cls=boundary_cls,
            boundary_kwargs=boundary_kwargs,
            foreground_weight=getattr(self.cfg, "foreground_weight", 1.0),
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = shim._get_deep_supervision_scales()

            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])

            if is_ddp:
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()

            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def build_pseudo_loss(self):
        return WeakStrongPseudoLabelLoss(adaptive_momentum=self.cfg.pseudo_threshold_momentum)

    @staticmethod
    def _unwrap_compound_loss(loss):
        """The underlying CompoundLoss, unwrapped from DeepSupervisionWrapper if present."""
        return loss.loss if isinstance(loss, DeepSupervisionWrapper) else loss

    def update_boundary_weight(self, loss, epoch: int) -> None:
        """
        Ramps CompoundLoss.boundary_weight linearly from 0 at epoch 0
        to boundary_weight_max at boundary_ramp_epochs (held at max
        beyond that). No-op if use_boundary is off. Called once per
        epoch (see lightning_module.py's on_train_epoch_start) --
        Dice+CE anchors training throughout the ramp so the
        no-floor-on-its-own boundary term (see BoundaryLoss docstring)
        never dominates early.
        """
        if not self.cfg.use_boundary:
            return
        ramp_epochs = max(int(self.cfg.boundary_ramp_epochs), 1)
        weight = self.cfg.boundary_weight_max * min(epoch / ramp_epochs, 1.0)
        self._unwrap_compound_loss(loss).set_boundary_weight(weight)

    def build_optimizer_and_scheduler(self, network):
        shim = type("S", (), {})()

        shim.network = network
        shim.initial_lr = self.cfg.initial_lr
        shim.weight_decay = self.cfg.weight_decay
        shim.num_epochs = self.cfg.num_epochs

        return nnUNetTrainer.configure_optimizers(shim)
