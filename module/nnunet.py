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

Sliding-window inference + writing case zips live in PredictionOps
(composed, not mixed in) -- self.predictor. It shares
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

from .adapter import EmbeddingAdapter
from .losses import ClusteringCMLoss, CompoundLoss

from utils import (
    write_prediction_case_zip as _write_prediction_case_zip,
    keep_largest_component as _keep_largest_component,
    override_patch_size as _override_patch_size,
)


class PredictionOps:
    """
    Sliding-window inference plus writing case zips.

    Constructed once by NNUnetSetup with its own pm/cm/lm/dataset_json
    -- no dependency on the LightningModule's `self`. `network`/`device`
    are passed into run_prediction() per call since those live on the
    LightningModule (network is trained in place; device can change
    with `.to()`).

    use_mirroring/tile_step_size are the TEST-time settings. They are not
    used for validation, which passes use_tta=False and runs a single
    non-overlapping pass instead: 3D mirroring is 2**3 forward passes per
    tile and a 0.5 step size roughly octuples the tile count, so full TTA
    costs ~64x a plain pass (measured: ~46 s/case vs the whole training
    epoch taking ~2 min). That is worth paying once for a final score,
    but per-epoch validation only has to rank checkpoints, and paying it
    300 times meant ~115 h of validation for ~10 h of training.
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

    def _make_predictor(self, net, device, use_mirroring, tile_step_size):
        predictor = nnUNetPredictor(
            tile_step_size=tile_step_size,
            use_gaussian=True,
            use_mirroring=use_mirroring,
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

    def _predict_logits(self, network, device, data, use_mirroring, tile_step_size):
        if self.cm.previous_stage_name is not None:
            raise RuntimeError(
                f"Configuration {self.configuration_name} is cascaded, "
                "but this module does not support cascaded inference."
            )
        net = self._unwrap_network(network)
        old_deep_supervision = net.decoder.deep_supervision
        net.decoder.deep_supervision = False
        predictor = self._make_predictor(net, device, use_mirroring, tile_step_size)
        try:
            logits = predictor.predict_sliding_window_return_logits(data)
        finally:
            net.decoder.deep_supervision = old_deep_supervision
            compute_gaussian.cache_clear()
        return logits

    def _restore_prediction_shape(self, logits, properties, return_probabilities=True):
        """
        Resample the logits back onto the case's original grid.

        The probability maps are only ever consumed by the case zip's
        per-class probability channels, so when nothing is being written they
        are skipped: nnU-Net returns the segmentation alone in that case, and
        holding a C-channel float volume at full original resolution is the
        single largest allocation in the eval path.
        """

        result = convert_predicted_logits_to_segmentation_with_correct_shape(
            predicted_logits=logits.cpu(),
            plans_manager=self.pm,
            configuration_manager=self.cm,
            label_manager=self.lm,
            properties_dict=properties,
            return_probabilities=return_probabilities,
        )

        if return_probabilities:
            return result

        return result, None

    # =========================================================================
    # Main prediction entry point
    # =========================================================================
    def run_prediction(self, network, device, batch, batch_idx=None, use_tta=True, need_probs=True):
        """
        use_tta=False forces a single non-overlapping sliding-window pass
        (no mirroring, step 1.0), which is what validation uses. True uses
        the configured tta_use_mirroring / tta_tile_step_size.

        need_probs=False skips building the full-resolution probability
        volume, which only the case zip reads.
        """

        use_mirroring = self.use_mirroring if use_tta else False
        tile_step_size = self.tile_step_size if use_tta else 1.0

        item = batch[0]
        logits = self._predict_logits(
            network=network,
            device=device,
            data=item["data"],
            use_mirroring=use_mirroring,
            tile_step_size=tile_step_size,
        )
        predicted_segments, predicted_probs = self._restore_prediction_shape(
            logits=logits, properties=item["properties"], return_probabilities=need_probs
        )
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


class NNUnetSetup:
    def __init__(self, litmodule_cfg, trainer_name="NNUnetSetup"):
        self.cfg = litmodule_cfg

        # Unconditional, so there is no flag and nothing to keep in step with
        # the datamodule (a disagreement there used to kill the first
        # training step on shapes). It is nnU-Net's own default and a real
        # contributor on thin structures like the leaflets, and it keeps
        # every seg_layer in the graph -- with it off the decoder applies
        # only seg_layers[-1], leaving the rest without gradients, which
        # trips DDP as engine.py builds it (no find_unused_parameters).
        #
        # The attribute stays because nnU-Net's own
        # _get_deep_supervision_scales() reads it off the trainer shim.
        #
        # The clustering loss is indifferent: it reads stage 0 only.
        self.enable_deep_supervision = True

        self.dataset_name = maybe_convert_to_dataset_name(litmodule_cfg.dataset_id)
        self.base = join(nnUNet_preprocessed, self.dataset_name)
        self.plans = load_json(join(self.base, litmodule_cfg.plans_identifier + ".json"))
        self.dataset_json = load_json(join(self.base, "dataset.json"))

        self.pm = PlansManager(self.plans)
        self.cm = self.pm.get_configuration(litmodule_cfg.configuration)
        self.lm = self.pm.get_label_manager(self.dataset_json)

        _override_patch_size(self.cm, litmodule_cfg.patch_size_override)

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
        """
        The nnU-Net trunk, ending in an embedding adapter.

        Stock nnU-Net's decoder seg_layers map straight to
        num_segmentation_heads logits. Here the SAME trunk is built with
        embedding_dim output channels and a shared 1x1 classifier bolted on
        top, so every voxel carries an explicit embedding_dim-dimensional
        feature vector the clustering loss can reach (see
        module/adapter.py). The returned module still emits class logits
        from forward(), so the loss, the deep-supervision wrapper and
        nnUNetPredictor are unaffected.

        This is unconditional. There is no nonlinearity between the trunk
        and the classifier, so their composition is still a linear map to
        logits and embedding_dim (128) >> num_classes (3) keeps it full
        rank: the adapter buys an embedding space to put a loss on, not
        expressiveness, and a purely supervised run is free to ignore it
        (cm_weight=0). Keeping it always on means one architecture and one
        code path, so a supervised baseline and a SuperCM run differ only in
        the clustering weight.

        Checkpoints are NOT compatible with stock nnU-Net -- the adapter
        renames every trunk parameter under "backbone." and adds
        "classifier." -- so pretrained stock weights need remapping, not
        just loading.
        """

        embedding_dim = int(self.cfg.embedding_dim)

        if embedding_dim < self.lm.num_segmentation_heads:
            raise ValueError(
                f"embedding_dim={embedding_dim} is below num_segmentation_heads="
                f"{self.lm.num_segmentation_heads}. The classifier is a linear map off the "
                "embedding, so a narrower embedding would bottleneck the logits."
            )

        backbone = nnUNetTrainer.build_network_architecture(
            self.pm, self.cm, self.num_input_channels, embedding_dim, self.enable_deep_supervision
        )

        # Conv class read off the trunk's own seg_layers rather than
        # hardcoded to Conv3d, so a 2d configuration needs no special case.
        conv_op = type(backbone.decoder.seg_layers[0])

        return EmbeddingAdapter(
            backbone=backbone,
            embedding_dim=embedding_dim,
            num_classes=self.lm.num_segmentation_heads,
            conv_op=conv_op,
        )

    def build_loss(self, is_ddp):
        shim = self._make_trainer_shim(is_ddp)

        assert not self.lm.has_regions, (
            "CompoundLoss (Dice + class-balanced CE) does "
            "not support region-based labels"
        )

        loss = CompoundLoss(
            batch_dice=self.cm.batch_dice,
            ddp=is_ddp,
            ignore_label=self.lm.ignore_label,
            foreground_weight=getattr(self.cfg, "foreground_weight", 1.0),
        )

        deep_supervision_scales = shim._get_deep_supervision_scales()

        weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])

        if is_ddp:
            weights[-1] = 1e-6
        else:
            weights[-1] = 0

        weights = weights / weights.sum()

        return DeepSupervisionWrapper(loss, weights)

    def build_cm_loss(self):
        """
        The SuperCM-style clustering regularizer.

        Always built, like the adapter it depends on. Disable its effect
        with cm_weight=0, which is what the supervised baseline sweep does:
        the centroids are still tracked and the energy still logged, but it
        contributes no gradient, so the baseline and a SuperCM run take the
        identical code path and differ only in that weight.

        num_classes and ignore_label are read off the label manager rather
        than the yaml so they cannot drift from what the supervised loss and
        the predictor use. embedding_dim comes from the config because it is
        the same value build_network() hands the trunk -- the two have to
        agree, and a mismatch surfaces immediately as a shape error in the
        centroid einsum.
        """

        return ClusteringCMLoss(
            embedding_dim=int(self.cfg.embedding_dim),
            num_classes=self.lm.num_segmentation_heads,
            cm_mode=str(self.cfg.cm_mode),
            weight=float(self.cfg.cm_weight),
            ema_weight=float(self.cfg.cm_ema_weight),
            conf_thr=float(self.cfg.cm_conf_thr),
            topk_frac=float(self.cfg.cm_topk_frac),
            min_keep=int(self.cfg.cm_min_keep),
            use_soft_pseudo=bool(getattr(self.cfg, "cm_use_soft_pseudo", False)),
            normalize=bool(getattr(self.cfg, "cm_normalize", False)),
            ignore_label=self.lm.ignore_label,
        )


    def build_optimizer_and_scheduler(self, network):
        shim = type("S", (), {})()

        shim.network = network
        shim.initial_lr = self.cfg.initial_lr
        shim.weight_decay = self.cfg.weight_decay
        shim.num_epochs = self.cfg.num_epochs

        return nnUNetTrainer.configure_optimizers(shim)
