"""
NNUnetMixin for SSL nnU-Net.

Loads nnU-Net plans/dataset.json and builds the PlansManager/
ConfigurationManager/LabelManager, wraps nnU-Net's own trainer static
methods (network architecture, loss, optimizer/scheduler) behind
lightweight shim objects, and builds the supervised loss variants
defined in losses.py.

All methods are prefixed with "nnu" so call sites (self._nnu_...) make
clear which mixin they come from.
"""

import numpy as np

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
)

from nnunetv2.paths import nnUNet_preprocessed

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnunetv2.utilities.dataset_name_id_conversion import (
    maybe_convert_to_dataset_name,
)

from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from .losses import (
    BoundaryLoss,
    CompoundLoss,
)


class NNUnetMixin:

    def _nnu_load_plans(self, litmodule_cfg):
        self.dataset_name = maybe_convert_to_dataset_name(
            litmodule_cfg.dataset_id
        )

        self.base = join(
            nnUNet_preprocessed,
            self.dataset_name,
        )

        self.plans = load_json(
            join(
                self.base,
                litmodule_cfg.plans_identifier + ".json",
            )
        )

        self.dataset_json = load_json(
            join(
                self.base,
                "dataset.json",
            )
        )

        self.pm = PlansManager(self.plans)

        self.cm = self.pm.get_configuration(
            litmodule_cfg.configuration
        )

        self.lm = self.pm.get_label_manager(
            self.dataset_json
        )

        self.num_input_channels = determine_num_input_channels(
            self.pm,
            self.cm,
            self.dataset_json,
        )

    def _nnu_ensure_built(self):
        if self._built:
            return

        self.network = self._nnu_build_network()
        self.loss = self._nnu_build_loss()

        self._built = True

    def _nnu_make_trainer_shim(self):
        shim = type("S", (), {})()

        shim.configuration_manager = self.cm
        shim.label_manager = self.lm
        shim.enable_deep_supervision = self.enable_deep_supervision
        shim.is_ddp = int(self.trainer.world_size) > 1

        shim._get_deep_supervision_scales = (
            lambda: nnUNetTrainer._get_deep_supervision_scales(shim)
        )

        shim._do_i_compile = lambda: False

        return shim

    def _nnu_build_network(self):
        return nnUNetTrainer.build_network_architecture(
            self.pm,
            self.cm,
            self.num_input_channels,
            self.lm.num_segmentation_heads,
            self.enable_deep_supervision,
        )

    def _nnu_build_loss(self):
        shim = self._nnu_make_trainer_shim()

        assert not self.lm.has_regions, (
            "CompoundLoss (DC_and_CE_loss + optional BoundaryLoss) does "
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
            ddp=shim.is_ddp,
            ignore_label=self.lm.ignore_label,
            boundary_cls=boundary_cls,
            boundary_kwargs=boundary_kwargs,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = shim._get_deep_supervision_scales()

            weights = np.array(
                [1 / (2**i) for i in range(len(deep_supervision_scales))]
            )

            if shim.is_ddp:
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()

            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def _nnu_compound_loss(self):
        """The underlying CompoundLoss, unwrapped from DeepSupervisionWrapper if present."""

        return (
            self.loss.loss
            if isinstance(self.loss, DeepSupervisionWrapper) else self.loss
        )

    def _nnu_update_boundary_weight(self, epoch: int) -> None:
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

        self._nnu_compound_loss().set_boundary_weight(weight)

    def _nnu_build_optimizer_and_scheduler(self):
        shim = type("S", (), {})()

        shim.network = self.network
        shim.initial_lr = self.cfg.initial_lr
        shim.weight_decay = self.cfg.weight_decay
        shim.num_epochs = self.cfg.num_epochs

        return nnUNetTrainer.configure_optimizers(shim)
