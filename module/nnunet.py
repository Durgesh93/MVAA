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
    CompoundLoss,
    FocalTverskyLoss,
    FocalLoss,
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
            "None of the compound losses in losses.py support "
            "region-based labels"
        )

        loss_cfg = self.cfg.losses[self.cfg.loss_type]

        soft_dice_kwargs = {
            "batch_dice": self.cm.batch_dice,
            "smooth": 1e-5,
            "do_bg": False,
            "ddp": shim.is_ddp,
        }

        dice_region_kwargs = {
            **soft_dice_kwargs,
            "alpha": 0.5,
            "beta": 0.5,
            "gamma": 1.0,
        }

        ce_pixel_kwargs = {"gamma": 0.0}

        if self.cfg.loss_type == "dice_ce":
            region_kwargs = dice_region_kwargs
            pixel_kwargs = ce_pixel_kwargs
        elif self.cfg.loss_type == "dice_focal":
            region_kwargs = dice_region_kwargs
            pixel_kwargs = {"gamma": loss_cfg.focal_gamma, "label_smoothing": 0.0}
        elif self.cfg.loss_type == "tversky_ce":
            region_kwargs = {
                **soft_dice_kwargs,
                "alpha": loss_cfg.tversky_alpha,
                "beta": loss_cfg.tversky_beta,
                "gamma": 1.0,
            }
            pixel_kwargs = ce_pixel_kwargs
        elif self.cfg.loss_type == "tversky_focal":
            region_kwargs = {
                **soft_dice_kwargs,
                "alpha": loss_cfg.tversky_alpha,
                "beta": loss_cfg.tversky_beta,
                "gamma": 1.0,
            }
            pixel_kwargs = {"gamma": loss_cfg.focal_gamma, "label_smoothing": 0.0}
        elif self.cfg.loss_type == "focaltversky_ce":
            region_kwargs = {
                **soft_dice_kwargs,
                "alpha": loss_cfg.tversky_alpha,
                "beta": loss_cfg.tversky_beta,
                "gamma": loss_cfg.focal_tversky_gamma,
            }
            pixel_kwargs = ce_pixel_kwargs
        else:
            region_kwargs = {
                **soft_dice_kwargs,
                "alpha": loss_cfg.tversky_alpha,
                "beta": loss_cfg.tversky_beta,
                "gamma": loss_cfg.focal_tversky_gamma,
            }
            pixel_kwargs = {"gamma": loss_cfg.focal_gamma, "label_smoothing": 0.0}

        loss = CompoundLoss(
            FocalTverskyLoss, region_kwargs,
            FocalLoss, pixel_kwargs,
            ignore_label=self.lm.ignore_label,
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

    def _nnu_build_optimizer_and_scheduler(self):
        shim = type("S", (), {})()

        shim.network = self.network
        shim.initial_lr = self.cfg.initial_lr
        shim.weight_decay = self.cfg.weight_decay
        shim.num_epochs = self.cfg.num_epochs

        return nnUNetTrainer.configure_optimizers(shim)
