"""
LightningModule for semi-supervised nnU-Net v2 training.
Uses Hydra/OmegaConf litmodule config object.
"""

from typing import Any, Dict

import torch
import lightning as L
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from utils import set_nnunet_env


class SSLnnUNetLightningModule(L.LightningModule):
    def __init__(self, litmodule_cfg):
        super().__init__()

        self.cfg = litmodule_cfg

        # ------------------------------------------------------------------
        # nnU-Net paths
        # ------------------------------------------------------------------
        self.nnunet_raw = self.cfg.paths.nnunet_raw
        self.nnunet_preprocessed = self.cfg.paths.nnunet_preprocessed
        self.nnunet_results = self.cfg.paths.nnunet_results

        set_nnunet_env(
            nnunet_raw=self.nnunet_raw,
            nnunet_preprocessed=self.nnunet_preprocessed,
            nnunet_results=self.nnunet_results,
        )

        self._env = {
            "nnunet_raw": self.nnunet_raw,
            "nnunet_preprocessed": self.nnunet_preprocessed,
            "nnunet_results": self.nnunet_results,
        }

        # ------------------------------------------------------------------
        # Config values
        # ------------------------------------------------------------------
        self.dataset_id = self.cfg.dataset_id
        self.configuration = self.cfg.configuration
        self.seed = self.cfg.get("seed", 12345)

        self.plans_identifier = self.cfg.get("plans_identifier", "nnUNetPlans")

        self.initial_lr = self.cfg.get("initial_lr", 1e-2)
        self.weight_decay = self.cfg.get("weight_decay", 3e-5)
        self.num_epochs = self.cfg.get("num_epochs", 1000)

        self.enable_deep_supervision = self.cfg.get(
            "enable_deep_supervision",
            True,
        )

        self.lambda_pseudo = self.cfg.get("lambda_pseudo", 0.0)
        self.pseudo_threshold = self.cfg.get("pseudo_threshold", 0.95)
        self.compile_network = self.cfg.get("compile_network", False)

        # ------------------------------------------------------------------
        # Load plans.json and dataset.json
        # ------------------------------------------------------------------
        self.dataset_name = maybe_convert_to_dataset_name(self.dataset_id)
        self.base = join(nnUNet_preprocessed, self.dataset_name)

        self.plans = load_json(join(self.base, self.plans_identifier + ".json"))
        self.dataset_json = load_json(join(self.base, "dataset.json"))

        # ------------------------------------------------------------------
        # Managers
        # ------------------------------------------------------------------
        self.pm = PlansManager(self.plans)
        self.cm = self.pm.get_configuration(self.configuration)
        self.lm = self.pm.get_label_manager(self.dataset_json)

        self.num_input_channels = determine_num_input_channels(
            self.pm,
            self.cm,
            self.dataset_json,
        )

        # ------------------------------------------------------------------
        # Lazy build
        # ------------------------------------------------------------------
        self.network = None
        self.loss = None
        self._built = False

        self.save_hyperparameters(
            ignore=[
                "litmodule_cfg",
                "plans",
                "dataset_json",
                "pm",
                "cm",
                "lm",
            ]
        )

    # ------------------------------------------------------------------
    # Lazy build
    # ------------------------------------------------------------------
    def _ensure_built(self):
        if self._built:
            return

        self.network = self._build_network()

        if self.compile_network:
            self.network = torch.compile(self.network)

        self.loss = self._build_loss()
        self._built = True

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def setup(self, stage=None):
        set_nnunet_env(**self._env)
        self._ensure_built()

    def on_train_start(self):
        self._ensure_built()

    def on_validation_start(self):
        self._ensure_built()

    def on_test_start(self):
        self._ensure_built()

    def on_predict_start(self):
        self._ensure_built()

    # ------------------------------------------------------------------
    # nnU-Net shim helpers
    # ------------------------------------------------------------------
    def _make_trainer_shim(self):
        shim = type("S", (), {})()

        shim.configuration_manager = self.cm
        shim.label_manager = self.lm
        shim.enable_deep_supervision = self.enable_deep_supervision

        try:
            shim.is_ddp = self.trainer.world_size > 1
        except RuntimeError:
            shim.is_ddp = False

        shim._get_deep_supervision_scales = (
            lambda: nnUNetTrainer._get_deep_supervision_scales(shim)
        )

        shim._do_i_compile = lambda: False

        return shim

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------
    def _build_network(self):
        return nnUNetTrainer.build_network_architecture(
            self.pm,
            self.cm,
            self.num_input_channels,
            self.lm.num_segmentation_heads,
            self.enable_deep_supervision,
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _build_loss(self):
        shim = self._make_trainer_shim()
        return nnUNetTrainer._build_loss(shim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        self._ensure_built()
        return self.network(x)

    # ------------------------------------------------------------------
    # Batch helper
    # ------------------------------------------------------------------
    def _get_batch_data_target(self, batch: Dict[str, Any]):
        data = batch["data"].to(self.device, non_blocking=True)

        target = batch.get("target", None)

        if target is not None:
            if isinstance(target, (list, tuple)):
                target = [
                    t.to(self.device, non_blocking=True)
                    for t in target
                ]
            else:
                target = target.to(self.device, non_blocking=True)

        return data, target

    # ------------------------------------------------------------------
    # Supervised loss
    # ------------------------------------------------------------------
    def _supervised_loss(self, batch: Dict[str, Any]):
        self._ensure_built()

        data, target = self._get_batch_data_target(batch)
        output = self.network(data)

        if self.enable_deep_supervision:
            if not isinstance(output, (list, tuple)):
                raise RuntimeError(
                    "enable_deep_supervision=True, but network output is not a list/tuple."
                )

            if not isinstance(target, (list, tuple)):
                raise RuntimeError(
                    "enable_deep_supervision=True, but target is not a list/tuple. "
                    "Fix SSLnnUNetDataModule so deep_supervision_scales is passed "
                    "to nnU-Net transforms."
                )

        else:
            if isinstance(output, (list, tuple)):
                output = output[0]

            if isinstance(target, (list, tuple)):
                target = target[0]

        loss = self.loss(output, target)

        return loss, output, target

    # ------------------------------------------------------------------
    # Optional pseudo-label loss
    # ------------------------------------------------------------------
    def _pseudo_loss(self, batch: Dict[str, Any]):
        self._ensure_built()

        if self.lambda_pseudo <= 0:
            return torch.tensor(0.0, device=self.device)

        data, _ = self._get_batch_data_target(batch)

        with torch.no_grad():
            pseudo_output = self.network(data)

            if isinstance(pseudo_output, (list, tuple)):
                pseudo_output = pseudo_output[0]

            if self.lm.has_regions:
                probs = torch.sigmoid(pseudo_output)
                pseudo_target = (probs > self.pseudo_threshold).float()

            else:
                probs = torch.softmax(pseudo_output, dim=1)
                conf, pseudo_target = probs.max(dim=1, keepdim=True)
                pseudo_target = pseudo_target.long()

                if self.lm.ignore_label is not None:
                    pseudo_target[conf < self.pseudo_threshold] = self.lm.ignore_label

        student_output = self.network(data)

        if isinstance(student_output, (list, tuple)):
            student_output = student_output[0]

        base_loss = getattr(self.loss, "loss", self.loss)

        pseudo_loss = base_loss(student_output, pseudo_target)

        return pseudo_loss

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        self._ensure_built()

        labeled_batch = batch["labeled"]
        unlabeled_batch = batch.get("unlabeled", None)

        sup_loss, _, _ = self._supervised_loss(labeled_batch)

        if unlabeled_batch is not None and self.lambda_pseudo > 0:
            pseudo_loss = self._pseudo_loss(unlabeled_batch)
        else:
            pseudo_loss = torch.tensor(0.0, device=self.device)

        total_loss = sup_loss + self.lambda_pseudo * pseudo_loss

        batch_size = labeled_batch["data"].shape[0]

        self.log(
            "train_loss",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        self.log(
            "train_sup_loss",
            sup_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        self.log(
            "train_pseudo_loss",
            pseudo_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return total_loss

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        self._ensure_built()

        val_loss, _, _ = self._supervised_loss(batch)

        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch["data"].shape[0],
        )

        return val_loss

    # ------------------------------------------------------------------
    # Test step
    # ------------------------------------------------------------------
    def test_step(self, batch, batch_idx):
        self._ensure_built()

        test_loss, _, _ = self._supervised_loss(batch)

        self.log(
            "test_loss",
            test_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch["data"].shape[0],
        )

        return test_loss

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        self._ensure_built()

        shim = type("S", (), {})()

        shim.network = self.network
        shim.initial_lr = self.initial_lr
        shim.weight_decay = self.weight_decay
        shim.num_epochs = self.num_epochs

        optimizer, scheduler = nnUNetTrainer.configure_optimizers(shim)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
