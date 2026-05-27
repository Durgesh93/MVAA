"""
LightningModule for semi-supervised nnU-Net v2 training.

Prediction logic is in:
    prediction_module.py

Metric tracking and progress plotting logic is in:
    metrics_module.py
"""

from typing import Any, Dict

import torch
import lightning as L
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
)

from nnunetv2.paths import (
    nnUNet_preprocessed,
    nnUNet_results,
)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from prediction_module import PredictionMixin
from metrics_module import MetricsMixin


class SSLnnUNetLightningModule(
    PredictionMixin,
    MetricsMixin,
    L.LightningModule,
):
    def __init__(self, litmodule_cfg):
        super().__init__()

        self.cfg = litmodule_cfg

        self.dataset_id = self.cfg.dataset_id
        self.configuration = self.cfg.configuration
        self.fold = self.cfg.fold
        self.seed = self.cfg.seed
        

        self.plans_identifier = self.cfg.plans_identifier

        self.initial_lr = self.cfg.initial_lr
        self.weight_decay = self.cfg.weight_decay
        self.num_epochs = self.cfg.num_epochs

        self.enable_deep_supervision = True

        self.lambda_pseudo = self.cfg.lambda_pseudo
        self.pseudo_threshold = self.cfg.pseudo_threshold
        self.task_id = self.cfg.task_id
        self.prefix = self.cfg.prefix

        # ------------------------------------------------------------
        # Load plans.json and dataset.json
        # ------------------------------------------------------------
        self.dataset_name = maybe_convert_to_dataset_name(self.dataset_id)
        self.base         = join(nnUNet_preprocessed, self.dataset_name)
        self.plans        = load_json(join(self.base, self.plans_identifier + ".json"))
        self.dataset_json = load_json(join(self.base, "dataset.json"))

        # ------------------------------------------------------------
        # nnU-Net managers
        # ------------------------------------------------------------
        self.pm = PlansManager(self.plans)
        self.cm = self.pm.get_configuration(self.configuration)
        self.lm = self.pm.get_label_manager(self.dataset_json)

        self.num_input_channels = determine_num_input_channels(self.pm,self.cm,self.dataset_json)

        # ------------------------------------------------------------
        # Output folders
        # ------------------------------------------------------------
        self.actual_validation_output_base = join(
            nnUNet_results,
            self.dataset_name,
            self.plans_identifier
            + "__"
            + self.configuration,
        )

        self.actual_validation_output_folder = join(
            self.actual_validation_output_base,
            f"fold_{self.fold}",
            "validation",
        )

        self.actual_prediction_output_folder = join(
            self.actual_validation_output_base,
            f"fold_{self.fold}",
            f"{self.prefix}",
        )

        # Used by PredictionMixin._zip_validation_cases()
        # Stores validation zip files: image + prediction + GT.
        self.actual_validation_cases_folder = join(
            self.actual_validation_output_folder,
            "cases",
        )

        self.actual_prediction_cases_folder = join(
            self.actual_validation_output_base,
            f"fold_{self.fold}",
            f"{self.prefix}",
            f"cases",
        )

        # Temporary validation predictions used for metrics.
        self.actual_validation_tmp_preds_folder = join(
            self.actual_validation_output_folder,
            "_tmp_predictions",
        )

        self.gt_folder = join(
            self.base,
            "gt_segmentations",
        )

        # ------------------------------------------------------------
        # Progress plot output
        # ------------------------------------------------------------
        self.progress_folder = join(
            self.actual_validation_output_base,
            f"fold_{self.fold}",
        )

        self.progress_png_file = join(
            self.progress_folder,
            "training_progress.png",
        )

        # ------------------------------------------------------------
        # Metric tracking
        # ------------------------------------------------------------
        self._init_metric_tracking()

        # ------------------------------------------------------------
        # Lazy build
        # ------------------------------------------------------------
        self.network = None
        self.loss = None
        self._built = False

    # ------------------------------------------------------------------
    # Lazy build
    # ------------------------------------------------------------------
    def _ensure_built(self):
        if self._built:
            return

        self.network = self._build_network()
        self.loss = self._build_loss()
        self._built = True

    def setup(self, stage=None):
        self._ensure_built()

    # ------------------------------------------------------------------
    # nnU-Net trainer shim
    # ------------------------------------------------------------------
    def _make_trainer_shim(self):
        shim = type("S", (), {})()

        shim.configuration_manager = self.cm
        shim.label_manager = self.lm
        shim.enable_deep_supervision = self.enable_deep_supervision

        trainer = getattr(self, "trainer", None)

        if trainer is not None:
            shim.is_ddp = getattr(self.trainer, "world_size", 1) > 1
        else:
            shim.is_ddp = False

        shim._get_deep_supervision_scales = (
            lambda: nnUNetTrainer._get_deep_supervision_scales(shim)
        )

        shim._do_i_compile = lambda: False

        return shim

    # ------------------------------------------------------------------
    # Network/loss
    # ------------------------------------------------------------------
    def _build_network(self):
        return nnUNetTrainer.build_network_architecture(
            self.pm,
            self.cm,
            self.num_input_channels,
            self.lm.num_segmentation_heads,
            self.enable_deep_supervision,
        )

    def _build_loss(self):
        shim = self._make_trainer_shim()
        return nnUNetTrainer._build_loss(shim)

    def forward(self, x):
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
        data, target = self._get_batch_data_target(batch)

        output = self.network(data)
        loss = self.loss(output, target)

        return loss, output, target

    # ------------------------------------------------------------------
    # Pseudo-label loss
    # ------------------------------------------------------------------
    def _pseudo_loss(self, batch: Dict[str, Any]):
        if self.lambda_pseudo <= 0:
            return torch.tensor(0.0, device=self.device)

        data, _ = self._get_batch_data_target(batch)

        with torch.no_grad():
            pseudo_output = self.network(data)

            if isinstance(pseudo_output, (list, tuple)):
                pseudo_logits = pseudo_output[0]
            else:
                pseudo_logits = pseudo_output

            if self.lm.has_regions:
                probs = torch.sigmoid(pseudo_logits)

                pseudo_target = (
                    probs > self.pseudo_threshold
                ).float()

                confident_voxels = probs > self.pseudo_threshold

            else:
                probs = torch.softmax(pseudo_logits, dim=1)

                conf, pseudo_target = probs.max(
                    dim=1,
                    keepdim=True,
                )

                pseudo_target = pseudo_target.long()
                confident_voxels = conf >= self.pseudo_threshold

                if self.lm.ignore_label is not None:
                    pseudo_target[~confident_voxels] = self.lm.ignore_label

        student_output = self.network(data)

        if isinstance(student_output, (list, tuple)):
            student_logits = student_output[0]
        else:
            student_logits = student_output

        # Use the base loss, not DeepSupervisionWrapper,
        # because pseudo targets are only highest-resolution targets.
        pseudo_loss = self.loss.loss(
            student_logits,
            pseudo_target,
        )

        if confident_voxels.sum() == 0:
            pseudo_loss = pseudo_loss * 0.0

        return pseudo_loss

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        labeled_batch = batch["labeled"]
        unlabeled_batch = batch.get("unlabeled", None)

        sup_loss, _, _ = self._supervised_loss(labeled_batch)

        if unlabeled_batch is not None and self.lambda_pseudo > 0:
            pseudo_loss = self._pseudo_loss(unlabeled_batch)
        else:
            pseudo_loss = torch.tensor(0.0, device=self.device)

        total_loss = sup_loss + self.lambda_pseudo * pseudo_loss
        batch_size = labeled_batch["data"].shape[0]

        self.loss_metrics["train_loss"].update(
            total_loss.detach(),
            weight=batch_size,
        )

        self.loss_metrics["train_sup_loss"].update(
            sup_loss.detach(),
            weight=batch_size,
        )

        self.loss_metrics["train_pseudo_loss"].update(
            pseudo_loss.detach(),
            weight=batch_size,
        )

        self.log(
            "train_loss",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        self.log(
            "train_sup_loss",
            sup_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        self.log(
            "train_pseudo_loss",
            pseudo_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        return total_loss

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        val_loss, _, _ = self._supervised_loss(batch)
        batch_size = batch["data"].shape[0]

        self.loss_metrics["val_loss"].update(
            val_loss.detach(),
            weight=batch_size,
        )

        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return val_loss

    # ------------------------------------------------------------------
    # Predict step
    # ------------------------------------------------------------------
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        No-op predict step.

        Actual test prediction export is handled in on_predict_start()
        using nnU-Net's raw-file predictor.
        """
        return None
        
    # ------------------------------------------------------------------
    # Validation epoch end
    # ------------------------------------------------------------------
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        run_actual_validation = (
            self.current_epoch % 5 == 0
            or self.current_epoch == self.trainer.max_epochs - 1
        )

        if not run_actual_validation:
            self.loss_metrics.reset()
            return

        epoch_losses = self.loss_metrics.compute()
        self.loss_metrics.reset()

        if not self.trainer.is_global_zero:
            return

        # ------------------------------------------------------------
        # 1. Actual validation with metrics
        # Uses labeled validation cases.
        # PredictionMixin handles prediction, zip creation, and metrics.
        # ------------------------------------------------------------
        metrics = self.run_validation_prediction_with_metrics()
        validation_scores = self._extract_validation_scores(metrics)

        # ------------------------------------------------------------
        # 2. Test prediction export from raw imagesTs
        # No metrics, no GT, no zip.
        # PredictionMixin writes to actual_prediction_output_folder.
        # ------------------------------------------------------------
        self.run_test_prediction(
            save_probabilities=False,
            overwrite=True,
        )

        self._update_progress_metrics(
            epoch_losses=epoch_losses,
            validation_scores=validation_scores,
        )

        self._save_training_progress_plot()

    # ------------------------------------------------------------------
    # Predict start
    # ------------------------------------------------------------------
    def on_predict_start(self):
        """
        Export test predictions from raw imagesTs.

        The checkpoint is selected outside this module by Lightning:

            trainer.predict(model, datamodule=datamodule, ckpt_path="best")
            trainer.predict(model, datamodule=datamodule, ckpt_path="last")

        By the time this hook runs, Lightning has already loaded the selected
        checkpoint weights into self.network.
        """

        self._ensure_built()

        if not self.trainer.is_global_zero:
            return

        self.run_test_prediction(
            save_probabilities=False,
            overwrite=True,
        )
        
    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
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
