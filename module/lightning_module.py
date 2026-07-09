"""
LightningModule for SSL nnU-Net.

Folder structure:
    fold_all/
    ├── validation/          # Slicer MRML case zips, includes gt
    ├── Test/                # Slicer MRML case zips, no gt
    ├── submission/          # *-pred.nii.gz / *_label_bin.png (task_id_predictions.json is built by engine.py submit, from zip contents)
    ├── _rank_outputs/       # temporary DDP rank-local outputs
    ├── checkpoints/
    └── training_progress.png
"""

from typing import Any, Dict

import lightning as L

from batchgenerators.utilities.file_and_folder_operations import (
    join,
)

from nnunetv2.paths import (
    nnUNet_results,
)

from .prediction import PredictionMixin
from .metrics import MetricsMixin
from .ddp import DDPMixin
from .nnunet import NNUnetMixin

from utils import (
    write_prediction_case_zip,
    write_submission_prediction,
    get_train_batch_data_target,
)


class SSLnnUNetLightningModule(
    PredictionMixin,
    DDPMixin,
    MetricsMixin,
    NNUnetMixin,
    L.LightningModule,
):
    def __init__(self, litmodule_cfg):
        super().__init__()

        self.cfg = litmodule_cfg

        assert self.cfg.loss_type in (
            "dice_ce",
            "dice_focal",
            "tversky_ce",
            "tversky_focal",
            "focaltversky_ce",
            "focaltversky_focal",
        ), (
            f"Unknown loss_type '{self.cfg.loss_type}'. "
            "Use 'dice_ce', 'dice_focal', 'tversky_ce', 'tversky_focal', "
            "'focaltversky_ce', or 'focaltversky_focal'."
        )

        self.enable_deep_supervision = True

        # ------------------------------------------------------------
        # Task-specific submission format
        # ------------------------------------------------------------
        self.is_t3_vid = str(self.cfg.prefix) == "t3_vid"

        # Task 3 is binary, so PNG submission should be 0/255.
        # Other tasks keep normal label values.
        self.convert_to_255 = self.is_t3_vid

        # Task 3 Codabench submission uses PNG.
        # CT/TEE stay as NIfTI.
        self.submission_output_format = (
            "png" if self.is_t3_vid else "nii.gz"
        )

        self._nnu_load_plans(litmodule_cfg)

        # Task 3 masks are multi-class (class_10 "LV", class_11).
        # Submission output keeps only these classes as foreground.
        # Currently just class_10 (LV); add more label names here
        # to keep additional classes in the submission.
        self.keep_classes = (
            [self.dataset_json["labels"]["class_10"]]
            if self.is_t3_vid
            else None
        )

        self.actual_validation_output_base = join(
            nnUNet_results,
            self.dataset_name,
            self.cfg.plans_identifier + "__" + self.cfg.configuration,
        )

        self.fold_output_folder = join(
            self.actual_validation_output_base,
            f"fold_{self.cfg.fold}",
        )

        self.actual_validation_output_folder = join(
            self.fold_output_folder,
            "validation",
        )

        self.actual_prediction_output_folder = join(
            self.fold_output_folder,
            "prediction",
        )

        self.actual_submission_output_folder = join(
            self.fold_output_folder,
            "submission",
        )

        self.progress_png_file = join(
            self.fold_output_folder,
            "training_progress.png",
        )

        self._metric_init_tracking()

        self.network = None
        self.loss = None
        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def setup(self, stage=None):
        self._nnu_ensure_built()

    def forward(self, x):
        return self.network(x)

    def on_train_epoch_start(self):
        self._nnu_update_boundary_weight(self.current_epoch)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def _supervised_loss(
        self,
        batch: Dict[str, Any],
    ):
        data, target = get_train_batch_data_target(
            batch,
            device=self.device,
        )

        output = self.network(data)

        loss = self.loss(
            output,
            target,
        )

        return loss, output, target

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    def training_step(
        self,
        batch,
        batch_idx,
    ):
        sup_loss, _, _ = self._supervised_loss(
            batch
        )

        self.metric_update_step_training_metrics(
            train_loss=sup_loss.detach(),
            train_sup_loss=sup_loss.detach(),
        )

        return sup_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if self.trainer.sanity_checking:
            return None

        prediction = self.pred_run_prediction(
            batch=batch,
            batch_idx=batch_idx,
        )

        rank_output_folder = self._ddp_rank_output_folder()

        # ------------------------------------------------------------
        # Dataloader 0:
        # validation cases with GT
        # ------------------------------------------------------------
        if dataloader_idx == 0:
            write_prediction_case_zip(
                prediction=prediction,
                zip_dir=rank_output_folder / "validation",
                configuration_manager=self.cm,
                include_gt=True,
                reset_direction=self.is_t3_vid,
            )

            metrics = self.metric_compute_metrics(
                prediction
            )

            self.metric_update_step_val_metrics(
                metrics=metrics,
            )

            return prediction

        # ------------------------------------------------------------
        # Dataloader 1:
        # test cases without GT
        # ------------------------------------------------------------
        write_prediction_case_zip(
            prediction=prediction,
            zip_dir=rank_output_folder / "prediction",
            configuration_manager=self.cm,
            include_gt=False,
            reset_direction=self.is_t3_vid,
        )

        write_submission_prediction(
            prediction=prediction,
            output_folder=rank_output_folder / "submission",
            configuration_manager=self.cm,
            convert_to_255=self.convert_to_255,
            keep_classes=self.keep_classes,
            output_format=self.submission_output_format,
        )

        return prediction

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def test_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        return self.validation_step(
            batch,
            batch_idx,
            dataloader_idx,
        )

    # ------------------------------------------------------------------
    # Shared validation/test epoch-end
    # ------------------------------------------------------------------
    def _eval_epoch_end(
        self,
        stage,
        save_training_progress,
        print_val_metrics=False,
    ):
        if self.trainer.sanity_checking:
            self.step_metrics.reset()
            return

        synced_metrics = self.step_metrics.compute()

        self.metric_log_validation_epoch_metrics(
            synced_metrics=synced_metrics,
        )

        if print_val_metrics:
            self.metric_print_validation_epoch_metrics(
                synced_metrics
            )

        # Always update epoch history.
        self._metric_update_epoch_metrics(
            synced_metrics=synced_metrics,
        )

        if self.trainer.is_global_zero:
            if save_training_progress:
                self._metric_save_training_progress_plot()

        self.step_metrics.reset()

        # Always merge rank outputs after every validation/test epoch.
        self._ddp_merge_rank_outputs()

    def on_validation_epoch_end(self):
        self._eval_epoch_end(
            stage="validation",
            save_training_progress=True,
        )

    def on_test_epoch_end(self):
        self._eval_epoch_end(
            stage="test",
            save_training_progress=False,
            print_val_metrics=True,
        )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        self._nnu_ensure_built()

        optimizer, scheduler = self._nnu_build_optimizer_and_scheduler()

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
