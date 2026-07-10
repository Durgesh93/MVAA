from pathlib import Path
from typing import Any, Dict

import lightning as L

from nnunetv2.paths import nnUNet_results

from .metrics import MetricsTracker
from .ddp import DDPHelper
from .nnunet import NNUnetSetup

from utils import get_train_batch_data_target
from utils import save_training_progress_plot as write_training_progress_plot


class SSLnnUNetLightningModule(L.LightningModule):
    def __init__(self, litmodule_cfg):
        super().__init__()

        self.cfg = litmodule_cfg
        self.enable_deep_supervision = True

        self.is_t3_vid = str(self.cfg.prefix) == "t3_vid"
        self.convert_to_255 = self.is_t3_vid

        self.submission_output_format = "png" if self.is_t3_vid else "nii.gz"

        self.nnunet = NNUnetSetup(
            litmodule_cfg, enable_deep_supervision=self.enable_deep_supervision, trainer_name=self.__class__.__name__
        )
        self.dataset_name = self.nnunet.dataset_name
        self.dataset_json = self.nnunet.dataset_json
        self.keep_classes = [self.dataset_json["labels"]["class_10"]] if self.is_t3_vid else None

        self.actual_validation_output_base = (
            Path(nnUNet_results) / self.dataset_name / (self.cfg.plans_identifier + "__" + self.cfg.configuration)
        )

        self.fold_output_folder = self.actual_validation_output_base / f"fold_{self.cfg.fold}"
        self.actual_validation_output_folder = self.fold_output_folder / "validation"
        self.actual_prediction_output_folder = self.fold_output_folder / "prediction"
        self.actual_submission_output_folder = self.fold_output_folder / "submission"
        self.progress_png_file = self.fold_output_folder / "training_progress.png"

        self.ddp = DDPHelper()
        self.metrics = MetricsTracker(label_manager=self.nnunet.lm)

        self.network = None
        self.loss = None

    def setup(self, stage=None):
        self.network = self.nnunet.build_network()
        is_ddp = int(self.trainer.world_size) > 1
        self.loss = self.nnunet.build_loss(is_ddp=is_ddp)

    def forward(self, x):
        return self.network(x)

    def on_train_epoch_start(self):
        self.nnunet.update_boundary_weight(self.loss, self.current_epoch)

    def _supervised_loss(self, batch: Dict[str, Any]):
        data, target = get_train_batch_data_target(batch, device=self.device)
        output = self.network(data)
        loss = self.loss(output, target)
        return loss, output, target

    def training_step(self, batch, batch_idx):
        sup_loss, _, _ = self._supervised_loss(batch)
        self.metrics.update_step_training_metrics(train_loss=sup_loss.detach(), train_sup_loss=sup_loss.detach())
        return sup_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.trainer.sanity_checking:
            return None

        prediction = self.nnunet.predictor.run_prediction(
            network=self.network, device=self.device, batch=batch, batch_idx=batch_idx
        )

        rank_output_folder = self.ddp.rank_output_folder(
            trainer=self.trainer, fold_output_folder=self.fold_output_folder
        )

        if dataloader_idx == 0:
            self.nnunet.predictor.write_prediction_case_zip(
                prediction=prediction,
                zip_dir=rank_output_folder / "validation",
                include_gt=True,
                reset_direction=self.is_t3_vid,
            )
            metrics = self.metrics.compute_metrics(prediction, voxel_spacing=prediction["gt_properties"]["spacing"])
            self.metrics.update_step_val_metrics(metrics=metrics)
            return prediction

        self.nnunet.predictor.write_prediction_case_zip(
            prediction=prediction,
            zip_dir=rank_output_folder / "prediction",
            include_gt=False,
            reset_direction=self.is_t3_vid,
        )

        self.nnunet.predictor.write_submission_prediction(
            prediction=prediction,
            output_folder=rank_output_folder / "submission",
            convert_to_255=self.convert_to_255,
            keep_classes=self.keep_classes,
            submission_output_format=self.submission_output_format,
        )

        return prediction

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def _eval_epoch_end(self, stage, save_training_progress, print_val_metrics=False):
        if self.trainer.sanity_checking:
            self.metrics.reset_step_metrics()
            return

        synced_metrics = self.metrics.compute_step_metrics()
        do_print = print_val_metrics and self.trainer.is_global_zero

        if do_print:
            print()
            print("=" * 80)
            print(f"[{stage}] segmentation metrics")
            print("=" * 80)

        for key, label in [("dice", "dice   "), ("asd_mm", "asd_mm "), ("hd_mm", "hd_mm  "), ("hd95_mm", "hd95_mm")]:
            value = synced_metrics[key]
            self.log(
                key, value, on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=False, batch_size=1
            )
            if do_print:
                print(f"{label} : {value:.4f}")

        if do_print:
            print("=" * 80)
            print()
        self.metrics.update_epoch_metrics(synced_metrics=synced_metrics, current_epoch=self.current_epoch)
        if self.trainer.is_global_zero and save_training_progress:
            write_training_progress_plot(
                history=self.metrics.compute_epoch_history(),
                progress_png_file=self.actual_validation_output_base / "training_progress.png",
                dataset_name=self.dataset_name,
                fold=self.cfg.fold,
            )
        self.metrics.reset_step_metrics()
        self.ddp.merge_rank_outputs(
            trainer=self.trainer, fold_output_folder=self.fold_output_folder, task_id=self.cfg.task_id
        )

    def on_validation_epoch_end(self):
        self._eval_epoch_end(stage="validation", save_training_progress=True)

    def on_test_epoch_end(self):
        self._eval_epoch_end(stage="test", save_training_progress=False, print_val_metrics=True)

    def configure_optimizers(self):
        optimizer, scheduler = self.nnunet.build_optimizer_and_scheduler(self.network)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1}}
