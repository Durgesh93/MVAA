"""
Metric utilities for SSL nnU-Net LightningModule.

Tracks training losses and computes validation segmentation metrics.
metric_compute_metrics() is called only in validation_step after
pred_run_prediction().

All methods are prefixed with "metric" so call sites (self._metric_... /
self.metric_...) make clear which mixin they come from.
"""

from pathlib import Path

import numpy as np
import torch

from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.aggregation import CatMetric

from utils import (
    save_training_progress_plot,
    safe_binary_segmentation_metrics,
    safe_float,
    to_numpy,
)


class MetricsMixin:
    def _metric_init_tracking(self):
        self.tracked_metric_keys = [
            "train_loss",
            "train_sup_loss",
            "dice",
            "asd_mm",
            "hd_mm",
            "hd95_mm",
        ]

        self.epoch_metric_keys = [
            "epoch",
            *self.tracked_metric_keys,
        ]

        self.step_metrics = MetricCollection(
            {
                key: MeanMetric(sync_on_compute=True)
                for key in self.tracked_metric_keys
            }
        )

        self.epoch_metrics = MetricCollection(
            {
                key: CatMetric(sync_on_compute=False)
                for key in self.epoch_metric_keys
            }
        )

    def _metric_mean_valid(self, values):
        valid_values = []

        for value in values:
            value = safe_float(value)

            if value is not None:
                valid_values.append(value)

        if len(valid_values) == 0:
            return None

        return float(np.mean(valid_values))

    def _metric_get_labels(self):
        foreground_labels = list(self.lm.foreground_labels)

        # Overall = background + foreground.
        # Background is normally label 0.
        all_labels = [0] + foreground_labels

        return all_labels, foreground_labels

    def _metric_log_scalar(
        self,
        name,
        value,
        prog_bar=False,
        logger=True,
        sync_dist=True,
    ):
        """
        Safe Lightning scalar logging.

        This does not use add_dataloader_idx because the installed
        Lightning version does not support that argument.
        """

        value = safe_float(value)

        if value is None:
            return None

        self.log(
            name,
            value,
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar,
            logger=logger,
            sync_dist=sync_dist,
            batch_size=1,
        )

        return value

    def metric_compute_metrics(self, prediction):
        """
        Compute validation segmentation metrics for one case.

        Expected:
            predicted_segments: Tensor or NumPy array, shape (X,Y,Z) or (H,W)
            gt_data:            Tensor or NumPy array, shape (X,Y,Z) or (H,W)

        Returns:
            {
                "overall_mean": {
                    "dice": ...,
                    "asd_mm": ...,
                    "hd_mm": ...,
                    "hd95_mm": ...,
                },
                "foreground_mean": {
                    "dice": ...,
                    "asd_mm": ...,
                    "hd_mm": ...,
                    "hd95_mm": ...,
                },
                "classwise": {
                    "0": {...},
                    "1": {...},
                    ...
                }
            }
        """

        pred = to_numpy(prediction["predicted_segments"])
        gt = to_numpy(prediction["gt_data"])

        pred = np.asarray(pred)
        gt = np.asarray(gt)

        # pred and gt are both label maps (integer class ids) here, not
        # multi-channel logits/probabilities -- the network's C-channel
        # logits were already collapsed to a label map upstream (nnU-Net's
        # convert_predicted_logits_to_segmentation_with_correct_shape).
        #
        # Only the 2D case can still carry a leftover leading singleton,
        # since a 2D slice is itself represented as (1, H, W):
        #   pred [1, H, W] -> [H, W]
        #   gt   [1, H, W] -> [H, W]
        # 3D pred/gt arrive as (D, H, W) already and are left unchanged.
        if pred.ndim >= 3 and pred.shape[0] == 1:
            pred = pred[0]

        if gt.ndim >= 3 and gt.shape[0] == 1:
            gt = gt[0]

        voxel_spacing = prediction["gt_properties"]["spacing"]
        voxel_spacing = to_numpy(voxel_spacing)
        voxel_spacing = np.asarray(voxel_spacing).reshape(-1)
        voxel_spacing = tuple(float(x) for x in voxel_spacing)

        # MedPy requires len(voxel_spacing) == pred.ndim.
        # For 2D masks, spacing may still be [z, y, x],
        # so use the last two values [y, x].
        if len(voxel_spacing) != pred.ndim:
            voxel_spacing = voxel_spacing[-pred.ndim:]

        all_labels, foreground_labels = self._metric_get_labels()

        classwise = {}

        for label in all_labels:
            pred_mask = pred == label
            gt_mask = gt == label

            scores = safe_binary_segmentation_metrics(
                pred_mask=pred_mask,
                gt_mask=gt_mask,
                voxel_spacing=voxel_spacing,
            )

            classwise[str(label)] = {
                "dice": scores["Dice"],
                "asd_mm": scores["ASD_mm"],
                "hd_mm": scores["HD_mm"],
                "hd95_mm": scores["HD95_mm"],
            }

        metric_names = [
            "dice",
            "asd_mm",
            "hd_mm",
            "hd95_mm",
        ]

        overall_mean = {}
        foreground_mean = {}

        for metric_name in metric_names:
            overall_mean[metric_name] = self._metric_mean_valid(
                [
                    classwise[str(label)][metric_name]
                    for label in all_labels
                ]
            )

            foreground_mean[metric_name] = self._metric_mean_valid(
                [
                    classwise[str(label)][metric_name]
                    for label in foreground_labels
                ]
            )

        return {
            "overall_mean": overall_mean,
            "foreground_mean": foreground_mean,
            "classwise": classwise,
        }

    def metric_update_step_training_metrics(
        self,
        train_loss,
        train_sup_loss,
    ):
        values = {
            "train_loss": train_loss,
            "train_sup_loss": train_sup_loss,
        }

        for key, value in values.items():
            value = safe_float(value)

            if value is not None:
                self.step_metrics[key].update(value)

    def metric_update_step_val_metrics(
        self,
        metrics,
    ):
        """
        Called inside validation_step.

        Do not self.log() validation metrics here.

        Reason:
            With two validation dataloaders, this Lightning version
            renames validation-step metrics like:

                dice -> dice/dataloader_idx_0

            The checkpoint monitors plain "dice", so validation
            metrics are logged later in metric_log_validation_epoch_metrics().
        """

        # Main Dice = foreground mean.
        main_mean = metrics["foreground_mean"]

        values = {
            "dice": main_mean["dice"],
            "asd_mm": main_mean["asd_mm"],
            "hd_mm": main_mean["hd_mm"],
            "hd95_mm": main_mean["hd95_mm"],
        }

        for key, value in values.items():
            value = safe_float(value)

            if value is not None:
                self.step_metrics[key].update(value)

    def metric_print_validation_epoch_metrics(
        self,
        synced_metrics,
        stage="val",
    ):
        """
        Print final epoch metrics explicitly.

        This is useful in test/predict mode because the final tqdm line may
        belong to an unlabeled prediction dataloader.
        """

        if not getattr(self.trainer, "is_global_zero", True):
            return

        dice = safe_float(synced_metrics["dice"])
        asd = safe_float(synced_metrics["asd_mm"])
        hd = safe_float(synced_metrics["hd_mm"])
        hd95 = safe_float(synced_metrics["hd95_mm"])

        print()
        print("=" * 80)
        print(f"[{stage}] segmentation metrics")
        print("=" * 80)

        if dice is not None:
            print(f"dice    : {dice:.4f}")

        if asd is not None:
            print(f"asd_mm  : {asd:.4f}")

        if hd is not None:
            print(f"hd_mm   : {hd:.4f}")

        if hd95 is not None:
            print(f"hd95_mm : {hd95:.4f}")

        print("=" * 80)
        print()
        
    def metric_log_validation_epoch_metrics(
        self,
        synced_metrics,
    ):
        """
        Log validation metrics once at validation epoch end.

        This produces plain metric names:
            dice
            asd_mm
            hd_mm
            hd95_mm
        """

        for key in [
            "dice",
            "asd_mm",
            "hd_mm",
            "hd95_mm",
        ]:
            value = safe_float(synced_metrics[key])

            if value is not None:
                self._metric_log_scalar(
                    name=key,
                    value=value,
                    prog_bar=True,
                    logger=False,
                    sync_dist=False,
                )

    def _metric_update_epoch_metrics(self, synced_metrics):
        epoch_value = safe_float(self.current_epoch)

        if epoch_value is not None:
            self.epoch_metrics["epoch"].update(
                torch.tensor(
                    [epoch_value],
                    dtype=torch.float32,
                )
            )

        for key in self.tracked_metric_keys:
            value = safe_float(synced_metrics[key])

            if value is not None:
                self.epoch_metrics[key].update(
                    torch.tensor(
                        [value],
                        dtype=torch.float32,
                    )
                )

    def _metric_save_training_progress_plot(self):
        history = self.epoch_metrics.compute()

        history_np = {
            key: value.detach().cpu().numpy()
            for key, value in history.items()
        }

        save_training_progress_plot(
            history=history_np,
            progress_png_file=Path(self.actual_validation_output_base)
            / "training_progress.png",
            dataset_name=self.dataset_name,
            fold=self.cfg.fold,
        )
