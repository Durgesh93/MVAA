"""
MetricsTracker for SSL nnU-Net LightningModule.

Tracks training losses and computes validation segmentation metrics.
compute_metrics() is called only in validation_step after
nnunet.run_prediction().

Standalone component: owns its own step_metrics/epoch_metrics state
plus the label_manager (injected once at construction, since it's
static for the LightningModule's lifetime like pm/cm/lm are for
NNUnetSetup). Pure tracking/computation only -- printing, self.log-ing,
and writing the progress plot to disk are Lightning/IO concerns and
live directly on SSLnnUNetLightningModule instead (see
lightning_module.py), which reads this tracker's state via
compute_step_metrics()/compute_epoch_history().
"""

import numpy as np
import torch

from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.aggregation import CatMetric

from utils import safe_binary_segmentation_metrics, to_numpy


class MetricsTracker:
    def __init__(self, label_manager):
        self.lm = label_manager
        self.tracked_metric_keys = ["train_loss", "train_sup_loss", "dice", "asd_mm", "hd_mm", "hd95_mm"]
        self.epoch_metric_keys = ["epoch", *self.tracked_metric_keys]
        self.step_metrics = MetricCollection(
            {key: MeanMetric(sync_on_compute=True) for key in self.tracked_metric_keys}
        )
        self.epoch_metrics = MetricCollection({key: CatMetric(sync_on_compute=False) for key in self.epoch_metric_keys})

    def compute_step_metrics(self):
        return self.step_metrics.compute()

    def reset_step_metrics(self):
        self.step_metrics.reset()

    def compute_metrics(self, prediction, voxel_spacing):
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

        voxel_spacing = to_numpy(voxel_spacing)
        voxel_spacing = np.asarray(voxel_spacing).reshape(-1)
        voxel_spacing = tuple(float(x) for x in voxel_spacing)

        # MedPy requires len(voxel_spacing) == pred.ndim.
        # For 2D masks, spacing may still be [z, y, x],
        # so use the last two values [y, x].
        if len(voxel_spacing) != pred.ndim:
            voxel_spacing = voxel_spacing[-pred.ndim :]

        # Overall = background + foreground. Background is normally label 0.
        foreground_labels = list(self.lm.foreground_labels)
        all_labels = [0] + foreground_labels

        classwise = {}
        for label in all_labels:
            pred_mask = pred == label
            gt_mask = gt == label
            scores = safe_binary_segmentation_metrics(pred_mask=pred_mask, gt_mask=gt_mask, voxel_spacing=voxel_spacing)
            classwise[str(label)] = {
                "dice": scores["Dice"],
                "asd_mm": scores["ASD_mm"],
                "hd_mm": scores["HD_mm"],
                "hd95_mm": scores["HD95_mm"],
            }

        metric_names = self.tracked_metric_keys[2:]

        # foreground_labels is always non-empty for a real segmentation
        # config, and classwise[label][metric_name] is always a finite
        # float (safe_binary_segmentation_metrics uses a fixed penalty
        # distance instead of None for missed/hallucinated classes).
        foreground_mean = {}
        for metric_name in metric_names:
            values = [classwise[str(label)][metric_name] for label in foreground_labels]
            foreground_mean[metric_name] = float(np.mean(values))
        return {"foreground_mean": foreground_mean}

    def update_step_training_metrics(self, train_loss, train_sup_loss):
        values = {"train_loss": train_loss, "train_sup_loss": train_sup_loss}
        for key, value in values.items():
            self.step_metrics[key].update(value)

    def update_step_val_metrics(self, metrics):
        main_mean = metrics["foreground_mean"]
        values = {
            "dice": main_mean["dice"],
            "asd_mm": main_mean["asd_mm"],
            "hd_mm": main_mean["hd_mm"],
            "hd95_mm": main_mean["hd95_mm"],
        }
        for key, value in values.items():
            self.step_metrics[key].update(value)

    def update_epoch_metrics(self, synced_metrics, current_epoch):
        self.epoch_metrics["epoch"].update(torch.tensor([current_epoch], dtype=torch.float32))
        for key in self.tracked_metric_keys:
            self.epoch_metrics[key].update(torch.tensor([synced_metrics[key]], dtype=torch.float32))

    def compute_epoch_history(self):
        history = self.epoch_metrics.compute()
        return {key: value.detach().cpu().numpy() for key, value in history.items()}
