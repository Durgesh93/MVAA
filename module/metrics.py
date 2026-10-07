"""
MetricsTracker for SSL nnU-Net LightningModule.

Tracks training losses and computes validation segmentation metrics.
compute_metrics() is called only in validation_step after
nnunet.run_prediction().

nn.Module, registered as self.metrics on SSLnnUNetLightningModule, so
it moves with the rest of the model whenever Lightning calls .to(device)
-- step_metrics uses sync_on_compute=True, which runs a cross-rank
torch.distributed.all_gather inside compute(), and DDP's NCCL backend
only supports CUDA tensors for that, never CPU. Letting Lightning's own
device placement handle this (rather than a manual .to(device) call
before every update) is the same "boring", idiomatic way any other
torchmetrics-based Metric is wired into a LightningModule.

A per-epoch history of these scalars used to be accumulated here too, in
an nn.ModuleDict of CatMetric so it rode along in the checkpoint, purely
to redraw training_progress.png each validation epoch. W&B tracks the same
scalars, so both the plot and the history are gone.

Printing and self.log-ing are Lightning/IO concerns and live directly on
SSLnnUNetLightningModule instead (see lightning_module.py), which reads
this tracker's state via compute_step_metrics().
"""

import numpy as np
from torch import nn

from torchmetrics import MeanMetric, MetricCollection

from utils import safe_binary_segmentation_metrics, to_numpy


class MetricsTracker(nn.Module):
    def __init__(self, tracked_labels):
        """
        tracked_labels: list of (label_value, label_name) pairs (e.g.
        [(1, "class_1")]) -- the dice/asd/hd/hd95-relevant classes for
        this task, read from litmodule_cfg.tracked_labels via
        lightning_module.py. Every tracked scalar metric (dice, asd_mm,
        hd_mm, hd95_mm) is a mean over just these classes since that's
        what checkpoint.monitor selects on; dice_<name> additionally
        tracks each class individually for the classwise plot.
        """
        super().__init__()

        self.tracked_labels = tracked_labels
        self.dice_keys = [f"dice_{name}" for _, name in tracked_labels]
        self.tracked_metric_keys = [
            "train_loss",
            "train_sup_loss",
            "train_cm_loss",
            "train_cm_weight",
            "train_cm_confident_frac",
            "dice",
            *self.dice_keys,
            "asd_mm",
            "hd_mm",
            "hd95_mm",
        ]
        self.step_metrics = MetricCollection(
            {key: MeanMetric(sync_on_compute=True) for key in self.tracked_metric_keys}
        )

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

        tracked_values = [label for label, _ in self.tracked_labels]

        classwise = {}
        for label in tracked_values:
            pred_mask = pred == label
            gt_mask = gt == label
            scores = safe_binary_segmentation_metrics(pred_mask=pred_mask, gt_mask=gt_mask, voxel_spacing=voxel_spacing)
            classwise[str(label)] = {
                "dice": scores["Dice"],
                "asd_mm": scores["ASD_mm"],
                "hd_mm": scores["HD_mm"],
                "hd95_mm": scores["HD95_mm"],
            }

        tracked_mean = {}
        for metric_name in ("asd_mm", "hd_mm", "hd95_mm"):
            values = [classwise[str(label)][metric_name] for label in tracked_values]
            tracked_mean[metric_name] = float(np.mean(values))

        tracked_dice = {name: classwise[str(label)]["dice"] for label, name in self.tracked_labels}
        dice_mean_tracked = float(np.mean(list(tracked_dice.values())))

        return {"tracked_mean": tracked_mean, "tracked_dice": tracked_dice, "dice_mean_tracked": dice_mean_tracked}

    def update_step_training_metrics(
        self,
        train_loss,
        train_sup_loss=None,
        train_cm_loss=None,
        train_cm_weight=None,
        train_cm_confident_frac=None,
    ):
        """
        Only train_loss is always present.

        train_cm_confident_frac is None whenever cm_mode sources no
        pseudo-labels (cm_mode="l", the default). A MeanMetric that is never
        updated computes to NaN, which the LightningModule's logging loop
        skips and the progress plot masks out -- so an inactive panel stays
        empty instead of plotting a misleading zero line.
        """

        self.step_metrics["train_loss"].update(train_loss)

        optional = {
            "train_sup_loss": train_sup_loss,
            "train_cm_loss": train_cm_loss,
            "train_cm_weight": train_cm_weight,
            "train_cm_confident_frac": train_cm_confident_frac,
        }

        for key, value in optional.items():
            if value is not None:
                self.step_metrics[key].update(value)

    def update_step_val_metrics(self, metrics):
        tracked_mean = metrics["tracked_mean"]
        values = {
            "dice": metrics["dice_mean_tracked"],
            "asd_mm": tracked_mean["asd_mm"],
            "hd_mm": tracked_mean["hd_mm"],
            "hd95_mm": tracked_mean["hd95_mm"],
        }
        for name, value in metrics["tracked_dice"].items():
            values[f"dice_{name}"] = value

        for key, value in values.items():
            self.step_metrics[key].update(value)
