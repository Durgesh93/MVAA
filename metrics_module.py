"""
Metric tracking and progress plotting utilities for SSL nnU-Net LightningModule.

Contains:
- _init_metric_tracking
- _extract_validation_scores
- _update_progress_metrics
- _save_training_progress_plot
"""


import math
import torch
from pathlib import Path

from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.aggregation import CatMetric

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import save_training_progress_plot


class MetricsMixin:
    # ------------------------------------------------------------------
    # Init metrics
    # ------------------------------------------------------------------
    def _init_metric_tracking(self):
        """
        Initialize synced loss metrics and rank-0 progress history metrics.

        Call this inside LightningModule.__init__ after super().__init__().
        """

        # Synced epoch averages across DDP ranks
        self.loss_metrics = MetricCollection(
            {
                "train_loss": MeanMetric(sync_on_compute=True),
                "train_sup_loss": MeanMetric(sync_on_compute=True),
                "train_pseudo_loss": MeanMetric(sync_on_compute=True),
                "val_loss": MeanMetric(sync_on_compute=True),
            }
        )

        # Rank-0 history for plotting
        self.progress_metrics = MetricCollection(
            {
                "epoch": CatMetric(sync_on_compute=False),
                "train_loss": CatMetric(sync_on_compute=False),
                "train_sup_loss": CatMetric(sync_on_compute=False),
                "train_pseudo_loss": CatMetric(sync_on_compute=False),
                "val_loss": CatMetric(sync_on_compute=False),
                "dice": CatMetric(sync_on_compute=False),
                "asd_mm": CatMetric(sync_on_compute=False),
                "hd_mm": CatMetric(sync_on_compute=False),
                "hd95_mm": CatMetric(sync_on_compute=False),
            }
        )

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _tensor_value(self, value):
        if value is None:
            value = float("nan")

        return torch.tensor(
            [float(value)],
            device=self.device,
            dtype=torch.float32,
        )

    def _extract_validation_scores(self, metrics):
        """
        Extract foreground/overall validation metrics from metrics.json output.

        Supports:
        - foreground_mean
        - overall_mean
        """

        if metrics is None:
            metrics = {}

        summary = metrics.get(
            "overall_mean",
            metrics.get("foreground_mean", {}),
        )

        return {
            "dice": summary.get("Dice"),
            "asd_mm": summary.get("ASD_mm"),
            "hd_mm": summary.get("HD_mm"),
            "hd95_mm": summary.get("HD95_mm"),
        }

    # ------------------------------------------------------------------
    # Progress history collection
    # ------------------------------------------------------------------
    def _update_progress_metrics(self, epoch_losses, validation_scores):
        """
        Store one epoch of losses and actual-validation metrics.
        Only call this on global rank 0.
        """

        self.progress_metrics["epoch"].update(
            self._tensor_value(self.current_epoch)
        )

        self.progress_metrics["train_loss"].update(
            epoch_losses["train_loss"].reshape(1)
        )

        self.progress_metrics["train_sup_loss"].update(
            epoch_losses["train_sup_loss"].reshape(1)
        )

        self.progress_metrics["train_pseudo_loss"].update(
            epoch_losses["train_pseudo_loss"].reshape(1)
        )

        self.progress_metrics["val_loss"].update(
            epoch_losses["val_loss"].reshape(1)
        )

        self.progress_metrics["dice"].update(
            self._tensor_value(validation_scores.get("dice"))
        )

        self.progress_metrics["asd_mm"].update(
            self._tensor_value(validation_scores.get("asd_mm"))
        )

        self.progress_metrics["hd_mm"].update(
            self._tensor_value(validation_scores.get("hd_mm"))
        )

        self.progress_metrics["hd95_mm"].update(
            self._tensor_value(validation_scores.get("hd95_mm"))
        )

    def _save_training_progress_plot(self):
        """
        Save training_progress.png.

        Only call this on rank 0.
        """

        history = self.progress_metrics.compute()

        history_np = {
            key: value.detach().cpu().numpy()
            for key, value in history.items()
        }

        save_training_progress_plot(
            history=history_np,
            progress_png_file= Path(self.actual_validation_output_base) / 'training_progress',
            dataset_name=self.dataset_name,
            fold=self.fold,
        )
