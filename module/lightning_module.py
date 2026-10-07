import math
import os
from pathlib import Path
from typing import Any, Dict

import torch
import lightning as L

from .adapter import embeddings_enabled
from .metrics import MetricsTracker
from .ddp import DDPHelper
from .nnunet import NNUnetSetup

from utils import get_train_batch_data_target, to_tensor


class SSLnnUNetLightningModule(L.LightningModule):
    def __init__(self, litmodule_cfg):
        super().__init__()

        self.cfg = litmodule_cfg

        # Deep supervision is unconditional -- see NNUnetSetup.__init__. The
        # network returns one head per decoder stage and the datamodule one
        # target per stage to match; the clustering loss reads stage 0 only.
        self.nnunet = NNUnetSetup(litmodule_cfg, trainer_name=self.__class__.__name__)
        self.dataset_name = self.nnunet.dataset_name
        self.dataset_json = self.nnunet.dataset_json

        self.tracked_labels = [(self.dataset_json["labels"][name], name) for name in self.cfg.tracked_labels]
        # Read from os.environ at construction time, not imported from
        # nnunetv2.paths at module level -- that constant is only bound
        # once, on the first `import module` anywhere in the process, so
        # a module-level import could pick up a stale value if something
        # imports `module` before engine.py's set_nnunet_env(cfg) call.
        self.actual_validation_output_base = (
            Path(os.environ["nnUNet_results"])
            / self.dataset_name
            / (self.cfg.plans_identifier + "__" + self.cfg.configuration)
        )

        self.output_folder = self.actual_validation_output_base
        self.actual_validation_output_folder = self.output_folder / "validation"
        self.actual_test_output_folder = self.output_folder / "test"

        self.ddp = DDPHelper()
        self.metrics = MetricsTracker(tracked_labels=self.tracked_labels)

        # Built here rather than in `setup()` so the network exists before
        # any callback's own `setup()` hook runs -- Lightning's
        # _call_setup_hook calls callback setup() before
        # LightningModule.setup(), so a callback that inspects or copies
        # the module's parameters would otherwise see none.
        # build_network() has no trainer/device dependency, so constructing
        # it immediately is safe.
        self.network = self.nnunet.build_network()

        # Built here rather than in setup() for the same reason as the
        # network: it owns the centroid buffers, and Lightning restores a
        # checkpoint onto whatever modules are registered when the restore
        # happens. Created later, the buffers would not exist yet and the
        # saved centroids would be silently dropped on resume.
        self.cm_loss = self.nnunet.build_cm_loss()

        self.loss = None

    def setup(self, stage=None):
        # "Are DDP collectives live", not "is world_size > 1". The two differ
        # at world_size 1 under strategy=ddp, and the difference is fatal:
        # build_loss zeroes the deepest deep-supervision weight when is_ddp is
        # False, which leaves that stage's seg_layer out of the loss entirely.
        # DDP then aborts with "parameters that were not used in producing the
        # loss" -- nnU-Net's 1e-6 instead of 0 exists precisely to keep those
        # parameters in the graph, and it has to key off the wrapper actually
        # being there rather than off how many ranks there are.
        is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        self.loss = self.nnunet.build_loss(is_ddp=is_ddp)

    def forward(self, x):
        return self.network(x)

    def forward_embeddings(self, x):
        """
        (embeddings, logits) for one batch of crops.

        The network is always an EmbeddingAdapter, so this always works. Nothing
        in this branch's training_step calls it yet -- it is the hook a
        clustering-style loss attaches to, in the role pre_heatmaps plays in
        the supercm project.

        Under deep supervision `embeddings` is one tensor per decoder stage,
        highest resolution first; a prototype loss almost certainly wants
        embeddings[0] on its own rather than all five pooled together.

        Goes through self.network, never the unwrapped module, so DDP
        gradient sync survives -- see embeddings_enabled's docstring.
        """
        with embeddings_enabled(self.network):
            return self.network(x)

    @staticmethod
    def _highest_resolution(value):
        """
        The full-resolution element of a deep-supervision list.

        With deep supervision on, the network returns one tensor per decoder
        stage (highest resolution first) and the datamodule returns one
        target per stage to match; with it off, both are bare tensors. None
        passes through so an absent unlabeled stream needs no special case.

        The clustering loss wants stage 0 alone. The coarser stages are
        downsampled views of the same embedding space, so pooling them into
        one set of centroids would weight a 7x6x8 map as heavily as the
        full-resolution one.
        """

        return value[0] if isinstance(value, (list, tuple)) else value

    def _unlabeled_embeddings(self, batch: Dict[str, Any]):
        """
        (embeddings, logits) for one weak view of the unlabeled crops.

        data_views[0] is the geometric-only view. The clustering loss is not
        a consistency loss -- it needs one view of a crop, not a weak/strong
        pair -- so the K-1 strong views go unread and datamodule.K can sit
        at the loader's minimum of 2.

        Deep supervision is switched off around this pass since only stage 0
        feeds the loss, so the coarser embedding maps would be pure waste.
        Safe under DDP because the labeled pass already ran with deep
        supervision on, so every seg_layer still receives a gradient within
        the step. The forward itself goes through self.network, never the
        unwrapped module, or DDP gradient sync breaks -- see
        embeddings_enabled's docstring.
        """

        data = to_tensor(batch["data_views"][0], device=self.device, dtype=torch.float32)

        network = self.network.module if hasattr(self.network, "module") else self.network
        previous = network.decoder.deep_supervision
        network.decoder.deep_supervision = False

        try:
            return self.forward_embeddings(data)
        finally:
            network.decoder.deep_supervision = previous

    def _cm_epoch_progress(self) -> float:
        """
        Fractional epochs elapsed, for the clustering weight's ramp.

        Built from Lightning's own global_step so it survives a resume and
        cannot drift from the optimiser's view of progress, divided by the
        batches per epoch so that cm_warmup_epochs means epochs rather than
        steps. num_training_batches is inf for an unbounded iterable
        dataloader; datamodule.limit_train_batches makes it finite here, but
        fall back to the integer epoch rather than returning inf (which
        would clamp the ramp to full weight on step one).
        """

        batches = float(getattr(self.trainer, "num_training_batches", 0) or 0)

        if not math.isfinite(batches) or batches <= 0:
            return float(self.current_epoch)

        return float(self.global_step) / batches

    def training_step(self, batch, batch_idx):
        """
        Supervised Dice+CE over the labeled crops plus the SuperCM-style
        clustering regularizer on the per-voxel embeddings.

        One code path, always. A purely supervised run is cm_weight=0, not a
        different branch: the centroids are still tracked and the energy
        still logged, it just contributes no gradient. That keeps a baseline
        and a SuperCM run byte-identical in everything but the weight.

        Both terms share ONE forward pass -- forward_embeddings returns the
        embeddings and the logits together -- so the clustering term costs no
        extra trunk evaluation on the labeled crops. A second pass is added
        only when cm_mode asks for unlabeled centroids (u / l+u); at
        cm_mode=l there is none, which makes the whole regularizer about one
        einsum per step.

        batch["unlabeled"] is always present even when unread -- the
        datamodule floors the unlabeled pool at one case so the
        CombinedLoader never drops the key.
        """

        data, target = get_train_batch_data_target(batch["labeled"], device=self.device)

        embeddings, logits = self.forward_embeddings(data)

        supervised_loss = self.loss(logits, target)

        unlabeled_embeddings = unlabeled_logits = None

        if self.cm_loss.needs_unlabeled:
            unlabeled_embeddings, unlabeled_logits = self._unlabeled_embeddings(batch["unlabeled"])

        cm_energy, cm_stats = self.cm_loss(
            embeddings=self._highest_resolution(embeddings),
            logits=self._highest_resolution(logits),
            target=self._highest_resolution(target),
            u_embeddings=self._highest_resolution(unlabeled_embeddings),
            u_logits=self._highest_resolution(unlabeled_logits),
        )

        cm_weight = self.cm_loss.weight_at(self._cm_epoch_progress())

        total_loss = supervised_loss + cm_weight * cm_energy

        self.metrics.update_step_training_metrics(
            train_loss=total_loss.detach(),
            train_sup_loss=supervised_loss.detach(),
            train_cm_loss=cm_stats["cm_loss"],
            train_cm_weight=torch.as_tensor(cm_weight, device=self.device),
            train_cm_confident_frac=cm_stats.get("cm_confident_frac"),
        )

        return total_loss

    def _eval_step(self, batch, batch_idx, subfolder, use_tta, write_case_zip):
        """
        Sliding-window inference on one raw case, written as a zip and
        scored against its ground truth.

        Both stages are scored -- MVSeg2023 releases the test split
        labeled, so there is no unscored prediction-only path. They differ
        in the output subfolder and in TTA: validation runs a single plain
        pass (use_tta=False) because it only has to rank checkpoints,
        while test pays for the configured mirroring and tile overlap to
        get the number worth reporting. See PredictionOps' docstring.
        """

        prediction = self.nnunet.predictor.run_prediction(
            network=self.network,
            device=self.device,
            batch=batch,
            batch_idx=batch_idx,
            use_tta=use_tta,
            need_probs=write_case_zip,
        )

        if write_case_zip:
            rank_output_folder = self.ddp.rank_output_folder(
                trainer=self.trainer, output_folder=self.output_folder
            )

            self.nnunet.predictor.write_prediction_case_zip(
                prediction=prediction,
                zip_dir=rank_output_folder / subfolder,
                include_gt=True,
            )

        metrics = self.metrics.compute_metrics(prediction, voxel_spacing=prediction["gt_properties"]["spacing"])
        self.metrics.update_step_val_metrics(metrics=metrics)

        return prediction

    def validation_step(self, batch, batch_idx):
        if self.trainer.sanity_checking:
            return None

        return self._eval_step(
            batch, batch_idx, "validation", use_tta=False, write_case_zip=bool(self.cfg.save_case_outputs)
        )

    def test_step(self, batch, batch_idx):
        return self._eval_step(
            batch, batch_idx, "test", use_tta=True, write_case_zip=bool(self.cfg.save_case_outputs)
        )

    def _eval_epoch_end(self, stage, merge_case_outputs, print_val_metrics=False):
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

        # The keys stay unprefixed because checkpoint.monitor selects on
        # "dice" by name -- renaming them to "val/dice" here would silently
        # stop ModelCheckpoint from ever finding its monitored metric.
        # validation and test never run in the same process, so the two
        # stages cannot collide on a key.
        metric_keys = ["dice", *self.metrics.dice_keys, "asd_mm", "hd_mm", "hd95_mm"]
        for key in metric_keys:
            value = synced_metrics[key]
            self.log(
                key, value, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False, batch_size=1
            )
            if do_print:
                print(f"{key:<12} : {value:.4f}")

        # Training-side scalars are already aggregated in synced_metrics but
        # were previously only reaching training_progress.png. Send them to
        # the logger too -- not the progress bar, which is full. NaN is
        # skipped rather than logged, since it reads as a gap in the chart.
        for key in self.metrics.tracked_metric_keys:
            if not key.startswith("train_"):
                continue
            value = synced_metrics[key]
            if torch.isnan(value):
                continue
            self.log(
                key, value, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False, batch_size=1
            )

        if do_print:
            print("=" * 80)
            print()

        self.metrics.reset_step_metrics()
        # Only when this stage actually wrote per-case zips: merge_rank_folders
        # raises if _rank_outputs is missing, and nothing creates it when the
        # writes are skipped. Every rank takes the same branch (the flag comes
        # from the shared config), so the barriers inside stay balanced.
        if merge_case_outputs:
            self.ddp.merge_rank_outputs(trainer=self.trainer, output_folder=self.output_folder)

    def on_validation_epoch_end(self):
        self._eval_epoch_end(
            stage="validation",
            merge_case_outputs=bool(self.cfg.save_case_outputs),
        )

    def on_test_epoch_end(self):
        self._eval_epoch_end(
            stage="test",
            print_val_metrics=True,
            merge_case_outputs=bool(self.cfg.save_case_outputs),
        )

    def configure_optimizers(self):
        optimizer, scheduler = self.nnunet.build_optimizer_and_scheduler(self.network)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1}}
