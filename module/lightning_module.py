import os
from pathlib import Path
from typing import Any, Dict

import torch
import lightning as L

from .metrics import MetricsTracker
from .ddp import DDPHelper
from .nnunet import NNUnetSetup

from utils import get_train_batch_data_target, to_tensor
from utils import save_training_progress_plot as write_training_progress_plot


class SSLnnUNetLightningModule(L.LightningModule):
    def __init__(self, litmodule_cfg):
        super().__init__()

        self.cfg = litmodule_cfg
        self.enable_deep_supervision = True

        self.nnunet = NNUnetSetup(
            litmodule_cfg, enable_deep_supervision=self.enable_deep_supervision, trainer_name=self.__class__.__name__
        )
        self.dataset_name = self.nnunet.dataset_name
        self.dataset_json = self.nnunet.dataset_json

        self.tracked_labels = [(self.dataset_json["labels"][name], name) for name in self.cfg.tracked_labels]
        self.all_labels = sorted(
            ((label_id, name) for name, label_id in self.dataset_json["labels"].items()), key=lambda item: item[0]
        )

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
        self.progress_png_file = self.output_folder / "training_progress.png"

        self.ddp = DDPHelper()
        self.metrics = MetricsTracker(tracked_labels=self.tracked_labels, all_labels=self.all_labels)

        # Built here (not in `setup()`) so the network exists before any
        # callback's own `setup()` hook runs -- Lightning's _call_setup_hook
        # calls callback setup() before LightningModule.setup(), and
        # StochasticWeightAveraging.setup() takes `deepcopy(pl_module)` as
        # its averaging base. With self.network still None at that point,
        # the copy had zero registered parameters, so SWA's per-epoch
        # update_parameters() (a zip() over empty parameter iterators) was
        # silently a no-op the whole run -- swa.ckpt ended up with an empty
        # state_dict. build_network() has no trainer/device dependency, so
        # it's safe to construct immediately.
        self.network = self.nnunet.build_network()
        self.loss = None
        self.pseudo_loss_fn = None

    def setup(self, stage=None):
        is_ddp = int(self.trainer.world_size) > 1
        self.loss = self.nnunet.build_loss(is_ddp=is_ddp)
        self.pseudo_loss_fn = self.nnunet.build_pseudo_loss()

    def forward(self, x):
        return self.network(x)

    def on_train_epoch_start(self):
        self.nnunet.update_boundary_weight(self.loss, self.current_epoch)

    def _supervised_loss(self, batch: Dict[str, Any]):
        data, target = get_train_batch_data_target(batch, device=self.device)
        output = self.network(data)
        loss = self.loss(output, target)
        return loss, output, target

    def _pseudo_loss(self, unlabeled_batch: Dict[str, Any]):
        """
        FixMatch-style consistency loss: data_views[0] is the weak
        (geometric-only) view, forwarded under no_grad through the
        unwrapped network purely to source a pseudo-label/confidence mask
        (no backward, so DDP gradient sync is untouched by this call).
        data_views[1:] are strong (weak + intensity aug) views, forwarded
        with grad through self.network and trained to match that label.

        Deep supervision is toggled off (mirroring
        PredictionOps._predict_logits's unwrap/restore pattern) since the
        pseudo-label only exists at one resolution -- but any forward that
        needs gradients must go through self.network directly (never the
        unwrapped `.module`), or DDP gradient sync breaks.
        """
        if self.current_epoch < self.cfg.pseudo_warmup_epochs:
            return self._zero_pseudo()

        net = self.network.module if hasattr(self.network, "module") else self.network
        old_deep_supervision = net.decoder.deep_supervision
        net.decoder.deep_supervision = False
        try:
            weak_data = to_tensor(unlabeled_batch["data_views"][0], device=self.device, dtype=torch.float32)
            with torch.no_grad():
                weak_logits = net(weak_data)

            strong_logits_list = [
                self.network(to_tensor(view, device=self.device, dtype=torch.float32))
                for view in unlabeled_batch["data_views"][1:]
            ]

            return self.pseudo_loss_fn(weak_logits, strong_logits_list)
        finally:
            net.decoder.deep_supervision = old_deep_supervision

    def _zero_pseudo(self):
        """Inactive pseudo-loss: zero loss, zero confident fraction, all-NaN per class."""
        zero = torch.zeros((), device=self.device)
        nan_per_class = torch.full((len(self.all_labels),), float("nan"), device=self.device)
        return zero, zero, nan_per_class

    def training_step(self, batch, batch_idx):
        sup_loss, _, _ = self._supervised_loss(batch["labeled"])

        # At labeled_fraction=1.0 the datamodule builds no unlabeled
        # loader at all, so the CombinedLoader yields only "labeled" and
        # this reduces to plain supervised training -- the baseline an
        # SSL run has to beat, through the identical code path.
        if "unlabeled" in batch:
            pseudo_loss, pseudo_confident_frac, pseudo_confident_frac_per_class = self._pseudo_loss(
                batch["unlabeled"]
            )
        else:
            pseudo_loss, pseudo_confident_frac, pseudo_confident_frac_per_class = self._zero_pseudo()
        total_loss = sup_loss + self.cfg.lambda_pseudo * pseudo_loss
        self.metrics.update_step_training_metrics(
            train_loss=total_loss.detach(),
            train_sup_loss=sup_loss.detach(),
            train_pseudo_loss=pseudo_loss.detach(),
            train_pseudo_confident_frac=pseudo_confident_frac.detach(),
            train_pseudo_confident_frac_per_class=pseudo_confident_frac_per_class.detach(),
        )
        return total_loss

    def _eval_step(self, batch, batch_idx, subfolder):
        """
        Sliding-window inference on one raw case, written as a zip and
        scored against its ground truth.

        Validation and test differ only in which subfolder the zip lands
        in. Both are scored: MVSeg2023 releases the test split labeled,
        so there is no longer an unscored prediction-only path.
        """

        prediction = self.nnunet.predictor.run_prediction(
            network=self.network, device=self.device, batch=batch, batch_idx=batch_idx
        )

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

        return self._eval_step(batch, batch_idx, "validation")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, "test")

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

        metric_keys = ["dice", *self.metrics.dice_keys, "asd_mm", "hd_mm", "hd95_mm"]
        for key in metric_keys:
            value = synced_metrics[key]
            self.log(
                key, value, on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=False, batch_size=1
            )
            if do_print:
                print(f"{key:<12} : {value:.4f}")

        if do_print:
            print("=" * 80)
            print()
        self.metrics.update_epoch_metrics(synced_metrics=synced_metrics, current_epoch=self.current_epoch)
        if self.trainer.is_global_zero and save_training_progress:
            write_training_progress_plot(
                history=self.metrics.compute_epoch_history(),
                progress_png_file=self.progress_png_file,
                dataset_name=self.dataset_name,
                dice_classwise_keys=self.metrics.dice_keys,
                pseudo_confident_frac_classwise_keys=self.metrics.pseudo_confident_frac_keys,
            )
        self.metrics.reset_step_metrics()
        self.ddp.merge_rank_outputs(trainer=self.trainer, output_folder=self.output_folder)

    def on_validation_epoch_end(self):
        self._eval_epoch_end(stage="validation", save_training_progress=True)

    def on_test_epoch_end(self):
        self._eval_epoch_end(stage="test", save_training_progress=False, print_val_metrics=True)

    def configure_optimizers(self):
        optimizer, scheduler = self.nnunet.build_optimizer_and_scheduler(self.network)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1}}
