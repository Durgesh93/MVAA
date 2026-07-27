"""
Stage-1 self-supervised pretraining: masked-volume reconstruction.

Builds the exact same nnU-Net encoder/decoder architecture stage 2 uses
(same plans/configuration -> same PlansManager/ConfigurationManager),
except the final 1x1x1 conv regresses num_input_channels (the original
intensities) instead of predicting num_segmentation_heads classes, and
deep supervision is off (a single reconstruction target at full
resolution is enough for this pretext task).

Pretext task: randomly zero out cuboid blocks of each patch (grid-based,
block size / mask ratio configurable), feed the masked patch through the
network, and train it to reconstruct the ORIGINAL full patch -- loss is
computed only over masked voxels (MAE-style), so the network can't just
copy its input through and has to actually learn to infer structure from
context. Labels never enter this stage at all: TrL and TrU cases are
pooled identically in PretrainDataModule.

Stage 2 (SSLnnUNetLightningModule / NNUnetSetup.build_network) loads this
stage's checkpoint via utils.load_pretrained_weights, which keeps every
tensor whose shape matches (the whole encoder/decoder body) and skips
only decoder.seg_layers.* (shape differs: num_input_channels here vs.
num_segmentation_heads there).
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import lightning as L

from batchgenerators.utilities.file_and_folder_operations import join, load_json

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

from utils import to_tensor
from utils import save_pretrain_progress_plot as write_pretrain_progress_plot


class PretrainLightningModule(L.LightningModule):
    def __init__(self, litmodule_cfg):
        super().__init__()

        self.cfg = litmodule_cfg

        self.mask_ratio = float(self.cfg.mask_ratio)
        self.mask_block_size = int(self.cfg.mask_block_size)

        self.dataset_name = maybe_convert_to_dataset_name(self.cfg.dataset_id)
        base = join(nnUNet_preprocessed, self.dataset_name)
        plans = load_json(join(base, self.cfg.plans_identifier + ".json"))
        dataset_json = load_json(join(base, "dataset.json"))

        self.pm = PlansManager(plans)
        self.cm = self.pm.get_configuration(self.cfg.configuration)
        self.num_input_channels = determine_num_input_channels(self.pm, self.cm, dataset_json)

        # Read from os.environ at construction time, not imported from
        # nnunetv2.paths at module level -- see the matching comment in
        # lightning_module.py for why the module-level import goes stale
        # once both LightningModule classes have been imported in the same
        # process.
        self.fold_output_folder = (
            Path(os.environ["nnUNet_results"])
            / self.dataset_name
            / (self.cfg.plans_identifier + "__" + self.cfg.configuration)
            / f"fold_{self.cfg.fold}"
        )
        self.progress_png_file = self.fold_output_folder / "training_progress.png"

        self._epoch_history = {"epoch": [], "train_loss": [], "val_loss": []}
        self._train_losses = []
        self._val_losses = []

        self.network = None

    def setup(self, stage=None):
        # Reconstruction head: num_input_channels in, num_input_channels
        # out, no deep supervision -- everything upstream of the final
        # conv (encoder + decoder stages/transpconvs) has shapes that
        # only depend on the plan, not on this output-channel choice, so
        # it's byte-for-byte compatible with stage 2's own network for
        # every key except decoder.seg_layers.*.
        self.network = nnUNetTrainer.build_network_architecture(
            self.pm, self.cm, self.num_input_channels, self.num_input_channels, enable_deep_supervision=False
        )

    def forward(self, x):
        return self.network(x)

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------
    def _block_mask(self, batch_size, spatial_shape, device):
        """
        Coarse per-sample block mask, upsampled (nearest) to full patch
        resolution. Independent random blocks per sample in the batch --
        a shared mask across the batch would make the pretext task
        trivially predictable of held-out slabs across samples.
        """

        grid_shape = [max(1, s // self.mask_block_size) for s in spatial_shape]
        num_blocks = int(np.prod(grid_shape))
        num_mask = max(1, int(round(num_blocks * self.mask_ratio)))

        coarse = torch.zeros(batch_size, num_blocks, device=device)
        for b in range(batch_size):
            idx = torch.randperm(num_blocks, device=device)[:num_mask]
            coarse[b, idx] = 1.0

        coarse = coarse.view(batch_size, 1, *grid_shape)
        mask = F.interpolate(coarse, size=spatial_shape, mode="nearest")
        return mask

    def _masked_reconstruction_loss(self, pred, target, mask):
        diff = (pred - target).abs() * mask
        return diff.sum() / mask.sum().clamp(min=1.0)

    def _step(self, batch):
        data = to_tensor(batch["data"], device=self.device, dtype=torch.float32)
        mask = self._block_mask(data.shape[0], data.shape[2:], data.device)
        masked_input = data * (1 - mask)
        pred = self.network(masked_input)
        return self._masked_reconstruction_loss(pred, data, mask)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self._train_losses.append(loss.detach())
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=False, batch_size=1
        )
        return loss

    def validation_step(self, batch, batch_idx):
        if self.trainer.sanity_checking:
            return None

        loss = self._step(batch)
        self._val_losses.append(loss.detach())
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=False, batch_size=1
        )
        return loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        train_loss = torch.stack(self._train_losses).mean().item() if self._train_losses else float("nan")
        val_loss = torch.stack(self._val_losses).mean().item() if self._val_losses else float("nan")

        self._epoch_history["epoch"].append(self.current_epoch)
        self._epoch_history["train_loss"].append(train_loss)
        self._epoch_history["val_loss"].append(val_loss)

        self._train_losses = []
        self._val_losses = []

        if self.trainer.is_global_zero:
            write_pretrain_progress_plot(
                history=self._epoch_history,
                progress_png_file=self.progress_png_file,
                dataset_name=self.dataset_name,
                fold=self.cfg.fold,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.cfg.num_epochs, power=0.9)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1}}
