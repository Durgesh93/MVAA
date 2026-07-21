"""
Visual sanity check for the real TrL/TrU training batch produced by
SSLnnUNetDataModule.train_dataloader() -- pulls one batch, takes item 0,
and dumps it as PNG(s) for a manual look: the labeled sample, the TrU weak
view, and each TrU strong view (1..K-1). 3D tasks (CT, TEE) dump every
z-slice as its own PNG; the 2D task (video) dumps a single PNG.

CUDA is force-disabled below: this script never runs the network, only
the dataloader, but SSLnnUNetDataModule.pin_memory defaults to
torch.cuda.is_available() -- if that's True, the augmenter's background
thread calls .pin_memory() on every returned tensor, which needs a CUDA
context even though nothing here does GPU compute. Hiding the device
avoids fighting a real training job for it.

Usage:
    python scripts/check_transforms_visual.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from config import build_config
from utils import set_nnunet_env
from datamodule import SSLnnUNetDataModule

OUT_DIR = Path(__file__).resolve().parent / "transform_check"

CONFIG_MAP = {"ct": "experiment_CT", "tee": "experiment_TEE", "video": "experiment_video"}


def to_uint8_png(img_chw, out_path, vmin, vmax):
    """
    vmin/vmax must be shared across everything dumped for one view (all of
    its z-slices for a 3D task) -- per-slice min-max normalization would
    independently re-stretch each slice back to full contrast, hiding
    exactly the brightness/contrast structure a manual look is meant to
    catch.
    """
    arr = img_chw.detach().cpu().numpy()
    arr = (arr - vmin) / (vmax - vmin + 1e-8)
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)

    if arr.shape[0] == 1:
        Image.fromarray(arr[0], mode="L").save(out_path)
    else:
        Image.fromarray(np.transpose(arr, (1, 2, 0)), mode="RGB").save(out_path)


def dump_view(tensor, out_dir, name):
    """
    tensor is a single sample with no batch dim: (C, H, W) for the 2D
    task, or (C, D, H, W) for a 3D task. Dumps one PNG for 2D, or one PNG
    per z-slice for 3D -- all slices of this one view share a single
    vmin/vmax.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    vmin, vmax = tensor.min().item(), tensor.max().item()

    if tensor.ndim == 3:
        to_uint8_png(tensor, out_dir / f"{name}.png", vmin, vmax)
        return 1

    num_slices = tensor.shape[1]
    for d in range(num_slices):
        to_uint8_png(tensor[:, d], out_dir / f"{name}_slice{d:03d}.png", vmin, vmax)
    return num_slices


def dump_task_batch(task_name, config_name):
    cfg = build_config(config_name=config_name, overrides=["fold=all"])
    set_nnunet_env(cfg)

    dm = SSLnnUNetDataModule(cfg.datamodule)
    dm.trainer = SimpleNamespace(world_size=1, global_rank=0, limit_train_batches=None)
    dm.setup()

    loader = dm.train_dataloader()
    batch, batch_idx, dataloader_idx = next(iter(loader))

    labeled_item = batch["labeled"]["data"][0]
    view_items = [v[0] for v in batch["unlabeled"]["data_views"]]

    task_dir = OUT_DIR / task_name

    n = dump_view(labeled_item, task_dir, "labeled")

    view_names = ["weak"] + [f"strong{k}" for k in range(1, len(view_items))]
    for view_name, view_item in zip(view_names, view_items):
        dump_view(view_item, task_dir, view_name)

    dims = "3D" if labeled_item.ndim == 4 else "2D"
    print(
        f"[{task_name}] {dims}, K={len(view_items)}, shape={tuple(labeled_item.shape)} "
        f"-> {n} file(s)/view dumped to {task_dir}"
    )


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for task_name, config_name in CONFIG_MAP.items():
        dump_task_batch(task_name, config_name)

    print(f"\nAll PNGs written under {OUT_DIR}")
