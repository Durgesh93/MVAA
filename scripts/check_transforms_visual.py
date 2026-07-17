"""
One-off visual sanity check for the new/swapped intensity transforms
(TEE's RicianNoiseTransform swap, video's SmokeHazeTransform and
BleedingBlobTransform) -- dumps before/after PNGs for a manual look.
Not part of the automated check_datamodule.py assertions since "does
this look like plausible haze/bleeding" isn't something a shape/NaN
assertion can verify.

Usage:
    python scripts/check_transforms_visual.py
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import build_config
from utils import set_nnunet_env
from datamodule import SSLnnUNetDataModule
from transforms import RicianNoiseTransform, SmokeHazeTransform, BleedingBlobTransform

OUT_DIR = Path(
    "/cluster/work/projects/nn8104k/dsi014/tmp/claude-240131/"
    "-cluster-work-projects-nn8104k-dsi014-envs/570164b7-2a0b-481a-a0be-19a9f82ee9cc/"
    "scratchpad/transform_check"
)


def to_uint8_png(img_chw, out_path, vmin=None, vmax=None):
    """
    vmin/vmax should be the SAME across a before/after pair (e.g. taken
    from the "before" image) -- per-image min-max normalization would
    independently re-stretch each image back to full contrast, hiding
    exactly the global brightness/contrast shift these transforms are
    meant to show.
    """
    arr = img_chw.detach().cpu().numpy()

    if vmin is None:
        vmin = arr.min()
    if vmax is None:
        vmax = arr.max()

    arr = (arr - vmin) / (vmax - vmin + 1e-8)
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)

    if arr.shape[0] == 1:
        Image.fromarray(arr[0], mode="L").save(out_path)
    else:
        Image.fromarray(np.transpose(arr, (1, 2, 0)), mode="RGB").save(out_path)


def check_tee():
    cfg = build_config(config_name="experiment_TEE", overrides=["fold=all"])
    set_nnunet_env(cfg)
    dm = SSLnnUNetDataModule(cfg.datamodule)

    case_id = sorted(dm.trl_all)[0]
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    ds = infer_dataset_class(dm.folder)(dm.folder, [case_id])
    data, seg, seg_prev, properties = ds.load_case(case_id)

    img = torch.from_numpy(np.asarray(data)).float()
    mid = img.shape[1] // 2
    slice_img = img[:, mid]  # (C, H, W)

    vmin, vmax = slice_img.min().item(), slice_img.max().item()
    to_uint8_png(slice_img, OUT_DIR / "tee_before.png", vmin=vmin, vmax=vmax)

    rician = RicianNoiseTransform(noise_variance=(0.03, 0.05))
    params = rician.get_parameters(slice_img)
    out = rician._apply_to_image(slice_img.clone(), **params)
    to_uint8_png(out, OUT_DIR / "tee_after_rician.png", vmin=vmin, vmax=vmax)

    print(f"TEE case {case_id}, slice shape {tuple(slice_img.shape)} -> dumped tee_before.png / tee_after_rician.png")


def check_video():
    cfg = build_config(config_name="experiment_video", overrides=["fold=all"])
    set_nnunet_env(cfg)
    dm = SSLnnUNetDataModule(cfg.datamodule)

    case_id = sorted(dm.trl_all)[0]
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    ds = infer_dataset_class(dm.folder)(dm.folder, [case_id])
    data, seg, seg_prev, properties = ds.load_case(case_id)

    img = torch.from_numpy(np.asarray(data)).float()
    if img.ndim == 4:
        img = img[:, 0]  # drop dummy z for 2D video

    vmin, vmax = img.min().item(), img.max().item()
    to_uint8_png(img, OUT_DIR / "video_before.png", vmin=vmin, vmax=vmax)

    smoke = SmokeHazeTransform(alpha_range=(0.3, 0.3))  # fixed mid-strength for a clear before/after
    params = smoke.get_parameters(img)
    out_smoke = smoke._apply_to_image(img.clone(), **params)
    to_uint8_png(out_smoke, OUT_DIR / "video_after_smoke.png", vmin=vmin, vmax=vmax)

    # scale/loc forced larger + centered here purely so the effect is
    # visible in a full-frame demo crop -- the actual pipeline uses the
    # narrower scale=(20,80), loc=(-0.2,1.2) from
    # transform_builders.py._build_intensity_transforms_video, which
    # simulates a smaller, off-center bleeding patch (a small droplet
    # anywhere in frame, not necessarily front-and-center).
    bleed = BleedingBlobTransform(scale=(150, 200), loc=(0.4, 0.6), max_strength=(0.9, 0.9))
    params = bleed.get_parameters(img)
    out_bleed = bleed._apply_to_image(img.clone(), **params)
    to_uint8_png(out_bleed, OUT_DIR / "video_after_bleed.png", vmin=vmin, vmax=vmax)

    print(f"Video case {case_id}, slice shape {tuple(img.shape)} -> dumped video_before.png / video_after_smoke.png / video_after_bleed.png")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    check_tee()
    check_video()
    print(f"\nAll PNGs written to {OUT_DIR}")
