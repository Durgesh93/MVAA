"""
Converts raw video frame PNGs into nnU-Net's per-channel nii.gz
representation on the fly, at inference time -- replicating
data_preparation/data_task3_VIDEO_nnunet.py's
write_video_rgb_image_as_nnunet_channels (same RGB-split and
orientation-normalize logic) without the full data-prep pipeline. Temp
channel files go to work_dir and are safe to discard after preprocessing.

ASSUMPTION, not yet verified against the real hidden test set: /input's
video task is raw per-frame PNGs grouped one subfolder per recording
(mirroring /output's REC_xxx/ convention) -- confirmed as PNG per
data_task3_VIDEO_nnunet.py's collect_video_files, but the nii.gz-per-channel
form is only how *our own* training pipeline stores it after conversion,
not necessarily how the organizers hand us raw frames.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


def align_video_spatial_shape(arr, target_format="hw"):
    larger_axis_first = arr.shape[0] >= arr.shape[1]
    want_larger_first = target_format == "hw"

    if larger_axis_first != want_larger_first:
        return arr.swapaxes(0, 1)

    return arr


def write_frame_as_nnunet_channels(png_path, case_id, work_dir):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(png_path).convert("RGB")
    arr = np.asarray(image)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got shape {arr.shape}: {png_path}")

    arr = align_video_spatial_shape(arr, target_format="hw")

    affine = np.eye(4)

    channel_paths = []

    for c in range(3):
        channel = arr[:, :, c].astype(np.uint8)

        nii = nib.Nifti1Image(channel, affine=affine)
        nii.set_data_dtype(np.uint8)

        channel_path = work_dir / f"{case_id}_{c:04d}.nii.gz"
        nib.save(nii, str(channel_path))

        channel_paths.append(str(channel_path))

    return channel_paths


def discover_video_frames(image_folder):
    """
    Map each case_id to its source PNG path, one per REC_xxx/ recording
    subfolder. case_id is '<rec_id>_<frame_stem>' unless the frame's own
    filename already starts with the recording id.
    """
    image_folder = Path(image_folder)

    frames = {}

    for rec_dir in sorted(p for p in image_folder.iterdir() if p.is_dir()):
        for png_path in sorted(rec_dir.glob("*.png")):
            stem = png_path.stem
            case_id = stem if stem.startswith(rec_dir.name) else f"{rec_dir.name}_{stem}"

            frames[case_id] = png_path

    return frames
