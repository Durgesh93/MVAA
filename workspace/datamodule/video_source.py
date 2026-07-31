"""
Converts raw video frame PNGs into nnU-Net's per-channel nii.gz
representation on the fly, at inference time -- replicating
data_preparation/data_task3_VIDEO_nnunet.py's
write_video_rgb_image_as_nnunet_channels (same RGB-split and
orientation-normalize logic) without the full data-prep pipeline. Temp
channel files go to work_dir and are safe to discard after preprocessing.

Which case_id maps to which PNG is resolved from test_cases.json (the
submission contract's manifest), not by walking the directory -- so the
on-disk layout underneath image_folder (e.g. one subfolder per recording)
only has to match whatever relative paths the manifest itself points at.
"""

import json
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
    Map each case_id to its source PNG path plus its output subfolder, per
    the official submission contract's test_cases.json manifest:
        {"cases": [{"case_id": ..., "image": "relative/path"}, ...]}
    "image" is resolved relative to image_folder's parent (/input itself),
    not to image_folder (/input/<prefix>) -- confirmed against a real
    failed hidden-test run (Task 1, same manifest format) whose error
    showed a doubled .../t1_ct/t1_ct/... path when "image" was joined onto
    image_folder directly, meaning "image" already carries the <prefix>/
    segment (e.g. "t3_vid/REC_xxx/frame.png").

    output_subdir strips that same leading <prefix>/ segment before
    carrying the rest (e.g. "REC_xxx") along, so mirroring it under
    /output/<prefix>/ doesn't double it up the same way on the output side.
    """
    image_folder = Path(image_folder)
    manifest_path = image_folder / "test_cases.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing test_cases.json manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())

    input_root = image_folder.parent
    prefix = image_folder.name

    frames = {}
    for entry in manifest["cases"]:
        image_rel = Path(entry["image"])
        within_task = image_rel.relative_to(prefix) if image_rel.parts[:1] == (prefix,) else image_rel
        frames[entry["case_id"]] = {"path": input_root / image_rel, "output_subdir": within_task.parent}

    return frames
