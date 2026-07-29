"""
Utility functions for the Docker inference entrypoint (run_inference.py).

Trimmed to only what's reachable from the inference path -- this file used
to be a 1:1 copy of the training codebase's utils.py (metrics, training-
progress plotting, DDP rank merging, the Slicer case-zip writer, etc.),
none of which run_inference.py/inference_module.py/nnunet.py ever call.
"""

import json
import os
import re
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

# Lets yaml configs reference env vars directly, e.g.
# input_dir: ${env:MVAA_INPUT_DIR,/input}/${prefix} -- optional 2nd arg is
# the default when the var is unset; omit it to require the var (raises).
OmegaConf.register_new_resolver("env", lambda name, default=None: os.environ.get(name, default))

from skimage.io import imsave as skimage_imsave
import SimpleITK as sitk
from scipy import ndimage


def to_numpy(x):
    """
    Convert Torch tensor or NumPy array to NumPy array.
    """

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    if isinstance(x, np.ndarray):
        return x

    return np.asarray(x)


# =============================================================================
# Segmentation -> 0/255 conversion
# =============================================================================
def convert_segmentation_to_255(segmentation):
    # Matches the Codabench baseline scripts' convention: binary mask as
    # plain 0/255 uint8.
    return (segmentation > 0).astype(np.uint8) * 255


# =============================================================================
# Keep-largest-component postprocessing
# =============================================================================
def keep_largest_component(segmentation, foreground_labels):
    """
    For each foreground label, zero out every connected component of that
    label's binary mask except the largest one. Only appropriate for
    labels whose anatomy is a single structure (e.g. CT/TEE valves) --
    not for classes with legitimately multi-component ground truth (e.g.
    video's equipment class, see the commit that removed this postprocessing
    for video).
    """
    out = segmentation.copy()
    for label in foreground_labels:
        mask = segmentation == label
        if not mask.any():
            continue
        components, num_components = ndimage.label(mask)
        if num_components <= 1:
            continue
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        largest_component = sizes.argmax()
        out[mask & (components != largest_component)] = 0
    return out


# =============================================================================
# Segment colors (RGB PNG output when >1 kept class)
# =============================================================================
SEGMENT_COLOR_PALETTE = [
    (1.0, 0.0, 0.0),  # red
    (0.0, 1.0, 0.0),  # green
    (0.2, 0.4, 1.0),  # blue
    (1.0, 1.0, 0.0),  # yellow
    (1.0, 0.0, 1.0),  # magenta
    (0.0, 1.0, 1.0),  # cyan
    (1.0, 0.5, 0.0),  # orange
]


# =============================================================================
# Shared segmentation image I/O
#
# nnU-Net's own image_reader_writer.write_seg() trips into 16-bit output
# whenever a mask's max value is exactly 255
# (np.uint8 if np.max(seg) < 255 else np.uint16). We control the dtype
# ourselves everywhere a segmentation is written, so this always writes
# uint8 directly and skips that writer entirely.
# =============================================================================
class SegmentationImageIO:

    def write_volume(self, image, path):
        sitk.WriteImage(image, str(path), useCompression=True)

    def build_segmentation_image(self, array, reference=None, spacing=None, origin=None, direction=None):
        image = sitk.GetImageFromArray(array.astype(np.uint8, copy=False))

        if reference is not None:
            if image.GetDimension() == reference.GetDimension():
                image.CopyInformation(reference)
        else:
            if spacing is not None:
                image.SetSpacing(spacing)
            if origin is not None:
                image.SetOrigin(origin)
            if direction is not None:
                image.SetDirection(direction)

        return image

    def write_segmentation_file(self, array, path, spacing, origin, direction):
        image = self.build_segmentation_image(array, spacing=spacing, origin=origin, direction=direction)

        self.write_volume(image, path)

    def write_png(self, array, path):
        skimage_imsave(str(path), array.astype(np.uint8, copy=False), check_contrast=False)


segmentation_io = SegmentationImageIO()


# =============================================================================
# Submission prediction writer
# =============================================================================
def write_submission_prediction(
    prediction,
    output_folder,
    configuration_manager,
    convert_to_255=False,
    keep_classes=None,
    output_format="nii.gz",  # "nii.gz" or "png"
    file_ending=".nii.gz",
):
    """
    Write one predict_step output in rank-local submission format.

    Supported output:
        <case_id>-pred.nii.gz
        <case_id>_label_bin.png

    PNG is only allowed for 2D nnU-Net predictions.
    """

    case_id = prediction["case_id"]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_format = str(output_format).lower().lstrip(".")

    if output_format not in {"nii.gz", "png"}:
        raise ValueError(f"Unsupported output_format={output_format}. Use 'nii.gz' or 'png'.")

    segmentation = to_numpy(prediction["predicted_segments"])
    segmentation = np.asarray(segmentation)

    if segmentation.ndim == 4 and segmentation.shape[0] == 1:
        segmentation = segmentation[0]

    # More than one kept class -> RGB PNG, one color per class.
    # Single kept class (or no filtering) -> single-channel {0, 255}, unchanged.
    build_rgb = output_format == "png" and keep_classes is not None and len(keep_classes) > 1

    if build_rgb:
        segmentation = np.where(np.isin(segmentation, keep_classes), segmentation, 0).astype(segmentation.dtype)
    elif keep_classes is not None:
        segmentation = np.isin(segmentation, keep_classes).astype(segmentation.dtype)

    plan_dim = len(configuration_manager.patch_size)

    # ------------------------------------------------------------------
    # NIfTI output
    # ------------------------------------------------------------------
    if output_format == "nii.gz":

        segmentation = segmentation.astype(np.uint8)

        if plan_dim == 2:
            if segmentation.ndim == 2:
                segmentation = segmentation[None, ...]

            elif segmentation.ndim == 3:
                if segmentation.shape[0] != 1:
                    raise ValueError(f"2D nnU-Net writer expects [1, H, W], got {segmentation.shape}")

            else:
                raise ValueError(f"2D nnU-Net writer expects [H, W] or [1, H, W], got {segmentation.shape}")

        elif plan_dim == 3:
            if segmentation.ndim != 3:
                raise ValueError(f"3D nnU-Net writer expects [D, H, W], got {segmentation.shape}")

        else:
            raise ValueError(f"Unsupported nnU-Net plan dimension: {configuration_manager.patch_size}")

        pred_file = output_folder / f"{case_id}-pred{file_ending}"

        tmp_pred_file = output_folder / (f"{case_id}-pred.tmp_{os.getpid()}{file_ending}")

        sitk_stuff = prediction["properties"]["sitk_stuff"]

        # 2D nnU-Net plans carry a dummy leading axis ([1, H, W]) for the
        # sliding-window machinery; a 2D nii.gz file has no such axis.
        segmentation_io.write_segmentation_file(
            segmentation[0] if plan_dim == 2 else segmentation,
            tmp_pred_file,
            spacing=sitk_stuff["spacing"],
            origin=sitk_stuff["origin"],
            direction=sitk_stuff["direction"],
        )

        os.replace(tmp_pred_file, pred_file)

        return pred_file

    # ------------------------------------------------------------------
    # PNG output
    # ------------------------------------------------------------------
    if output_format == "png":

        if plan_dim != 2:
            raise ValueError(
                f"PNG output is only allowed for 2D nnU-Net plans. "
                f"Got patch_size={configuration_manager.patch_size}"
            )

        if build_rgb:
            if segmentation.ndim == 3 and segmentation.shape[0] == 1:
                segmentation = segmentation[0]

            if segmentation.ndim != 2:
                raise ValueError(f"RGB PNG writer expects [H, W], got {segmentation.shape}")

            rgb = np.zeros(segmentation.shape + (3,), dtype=np.uint8)

            for color_idx, class_value in enumerate(keep_classes):
                color = SEGMENT_COLOR_PALETTE[color_idx % len(SEGMENT_COLOR_PALETTE)]

                rgb[segmentation == class_value] = tuple(int(round(c * 255)) for c in color)

            pred_file = output_folder / f"{case_id}_label_bin.png"

            tmp_pred_file = output_folder / (f"{case_id}_label_bin.tmp_{os.getpid()}.png")

            segmentation_io.write_png(rgb, tmp_pred_file)

            os.replace(tmp_pred_file, pred_file)

            return pred_file

        if convert_to_255:
            segmentation = convert_segmentation_to_255(segmentation)
        else:
            segmentation = segmentation.astype(np.uint8)

        if segmentation.ndim == 2:
            segmentation = segmentation[None, ...]

        elif segmentation.ndim == 3:
            if segmentation.shape[0] != 1:
                raise ValueError(f"PNG writer expects [H, W] or [1, H, W], got {segmentation.shape}")

        else:
            raise ValueError(f"PNG writer expects [H, W] or [1, H, W], got {segmentation.shape}")

        pred_file = output_folder / f"{case_id}_label_bin.png"

        tmp_pred_file = output_folder / (f"{case_id}_label_bin.tmp_{os.getpid()}.png")

        segmentation_io.write_png(segmentation[0], tmp_pred_file)

        os.replace(tmp_pred_file, pred_file)

        return pred_file


# =============================================================================
# Inference entrypoint helpers (run_inference.py)
# =============================================================================

VIDEO_CASE_ID_PATTERN = re.compile(r"^(?P<video_id>.+)_(?P<frame>\d{6})$")
NIFTI_SUBMISSION_SUFFIX = "-pred.nii.gz"
VIDEO_SUBMISSION_SUFFIX = "_label_bin.png"


def load_task_cfg(task):
    config_dir = Path(__file__).resolve().parent / "config"
    cfg = OmegaConf.load(config_dir / f"{task}.yaml")
    OmegaConf.resolve(cfg)
    return cfg


def load_checkpoint(module, ckpt_path: Path) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    own_keys = set(module.state_dict().keys())
    state_dict = {key: value for key, value in checkpoint["state_dict"].items() if key in own_keys}

    missing = own_keys - state_dict.keys()

    if missing:
        raise RuntimeError(f"Checkpoint {ckpt_path} is missing expected keys: {sorted(missing)[:5]}")

    module.load_state_dict(state_dict)


def build_prediction_entry(pred_file: Path):
    """
    Return (case_id, relative_path) for one written prediction file, moving
    video frames under a <video_id>/ subfolder to match the contract's
    layout (t3_vid/<video_id>/<frame>_label_bin.png).
    """

    name = pred_file.name

    if name.endswith(VIDEO_SUBMISSION_SUFFIX):
        case_id = name[: -len(VIDEO_SUBMISSION_SUFFIX)]

        match = VIDEO_CASE_ID_PATTERN.match(case_id)

        if match is None:
            raise ValueError(f"Video case_id does not match '<video_id>_<6-digit frame>' pattern: {case_id}")

        video_id = match.group("video_id")

        video_dir = pred_file.parent / video_id
        video_dir.mkdir(parents=True, exist_ok=True)

        moved = video_dir / name
        pred_file.replace(moved)

        return case_id, f"{video_id}/{name}"

    if name.endswith(NIFTI_SUBMISSION_SUFFIX):
        case_id = name[: -len(NIFTI_SUBMISSION_SUFFIX)]

        return case_id, name

    raise ValueError(f"Unrecognized prediction file name: {name}")


def write_predictions_json(pred_files, task_output_dir: Path, task_id: str) -> Path:
    cases = []

    for pred_file in pred_files:
        case_id, relative_path = build_prediction_entry(pred_file)

        relative = Path(relative_path)

        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe segmentation path: {relative_path}")

        cases.append({"case_id": case_id, "segmentation": relative_path})

    json_path = Path(task_output_dir) / f"{task_id}_predictions.json"
    json_path.write_text(json.dumps({"cases": sorted(cases, key=lambda c: c["case_id"])}, indent=2))

    return json_path
