"""
Utility functions for SSL nnU-Net Lightning wrappers.

Rule:
    Only functions imported by datamodule.py, module.py,
    metrics_module.py, and engine.py are top-level.

    No private helper functions.
    No nested helper functions.
"""

import os
import math
import shutil
import zipfile
from pathlib import Path
import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator

from medpy.metric.binary import dc, asd, hd, hd95
from skimage.io import imsave as skimage_imsave
import SimpleITK as sitk
from scipy import ndimage

# =============================================================================
# Basic shared helpers
# =============================================================================


def safe_float(value):
    """
    Convert scalar-like values to a finite Python float.
    """

    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None

        value = value.detach().cpu().float().mean().item()

    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return None

        value = float(np.nanmean(value))

    try:
        value = float(value)
    except Exception:
        return None

    if not np.isfinite(value):
        return None

    return value


def to_tensor(x, device=None, dtype=None, non_blocking=True):
    """
    Convert NumPy array or Torch tensor to Torch tensor.
    """

    if isinstance(x, torch.Tensor):
        tensor = x

    elif isinstance(x, np.ndarray):
        tensor = torch.from_numpy(x)

    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")

    if dtype is not None:
        tensor = tensor.to(dtype=dtype)

    if device is not None:
        tensor = tensor.to(device, non_blocking=non_blocking)

    return tensor


def to_numpy(x):
    """
    Convert Torch tensor or NumPy array to NumPy array.
    """

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    if isinstance(x, np.ndarray):
        return x

    return np.asarray(x)


def get_train_batch_data_target(batch, device):
    """
    Extract (data, target) from an nnU-Net train batch and move to device.

    target is either a Tensor or a list/tuple (deep supervision).
    """

    data = to_tensor(batch["data"], device=device, dtype=torch.float32)

    target = batch["target"]

    if isinstance(target, (list, tuple)):
        target = [to_tensor(t, device=device) for t in target]
    else:
        target = to_tensor(target, device=device)

    return data, target


# =============================================================================
# DDP case splitting
# =============================================================================


def split_by_rank(case_ids, global_rank, world_size):
    """
    Split case IDs by DDP rank.
    """

    case_ids = list(sorted(case_ids))

    world_size = max(1, int(world_size))

    global_rank = int(global_rank)

    if len(case_ids) == 0:
        return []

    if world_size <= 1:
        return case_ids

    if len(case_ids) < world_size:
        return [case_ids[global_rank % len(case_ids)]]

    return case_ids[global_rank::world_size]


# =============================================================================
# Engine / runtime helpers
# =============================================================================


def set_nnunet_env(cfg):
    """
    Set nnU-Net environment variables.
    """

    nnunet_raw = str(cfg.paths.nnunet_raw)
    nnunet_preprocessed = str(cfg.paths.nnunet_preprocessed)
    nnunet_results = str(cfg.paths.nnunet_results)

    os.environ["nnUNet_raw"] = nnunet_raw
    os.environ["nnUNet_preprocessed"] = nnunet_preprocessed
    os.environ["nnUNet_results"] = nnunet_results

    return {"nnunet_raw": nnunet_raw, "nnunet_preprocessed": nnunet_preprocessed, "nnunet_results": nnunet_results}


def override_patch_size(configuration_manager, patch_size_override):
    """
    Override a ConfigurationManager's patch_size in place.

    patch_size is a read-only property backed by the manager's own
    .configuration dict (no setter) -- this mutates that dict directly so
    every downstream reader of .patch_size (data loading's final_patch_size,
    the rotation-margin init_ps calculation, sliding-window inference's
    tile size) picks up the override consistently. Both SSLnnUNetDataModule
    and NNUnetSetup build their own ConfigurationManager from the same
    plans, so both must call this with the same value or training and
    inference would disagree on patch size.

    The new size must stay divisible by the network's per-axis cumulative
    pooling stride (product of architecture.arch_kwargs.strides per axis)
    or skip-connection concatenation between encoder/decoder stages will
    crash on a shape mismatch -- this function does not validate that,
    since it has no network reference to check against.
    """

    if patch_size_override is not None:
        configuration_manager.configuration["patch_size"] = list(patch_size_override)


def resolve_runtime_config(cfg, prediction=False):
    """
    Resolve runtime placeholders:
        devices: ???
        training_strategy: ???
    """

    if torch.cuda.is_available():
        cfg.devices = max(1, torch.cuda.device_count())
    else:
        cfg.devices = 1

    if int(cfg.devices) > 1:
        cfg.training_strategy = "ddp"
    else:
        cfg.training_strategy = "auto"

    cfg.trainer.devices = cfg.devices
    cfg.trainer.strategy = cfg.training_strategy
    return cfg


def resolve_prediction_ckpt(cfg, ckpt):
    ckpt = str(ckpt)

    checkpoint_dir = (
        Path(cfg.paths.nnunet_results)
        / cfg.dataset_id
        / f"{cfg.plans_identifier}__{cfg.configuration}"
        / f"fold_{cfg.fold}"
        / "checkpoints"
    )

    if ckpt == "last":
        return str(checkpoint_dir / "_last_tracker.ckpt")

    if ckpt == "best":
        best_ckpts = sorted(checkpoint_dir.glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)

        if len(best_ckpts) == 0:
            raise FileNotFoundError(f"No best checkpoint found in {checkpoint_dir}")

        return str(best_ckpts[0])

    return ckpt


def validate_experiment_name(experiment, config_map):
    """
    Validate experiment name against CONFIG_MAP.
    """

    experiment = str(experiment).lower().strip()

    if experiment not in config_map:
        valid = ", ".join(config_map.keys())

        raise ValueError(f"Unknown experiment '{experiment}'. Choose one of: {valid}")

    return experiment, config_map[experiment]


def collect_submission_files(cfg, fold_num):
    """
    Collect submission files for one experiment.

    Format:
        task1 -> *-pred.nii.gz
        task2 -> *-pred.nii.gz
        task3 -> *_label_bin.png
    """

    dataset_name = str(cfg.litmodule.dataset_id)

    configuration = cfg.litmodule.configuration
    plans_identifier = cfg.litmodule.plans_identifier
    task_id = cfg.litmodule.task_id
    prefix = cfg.litmodule.prefix

    fold_output_folder = (
        Path(cfg.paths.nnunet_results) / dataset_name / f"{plans_identifier}__{configuration}" / f"fold_{fold_num}"
    )

    submission_folder = fold_output_folder / "submission"

    task_id_str = str(task_id).lower().strip()

    if task_id_str == "task3":
        prediction_pattern = "*_label_bin.png"

    elif task_id_str in ["task1", "task2"]:
        prediction_pattern = "*-pred.nii.gz"

    else:
        raise ValueError(f"Unsupported task_id={task_id}. Expected task1, task2, or task3.")

    prediction_files = sorted(p for p in submission_folder.glob(prediction_pattern) if p.is_file())

    if len(prediction_files) == 0:
        raise FileNotFoundError(
            f"No prediction files found in: {submission_folder}\n"
            f"Task: {task_id}\n"
            f"Expected pattern: {prediction_pattern}"
        )

    return {"prefix": prefix, "task_id": task_id, "prediction_files": prediction_files}


# =============================================================================
# Training progress plot
# =============================================================================


def save_training_progress_plot(
    history, progress_png_file, dataset_name, fold, dice_classwise_keys=None, pseudo_confident_frac_classwise_keys=None
):
    """
    Save training_progress.png from already-computed NumPy history.

    dice_classwise_keys: history keys (e.g. ["dice_class_1", "dice_class_2"])
    each holding one tracked class's dice -- when given, the "Dice" panel
    plots one line per class instead of the single aggregate "dice" mean,
    so classwise performance (e.g. the submitted class vs. training-only
    auxiliary classes) is visible directly in the plot.

    pseudo_confident_frac_classwise_keys: same idea for the "Pseudo
    confident pixel frac" panel, but covering every class (background and
    auxiliary classes included, not just tracked_labels) -- the aggregate
    confident_frac is computed over every pixel regardless of class, so a
    class occupying only a couple percent of a volume can be silently
    filtered out (or not) with almost no visible effect on the aggregate,
    which stays dominated by whatever the majority class is.
    """

    if "epoch" not in history:
        return

    epochs = np.asarray(history["epoch"])

    if epochs.size == 0:
        return

    plot_keys = [
        ("train_loss", "Train loss", "min"),
        ("train_sup_loss", "Supervised loss", "min"),
        ("train_pseudo_loss", "Pseudo loss", "min"),
        ("train_pseudo_confident_frac", "Pseudo confident pixel frac", "max"),
        ("dice", "Dice", "max"),
        ("asd_mm", "ASD mm", "min"),
        ("hd_mm", "HD mm", "min"),
        ("hd95_mm", "HD95 mm", "min"),
    ]

    plot_keys = [item for item in plot_keys if item[0] in history]

    if len(plot_keys) == 0:
        return

    n_plots = len(plot_keys)
    n_cols = math.ceil(math.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13 * n_cols, 11 * n_rows))

    axes = np.asarray(axes).reshape(-1)

    for ax, (key, title, best_mode) in zip(axes, plot_keys):
        if key == "dice" and dice_classwise_keys:
            series = {k.removeprefix("dice_"): np.asarray(history[k], dtype=float) for k in dice_classwise_keys}
        elif key == "train_pseudo_confident_frac" and pseudo_confident_frac_classwise_keys:
            series = {
                k.removeprefix("train_pseudo_confident_frac_"): np.asarray(history[k], dtype=float)
                for k in pseudo_confident_frac_classwise_keys
            }
        else:
            series = {key: np.asarray(history[key], dtype=float)}

        ax.set_title(title, fontsize=13)

        ax.set_xlabel("Epoch")

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.grid(True, which="major", linestyle="-", linewidth=0.7, alpha=0.45)

        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.30)

        all_ys = []

        for series_label, values in series.items():
            mask = ~np.isnan(values)

            xs = epochs[mask]
            ys = values[mask]

            if len(xs) == 0:
                continue

            all_ys.append(ys)

            last_value = ys[-1]

            if best_mode == "min":
                best_value = ys.min()
                best_word = "best/min"
            else:
                best_value = ys.max()
                best_word = "best/max"

            label_prefix = f"{series_label}: " if len(series) > 1 else ""

            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=(f"{label_prefix}last={last_value:.4f}, " f"{best_word}={best_value:.4f}"),
            )

        if len(all_ys) == 0:
            ax.text(0.5, 0.5, "No values yet", ha="center", va="center", transform=ax.transAxes)
            continue

        ys = np.concatenate(all_ys)

        if key == "dice":
            ymin = max(0.0, float(ys.min()) - 0.02)

            ymax = min(1.0, float(ys.max()) + 0.02)

            if ymax - ymin < 0.05:
                center = (ymin + ymax) / 2
                ymin = max(0.0, center - 0.03)
                ymax = min(1.0, center + 0.03)

            ax.set_ylim(ymin, ymax)

            ax.yaxis.set_major_locator(MultipleLocator(0.01))

            ax.yaxis.set_minor_locator(MultipleLocator(0.005))

        elif key in ["asd_mm", "hd_mm", "hd95_mm"]:
            ymin = max(0.0, float(ys.min()) * 0.95)

            ymax = float(ys.max()) * 1.05

            if ymax <= ymin:
                ymax = ymin + 1.0

            ax.set_ylim(ymin, ymax)

            ax.yaxis.set_major_locator(MaxNLocator(nbins=8))

            ax.yaxis.set_minor_locator(MaxNLocator(nbins=16))

        else:
            ymin = float(ys.min())
            ymax = float(ys.max())

            margin = 0.05 * max(abs(ymax - ymin), 1e-6)

            ymin = ymin - margin
            ymax = ymax + margin

            if ymax <= ymin:
                ymax = ymin + 1.0

            ax.set_ylim(ymin, ymax)

            ax.yaxis.set_major_locator(MaxNLocator(nbins=8))

            ax.yaxis.set_minor_locator(MaxNLocator(nbins=16))

        ax.legend(loc="best", fontsize=10)

    for ax in axes[len(plot_keys) :]:
        ax.axis("off")

    progress_png_file = Path(progress_png_file)

    progress_png_file.parent.mkdir(parents=True, exist_ok=True)

    fig.suptitle(f"Training progress | {dataset_name} | fold {fold}", fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(progress_png_file, dpi=180)

    plt.close(fig)


# =============================================================================
# Safe MedPy segmentation metrics
# =============================================================================

# Surface distance is undefined when only one of pred/gt is empty for a
# label (missed or hallucinated class -- no surface on one side to
# measure to/from). Used as a fixed worst-case penalty instead of None,
# so it actively drags down the mean instead of being excluded from it.
MISSED_CLASS_DISTANCE_MM = 1000.0


def safe_binary_segmentation_metrics(pred_mask, gt_mask, voxel_spacing):
    """
    Compute binary segmentation metrics safely.
    """

    pred_mask = np.asarray(pred_mask).astype(bool)
    gt_mask = np.asarray(gt_mask).astype(bool)

    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            "pred_mask and gt_mask must have the same shape. " f"Got pred={pred_mask.shape}, gt={gt_mask.shape}"
        )

    if voxel_spacing is not None:
        voxel_spacing = tuple(float(x) for x in voxel_spacing)

        if len(voxel_spacing) != pred_mask.ndim:
            voxel_spacing = voxel_spacing[-pred_mask.ndim :]

    pred_empty = not pred_mask.any()
    gt_empty = not gt_mask.any()

    if pred_empty and gt_empty:
        return {"Dice": 1.0, "ASD_mm": 0.0, "HD_mm": 0.0, "HD95_mm": 0.0}

    if pred_empty or gt_empty:
        return {
            "Dice": 0.0,
            "ASD_mm": MISSED_CLASS_DISTANCE_MM,
            "HD_mm": MISSED_CLASS_DISTANCE_MM,
            "HD95_mm": MISSED_CLASS_DISTANCE_MM,
        }

    return {
        "Dice": float(dc(pred_mask, gt_mask)),
        "ASD_mm": float(asd(pred_mask, gt_mask, voxelspacing=voxel_spacing)),
        "HD_mm": float(hd(pred_mask, gt_mask, voxelspacing=voxel_spacing)),
        "HD95_mm": float(hd95(pred_mask, gt_mask, voxelspacing=voxel_spacing)),
    }


# =============================================================================
# Rank-local output helpers
# =============================================================================


def get_rank_output_folder(fold_output_folder, global_rank):
    """
    Rank-local output folder for DDP.
    """

    rank_folder = Path(fold_output_folder) / "_rank_outputs" / f"rank_{int(global_rank)}"

    rank_folder.mkdir(parents=True, exist_ok=True)

    return rank_folder


def cleanup_rank_outputs(fold_output_folder):
    """
    Remove fold_all/_rank_outputs.
    """

    rank_root = Path(fold_output_folder) / "_rank_outputs"

    if rank_root.exists():
        shutil.rmtree(rank_root, ignore_errors=True)


def merge_rank_folders(fold_output_folder, task_id, overwrite=True):
    """
    Merge rank-local outputs into final folders.

    Submission format:
        task1/task2 -> *-pred.nii.gz
        task3       -> *_label_bin.png
    """

    fold_output_folder = Path(fold_output_folder)

    rank_root = fold_output_folder / "_rank_outputs"

    final_validation = fold_output_folder / "validation"
    final_prediction = fold_output_folder / "prediction"
    final_submission = fold_output_folder / "submission"

    final_validation.mkdir(parents=True, exist_ok=True)

    final_prediction.mkdir(parents=True, exist_ok=True)

    final_submission.mkdir(parents=True, exist_ok=True)

    if not rank_root.exists():
        raise FileNotFoundError(f"Rank output root does not exist: {rank_root}")

    rank_folders = sorted(p for p in rank_root.glob("rank_*") if p.is_dir())

    if len(rank_folders) == 0:
        raise RuntimeError(f"No rank folders found inside: {rank_root}")

    task_id_str = str(task_id).lower().strip()

    if task_id_str == "task3":
        submission_pattern = "*_label_bin.png"
    else:
        submission_pattern = "*-pred.nii.gz"

    copied_validation = 0
    copied_prediction = 0
    copied_submission = 0

    for rank_folder in rank_folders:
        rank_validation = rank_folder / "validation"
        rank_prediction = rank_folder / "prediction"
        rank_submission = rank_folder / "submission"

        # ------------------------------------------------------------
        # Validation zips
        # ------------------------------------------------------------
        if rank_validation.exists():
            for src in sorted(rank_validation.glob("*.zip")):
                if not src.is_file():
                    continue

                dst = final_validation / src.name

                if dst.exists() and not overwrite:
                    raise RuntimeError(f"Duplicate output file during rank merge: {dst}")

                shutil.copy2(src, dst)

                copied_validation += 1

        # ------------------------------------------------------------
        # Prediction zips
        # ------------------------------------------------------------
        if rank_prediction.exists():
            for src in sorted(rank_prediction.glob("*.zip")):
                if not src.is_file():
                    continue

                dst = final_prediction / src.name

                if dst.exists() and not overwrite:
                    raise RuntimeError(f"Duplicate output file during rank merge: {dst}")

                shutil.copy2(src, dst)

                copied_prediction += 1

        # ------------------------------------------------------------
        # Submission predictions
        # ------------------------------------------------------------
        if rank_submission.exists():
            for src in sorted(rank_submission.glob(submission_pattern)):
                if not src.is_file():
                    continue

                dst = final_submission / src.name

                if dst.exists() and not overwrite:
                    raise RuntimeError(f"Duplicate output file during rank merge: {dst}")

                shutil.copy2(src, dst)

                copied_submission += 1

    if copied_submission == 0:
        raise RuntimeError(
            "No submission files were copied from rank outputs. "
            "Expected files like:\n"
            f"  {rank_root}/rank_*/submission/{submission_pattern}\n\n"
            "This usually means prediction wrote to the wrong folder, "
            "or rank 0 merged before other ranks finished writing."
        )

    return {"validation": copied_validation, "prediction": copied_prediction, "submission": copied_submission}


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
# Segment colors
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

# Complement of SEGMENT_COLOR_PALETTE (1 - r, 1 - g, 1 - b), so ground
# truth is never the same color as a prediction at the same segment_idx --
# they need to be visually distinguishable when overlaid in the same
# scene, not just correct per-segment.
GT_COLOR_PALETTE = [tuple(1.0 - c for c in rgb) for rgb in SEGMENT_COLOR_PALETTE]


def segment_color_string(segment_idx, palette=SEGMENT_COLOR_PALETTE):
    r, g, b = palette[segment_idx % len(palette)]
    return f"{r} {g} {b}"


# =============================================================================
# Shared segmentation image I/O
#
# nnU-Net's own image_reader_writer.write_seg() trips into 16-bit output
# whenever a mask's max value is exactly 255
# (np.uint8 if np.max(seg) < 255 else np.uint16). We control the dtype
# ourselves everywhere a segmentation is written, so this always writes
# uint8 directly and skips that writer entirely. Used by both the Slicer
# case zip writer and the Codabench submission writer, so both paths stay
# consistent with each other.
# =============================================================================
class SegmentationImageIO:

    def read(self, path, reset_direction=False):
        image = sitk.ReadImage(str(path))

        if reset_direction:
            image.SetDirection(tuple(np.eye(image.GetDimension()).flatten()))

        return image

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
# Prediction zip writer
# =============================================================================
def write_prediction_case_zip(
    prediction, zip_dir, configuration_manager, include_gt=False, keep_temp_folder=False, reset_direction=False
):
    """
    Write one prediction dictionary as one Slicer-friendly case zip.

    This version keeps the logic direct.
    No private helper functions.
    No nested helper functions.
    """

    case_id = str(prediction["case_id"])

    zip_dir = Path(zip_dir)

    zip_dir.mkdir(parents=True, exist_ok=True)

    zip_file = zip_dir / f"{case_id}.zip"
    tmp_zip_file = zip_dir / f"{case_id}.zip.tmp_{os.getpid()}"

    case_tmp_dir = zip_dir / f"{case_id}_tmp_{os.getpid()}"

    if case_tmp_dir.exists():
        shutil.rmtree(case_tmp_dir)

    case_tmp_dir.mkdir(parents=True, exist_ok=True)

    if tmp_zip_file.exists():
        tmp_zip_file.unlink()

    written_files = []

    image_files = prediction.get("image_files", None)

    while (
        isinstance(image_files, (list, tuple)) and len(image_files) == 1 and isinstance(image_files[0], (list, tuple))
    ):
        image_files = image_files[0]

    if image_files is None:
        raise RuntimeError("prediction['image_files'] is missing or empty.")

    if isinstance(image_files, (str, Path)):
        image_files = [image_files]

    image_files = [Path(str(p)) for p in image_files]

    if len(image_files) == 0:
        raise RuntimeError("prediction['image_files'] is empty.")

    display_image_files = []
    reference_image = None

    for idx, src in enumerate(image_files):
        if not src.exists():
            raise FileNotFoundError(f"Raw image file not found: {src}")

        dst = case_tmp_dir / f"{case_id}_image_view_channel_{idx:04d}.nii.gz"

        # reset_direction=True only for tasks whose raw nii.gz direction is
        # a placeholder artifact (e.g. task3 video: nibabel writes with an
        # identity affine, then SimpleITK/ITK applies its RAS -> LPS sign
        # convention on read, producing a (-1, -1) direction with no real
        # spatial meaning). Tasks with real scan geometry (CT/TEE) must
        # keep their true spacing/origin/direction, so this defaults False.
        image = segmentation_io.read(src, reset_direction=reset_direction)
        segmentation_io.write_volume(image, dst)

        if idx == 0:
            reference_image = image

        display_image_files.append(dst)
        written_files.append(dst)

    probability_files = []

    if prediction.get("predicted_probs", None) is not None:
        probs = to_numpy(prediction["predicted_probs"])
        probs = np.asarray(probs)

        if probs.ndim >= 4 and probs.shape[0] == 1:
            probs = probs[0]

        for c in range(probs.shape[0]):
            prob_array = np.asarray(probs[c], dtype=np.float32)

            prob_array = np.clip(prob_array, 0.0, 1.0)

            # Store probability as scalar uint8:
            # 0.0 -> 1
            # 1.0 -> 255
            # Colors are applied later by MRML colorNodeRef.
            prob_display = np.rint(1.0 + prob_array * 254.0).clip(1, 255).astype(np.uint8)

            if reference_image.GetDimension() == 2 and prob_display.ndim == 3 and prob_display.shape[0] == 1:
                prob_display = prob_display[0]

            prob_img = segmentation_io.build_segmentation_image(prob_display, reference=reference_image)

            prob_file = case_tmp_dir / f"{case_id}_probability_channel_{c:04d}.nii.gz"

            segmentation_io.write_volume(prob_img, prob_file)

            probability_files.append(prob_file)
            written_files.append(prob_file)

    segmentation = to_numpy(prediction["predicted_segments"])
    segmentation = np.asarray(segmentation)

    if segmentation.ndim >= 3 and segmentation.shape[0] == 1:
        segmentation = segmentation[0]

    prediction_seg_file = case_tmp_dir / f"{case_id}_prediction.seg.nrrd"

    prediction_seg_image = segmentation_io.build_segmentation_image(segmentation, reference=reference_image)

    unique_labels = sorted(int(x) for x in np.unique(segmentation) if int(x) > 0)

    if len(unique_labels) == 0:
        unique_labels = [1]

    for segment_idx, label_value in enumerate(unique_labels):
        prediction_seg_image.SetMetaData(f"Segment{segment_idx}_ID", f"{case_id}_prediction_{label_value}")

        prediction_seg_image.SetMetaData(f"Segment{segment_idx}_Name", f"{case_id}_prediction_{label_value}")

        prediction_seg_image.SetMetaData(f"Segment{segment_idx}_Color", segment_color_string(segment_idx))

        prediction_seg_image.SetMetaData(f"Segment{segment_idx}_LabelValue", str(label_value))

        prediction_seg_image.SetMetaData(f"Segment{segment_idx}_Layer", "0")

    prediction_seg_image.SetMetaData("Segmentation_ContainedRepresentationNames", "Binary labelmap")

    prediction_seg_image.SetMetaData("Segmentation_ConversionParameters", "")

    prediction_seg_image.SetMetaData("Segmentation_MasterRepresentation", "Binary labelmap")

    prediction_seg_image.SetMetaData("Segmentation_ReferenceImageExtentOffset", "0 0 0")

    segmentation_io.write_volume(prediction_seg_image, prediction_seg_file)

    written_files.append(prediction_seg_file)

    gt_seg_file = None

    if include_gt and prediction.get("gt_data", None) is not None:
        gt = to_numpy(prediction["gt_data"])
        gt = np.asarray(gt)

        if gt.ndim >= 3 and gt.shape[0] == 1:
            gt = gt[0]

        gt_seg_file = case_tmp_dir / f"{case_id}_gt.seg.nrrd"

        gt_seg_image = segmentation_io.build_segmentation_image(gt, reference=reference_image)

        unique_gt_labels = sorted(int(x) for x in np.unique(gt) if int(x) > 0)

        if len(unique_gt_labels) == 0:
            unique_gt_labels = [1]

        for segment_idx, label_value in enumerate(unique_gt_labels):
            gt_seg_image.SetMetaData(f"Segment{segment_idx}_ID", f"{case_id}_ground_truth_{label_value}")

            gt_seg_image.SetMetaData(f"Segment{segment_idx}_Name", f"{case_id}_ground_truth_{label_value}")

            gt_seg_image.SetMetaData(f"Segment{segment_idx}_Color", segment_color_string(segment_idx, palette=GT_COLOR_PALETTE))

            gt_seg_image.SetMetaData(f"Segment{segment_idx}_LabelValue", str(label_value))

            gt_seg_image.SetMetaData(f"Segment{segment_idx}_Layer", "0")

        gt_seg_image.SetMetaData("Segmentation_ContainedRepresentationNames", "Binary labelmap")

        gt_seg_image.SetMetaData("Segmentation_ConversionParameters", "")

        gt_seg_image.SetMetaData("Segmentation_MasterRepresentation", "Binary labelmap")

        gt_seg_image.SetMetaData("Segmentation_ReferenceImageExtentOffset", "0 0 0")

        segmentation_io.write_volume(gt_seg_image, gt_seg_file)

        written_files.append(gt_seg_file)

    storage_nodes = []
    display_nodes = []
    volume_nodes = []

    node_idx = 1

    for idx, image_file in enumerate(display_image_files):
        volume_name = f"{case_id}_image_view_channel_{idx:04d}"

        storage_id = f"vtkMRMLVolumeArchetypeStorageNode{node_idx}"
        volume_id = f"vtkMRMLScalarVolumeNode{node_idx}"
        display_id = f"vtkMRMLScalarVolumeDisplayNode{node_idx}"

        try:
            img = sitk.ReadImage(str(image_file))
            arr = sitk.GetArrayFromImage(img)
            arr = np.asarray(arr, dtype=np.float32)
            arr = arr[np.isfinite(arr)]

            if arr.size > 0:
                lo = float(np.percentile(arr, 0.5))
                hi = float(np.percentile(arr, 99.5))

                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo = float(np.min(arr))
                    hi = float(np.max(arr))

                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    window = hi - lo
                    level = (hi + lo) / 2.0
                else:
                    window = 255.0
                    level = 127.5
            else:
                window = 255.0
                level = 127.5

        except Exception:
            window = 255.0
            level = 127.5

        visibility = "true" if idx == 0 else "false"

        storage_nodes.append(f"""  <VolumeArchetypeStorage
    id="{storage_id}"
    name="{volume_name}Storage"
    fileName="{image_file.name}"
    useCompression="1"/>""")

        display_nodes.append(f"""  <VolumeDisplay
    id="{display_id}"
    name="{volume_name}Display"
    colorNodeRef="vtkMRMLColorTableNodeGrey"
    interpolate="true"
    autoWindowLevel="false"
    window="{window}"
    level="{level}"/>""")

        volume_nodes.append(f"""  <Volume
    id="{volume_id}"
    name="{volume_name}"
    storageNodeRef="{storage_id}"
    displayNodeRef="{display_id}"
    labelMap="0"
    selectable="true"
    visibility="{visibility}"/>""")

        node_idx += 1

    for idx, prob_file in enumerate(probability_files):
        volume_name = f"{case_id}_probability_channel_{idx:04d}"

        storage_id = f"vtkMRMLVolumeArchetypeStorageNode{node_idx}"
        volume_id = f"vtkMRMLScalarVolumeNode{node_idx}"
        display_id = f"vtkMRMLScalarVolumeDisplayNode{node_idx}"

        storage_nodes.append(f"""  <VolumeArchetypeStorage
    id="{storage_id}"
    name="{volume_name}Storage"
    fileName="{prob_file.name}"
    useCompression="1"/>""")

        display_nodes.append(f"""  <VolumeDisplay
    id="{display_id}"
    name="{volume_name}Display"
    colorNodeRef="vtkMRMLColorTableNodeFileColdToHotRainbow.txt"
    interpolate="true"
    autoWindowLevel="false"
    window="254.0"
    level="128.0"/>""")

        volume_nodes.append(f"""  <Volume
    id="{volume_id}"
    name="{volume_name}"
    storageNodeRef="{storage_id}"
    displayNodeRef="{display_id}"
    labelMap="0"
    selectable="true"
    visibility="false"/>""")

        node_idx += 1

    storage_id = f"vtkMRMLSegmentationStorageNode{node_idx}"
    seg_id = f"vtkMRMLSegmentationNode{node_idx}"
    display_id = f"vtkMRMLSegmentationDisplayNode{node_idx}"

    storage_nodes.append(f"""  <SegmentationStorage
    id="{storage_id}"
    name="{case_id}_predictionStorage"
    fileName="{prediction_seg_file.name}"/>""")

    display_nodes.append(f"""  <SegmentationDisplay
    id="{display_id}"
    name="{case_id}_predictionDisplay"
    visibility="true"
    opacity="0.5"
    opacity3D="0.5"
    opacity2DFill="0.5"
    opacity2DOutline="1"
    visibility2DFill="true"
    visibility2DOutline="true"
    visibility3D="true"/>""")

    volume_nodes.append(f"""  <Segmentation
    id="{seg_id}"
    name="{case_id}_prediction"
    storageNodeRef="{storage_id}"
    displayNodeRef="{display_id}"
    selectable="true"/>""")

    node_idx += 1

    if gt_seg_file is not None:
        storage_id = f"vtkMRMLSegmentationStorageNode{node_idx}"
        seg_id = f"vtkMRMLSegmentationNode{node_idx}"
        display_id = f"vtkMRMLSegmentationDisplayNode{node_idx}"

        storage_nodes.append(f"""  <SegmentationStorage
    id="{storage_id}"
    name="{case_id}_ground_truthStorage"
    fileName="{gt_seg_file.name}"/>""")

        display_nodes.append(f"""  <SegmentationDisplay
    id="{display_id}"
    name="{case_id}_ground_truthDisplay"
    visibility="true"
    opacity="0.35"
    opacity3D="0.35"
    opacity2DFill="0.35"
    opacity2DOutline="1"
    visibility2DFill="true"
    visibility2DOutline="true"
    visibility3D="true"/>""")

        volume_nodes.append(f"""  <Segmentation
    id="{seg_id}"
    name="{case_id}_ground_truth"
    storageNodeRef="{storage_id}"
    displayNodeRef="{display_id}"
    selectable="true"/>""")

    mrml_text = f"""<?xml version="1.0" encoding="UTF-8"?>
<MRML version="Slicer 5.0.0">
{chr(10).join(storage_nodes)}
{chr(10).join(display_nodes)}
{chr(10).join(volume_nodes)}
</MRML>
"""

    mrml_file = case_tmp_dir / f"{case_id}.mrml"

    with open(mrml_file, "w", encoding="utf-8") as f:
        f.write(mrml_text)

    written_files.append(mrml_file)

    with zipfile.ZipFile(tmp_zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in written_files:
            file_path = Path(file_path)

            zf.write(file_path, arcname=str(Path(case_id) / file_path.name))

    os.replace(tmp_zip_file, zip_file)

    if not keep_temp_folder and case_tmp_dir.exists():
        shutil.rmtree(case_tmp_dir)

    return zip_file


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
