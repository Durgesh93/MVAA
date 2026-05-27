"""
Utility functions for Lightning nnU-Net wrappers.
"""

import os
import sys
from pathlib import Path
import shutil
import zipfile
import math
import json

import numpy as np
import matplotlib

from tqdm import tqdm
from medpy.metric.binary import dc, asd, hd, hd95
import SimpleITK as sitk
from lightning.pytorch.callbacks import Callback

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator


# =============================================================================
# nnU-Net environment
# =============================================================================

def set_nnunet_env(cfg):
    """
    Set nnU-Net environment variables.
    """

    nnunet_raw = str(cfg.paths.nnunet_raw)
    nnunet_preprocessed = str(cfg.paths.nnunet_preprocessed)
    nnunet_results = str(cfg.paths.nnunet_results)

    env = {
        "nnunet_raw": nnunet_raw,
        "nnunet_preprocessed": nnunet_preprocessed,
        "nnunet_results": nnunet_results,
    }

    os.environ["nnUNet_raw"] = nnunet_raw
    os.environ["nnUNet_preprocessed"] = nnunet_preprocessed
    os.environ["nnUNet_results"] = nnunet_results

    return env


def convert_predictions_to_255(output_folder):
    """
    Convert nnU-Net binary predictions from 0/1 to 0/255.
    Saves back to the same prediction files.
    """

    output_folder = Path(output_folder)
    pred_files = sorted(output_folder.glob("*.nii.gz"))

    for pred_file in pred_files:
        img = sitk.ReadImage(str(pred_file))
        arr = sitk.GetArrayFromImage(img)

        arr_255 = (arr > 0).astype(np.uint8) * 255

        out_img = sitk.GetImageFromArray(arr_255)
        out_img.CopyInformation(img)

        sitk.WriteImage(out_img, str(pred_file))


# =============================================================================
# TQDM progress bar for actual validation / prediction
# =============================================================================

class ActualValidationTQDMCallback(Callback):
    """
    TQDM callback for actual nnU-Net validation/prediction stages.
    """

    def __init__(self, every=1):
        super().__init__()

        self.every = every
        self.pbar = None
        self.stage = None

    def update_actual_validation(self, stage, current, total):
        if total is None or total <= 0:
            total = 1

        current = min(current, total)

        if self.pbar is None or self.stage != stage:
            if self.pbar is not None:
                self.pbar.close()

            self.stage = stage

            self.pbar = tqdm(
                total=total,
                desc=f"actual validation | {stage}",
                file=sys.stdout,
                leave=True,
                dynamic_ncols=True,
                disable=False,
            )

        delta = current - self.pbar.n

        if delta > 0:
            self.pbar.update(delta)

        if current == 1 or current == total or current % self.every == 0:
            sys.stdout.flush()

        if current >= total:
            self.pbar.close()
            self.pbar = None
            self.stage = None


def update_actual_validation_progress(trainer, stage, current, total):
    """
    Find ActualValidationTQDMCallback inside trainer.callbacks and update it.
    Does nothing on non-global-zero ranks in DDP.
    """

    if trainer is None:
        return

    if not trainer.is_global_zero:
        return

    for callback in trainer.callbacks:
        if hasattr(callback, "update_actual_validation"):
            callback.update_actual_validation(
                stage=stage,
                current=current,
                total=total,
            )
            return


# =============================================================================
# Prediction JSON
# =============================================================================

def write_predictions_json(task_id, output_folder):
    """
    Write simple predictions JSON from exported .nii.gz files.

    Only includes .nii.gz files directly inside output_folder.
    Does not include files from output_folder/cases.
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    cases = []
    pred_files = sorted(output_folder.glob("*.nii.gz"))

    for pred_file in pred_files:
        case_id = pred_file.name.replace(".nii.gz", "")

        cases.append(
            {
                "case_id": case_id,
                "segmentation": pred_file.name,
            }
        )

    cases = sorted(cases, key=lambda x: x["case_id"])

    output_json = output_folder / f"{task_id}_predictions.json"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cases": cases,
            },
            f,
            indent=2,
        )

    return output_json


# =============================================================================
# Training progress plot
# =============================================================================

def save_training_progress_plot(
    history,
    progress_png_file,
    dataset_name,
    fold,
):
    """
    Save training_progress.png from already-computed NumPy history.
    """

    epochs = history["epoch"]

    if epochs.size == 0:
        return

    plot_keys = [
        ("train_loss", "Train loss", "min"),
        ("train_sup_loss", "Supervised loss", "min"),
        ("train_pseudo_loss", "Pseudo loss", "min"),
        ("val_loss", "Validation loss", "min"),
        ("dice", "Dice", "max"),
        ("asd_mm", "ASD mm", "min"),
        ("hd_mm", "HD mm", "min"),
        ("hd95_mm", "HD95 mm", "min"),
    ]

    n_plots = len(plot_keys)
    n_cols = math.ceil(math.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(13 * n_cols, 11 * n_rows),
    )

    axes = np.asarray(axes).reshape(-1)

    for ax, (key, title, best_mode) in zip(axes, plot_keys):
        values = history[key]
        mask = ~np.isnan(values)

        xs = epochs[mask]
        ys = values[mask]

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.grid(
            True,
            which="major",
            linestyle="-",
            linewidth=0.7,
            alpha=0.45,
        )
        ax.grid(
            True,
            which="minor",
            linestyle=":",
            linewidth=0.5,
            alpha=0.30,
        )

        if len(xs) == 0:
            ax.text(
                0.5,
                0.5,
                "No values yet",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        last_value = ys[-1]

        if best_mode == "min":
            best_value = ys.min()
            best_word = "best/min"
        else:
            best_value = ys.max()
            best_word = "best/max"

        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=f"last={last_value:.4f}, {best_word}={best_value:.4f}",
        )

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

    for ax in axes[len(plot_keys):]:
        ax.axis("off")

    progress_png_file = Path(progress_png_file)
    progress_png_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.suptitle(
        f"Training progress | {dataset_name} | fold {fold}",
        fontsize=16,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(progress_png_file, dpi=180)
    plt.close(fig)


# =============================================================================
# Folder cleanup helpers
# =============================================================================

def reset_folder(path):
    """
    Delete a folder if it exists, then recreate it.
    """

    path = Path(path)

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)

    return path


def remove_folder(path):
    """
    Delete a folder if it exists.
    """

    path = Path(path)

    if path.exists():
        shutil.rmtree(path)


# =============================================================================
# Case zip helpers
# =============================================================================

def find_raw_image_files_for_case(
    raw_dataset_dir,
    case_id,
    search_images_tr=True,
    search_images_ts=True,
):
    """
    Find raw nnU-Net image channel files for one case.

    Validation:
        search imagesTr first, then imagesTs.

    Test:
        search imagesTs only.
    """

    raw_dataset_dir = Path(raw_dataset_dir)

    image_files = []

    if search_images_tr:
        raw_images_tr = raw_dataset_dir / "imagesTr"
        image_files = sorted(raw_images_tr.glob(f"{case_id}_*.nii.gz"))

    if len(image_files) == 0 and search_images_ts:
        raw_images_ts = raw_dataset_dir / "imagesTs"
        image_files = sorted(raw_images_ts.glob(f"{case_id}_*.nii.gz"))

    return image_files


def write_validation_case_zip(
    case_id,
    image_files,
    pred_file,
    gt_file,
    zip_dir,
):
    """
    Create one validation-case zip.

    Inside zip:
        image_<raw_image_filename>
        pred_<prediction_filename>
        gt_<gt_filename>
    """

    zip_dir = Path(zip_dir)
    zip_dir.mkdir(parents=True, exist_ok=True)

    pred_file = Path(pred_file)
    gt_file = Path(gt_file)

    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    if not gt_file.exists():
        raise FileNotFoundError(f"GT file not found: {gt_file}")

    if image_files is None or len(image_files) == 0:
        raise FileNotFoundError(f"No raw image files found for case: {case_id}")

    image_files = [Path(p) for p in image_files]

    zip_file = zip_dir / f"{case_id}.zip"

    if zip_file.exists():
        zip_file.unlink()

    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for image_file in image_files:
            if not image_file.exists():
                raise FileNotFoundError(f"Image file not found: {image_file}")

            zf.write(
                image_file,
                arcname=f"image_{image_file.name}",
            )

        zf.write(
            pred_file,
            arcname=f"pred_{pred_file.name}",
        )

        zf.write(
            gt_file,
            arcname=f"gt_{gt_file.name}",
        )

    return zip_file


def write_test_case_zip(
    case_id,
    image_files,
    pred_file,
    zip_dir,
):
    """
    Create one test-case zip.

    Inside zip:
        image_<raw_image_filename>
        <prediction_filename>

    No GT is included for test cases.
    """

    zip_dir = Path(zip_dir)
    zip_dir.mkdir(parents=True, exist_ok=True)

    pred_file = Path(pred_file)

    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    if image_files is None or len(image_files) == 0:
        raise FileNotFoundError(f"No raw image files found for case: {case_id}")

    image_files = [Path(p) for p in image_files]

    zip_file = zip_dir / f"{case_id}.zip"

    if zip_file.exists():
        zip_file.unlink()

    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for image_file in image_files:
            if not image_file.exists():
                raise FileNotFoundError(f"Image file not found: {image_file}")

            zf.write(
                image_file,
                arcname=f"image_{image_file.name}",
            )

        zf.write(
            pred_file,
            arcname=pred_file.name,
        )

    return zip_file


def zip_validation_cases(
    val_keys,
    raw_dataset_dir,
    prediction_folder,
    gt_folder,
    zip_dir,
    file_ending,
    progress_fn=None,
):
    """
    Create one zip per validation case.

    One progress bar only:
        actual validation | zip validation cases
    """

    prediction_folder = Path(prediction_folder)
    gt_folder = Path(gt_folder)
    zip_dir = reset_folder(zip_dir)

    total_cases = len(val_keys)

    for i, case_id in enumerate(val_keys, start=1):
        if progress_fn is not None:
            progress_fn(
                stage="zip validation cases",
                current=i,
                total=total_cases,
            )

        image_files = find_raw_image_files_for_case(
            raw_dataset_dir=raw_dataset_dir,
            case_id=case_id,
            search_images_tr=True,
            search_images_ts=True,
        )

        pred_file = prediction_folder / f"{case_id}_pred{file_ending}"
        gt_file = gt_folder / f"{case_id}{file_ending}"

        write_validation_case_zip(
            case_id=case_id,
            image_files=image_files,
            pred_file=pred_file,
            gt_file=gt_file,
            zip_dir=zip_dir,
        )


def zip_test_cases(
    raw_dataset_dir,
    prediction_folder,
    zip_dir,
    file_ending,
    progress_fn=None,
):
    """
    Create one zip per test prediction case.

    One progress bar only:
        actual validation | zip test cases

    For test:
        imagesTs/<case_id>_*.nii.gz
        prediction_folder/<case_id>.nii.gz
    """

    prediction_folder = Path(prediction_folder)
    zip_dir = reset_folder(zip_dir)

    pred_files = sorted(prediction_folder.glob(f"*{file_ending}"))

    pred_files = [
        p for p in pred_files
        if p.is_file()
        and p.parent == prediction_folder
    ]

    total_cases = len(pred_files)

    for i, pred_file in enumerate(pred_files, start=1):
        case_id = pred_file.name[: -len(file_ending)]

        if progress_fn is not None:
            progress_fn(
                stage="zip test cases",
                current=i,
                total=total_cases,
            )

        image_files = find_raw_image_files_for_case(
            raw_dataset_dir=raw_dataset_dir,
            case_id=case_id,
            search_images_tr=False,
            search_images_ts=True,
        )

        write_test_case_zip(
            case_id=case_id,
            image_files=image_files,
            pred_file=pred_file,
            zip_dir=zip_dir,
        )


# =============================================================================
# Safe MedPy segmentation metrics
# =============================================================================

def safe_binary_segmentation_metrics(pred_mask, gt_mask, voxel_spacing):
    """
    Compute binary segmentation metrics safely.

    Returns:
        Dice    : unitless
        ASD_mm  : average surface distance in mm
        HD_mm   : Hausdorff distance in mm
        HD95_mm : 95th percentile Hausdorff distance in mm
    """

    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    pred_empty = not pred_mask.any()
    gt_empty = not gt_mask.any()

    if pred_empty and gt_empty:
        return {
            "Dice": 1.0,
            "ASD_mm": 0.0,
            "HD_mm": 0.0,
            "HD95_mm": 0.0,
        }

    if pred_empty or gt_empty:
        return {
            "Dice": 0.0,
            "ASD_mm": None,
            "HD_mm": None,
            "HD95_mm": None,
        }

    return {
        "Dice": float(dc(pred_mask, gt_mask)),
        "ASD_mm": float(asd(pred_mask, gt_mask, voxelspacing=voxel_spacing)),
        "HD_mm": float(hd(pred_mask, gt_mask, voxelspacing=voxel_spacing)),
        "HD95_mm": float(hd95(pred_mask, gt_mask, voxelspacing=voxel_spacing)),
    }
