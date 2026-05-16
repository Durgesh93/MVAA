from pathlib import Path
import tarfile
import random
import io
import gzip

import numpy as np
import nibabel as nib
import pandas as pd
from PIL import Image
import typer

from utils import (
    log_section,
    log_info,
    log_ok,
    log_warn,
    print_group_counts,
    make_case_name,
    find_task_dirs,
    build_sample_dataframe,
    write_dataset,
    write_dummy_mask_from_image,
)


app = typer.Typer(help="Prepare MVAA Task 3 VIDEO dataset in nnU-Net format.")


DATA_TYPE = "video"
DATASET_ID = "Dataset003_MVAA_VIDEO_SSL"
PREFIX = "video"
PATTERNS = ["t3_vid", "*vid*", "*video*"]
CASE_SUFFIXES = [".png", ".jpg", ".jpeg", "_png_Label.tar"]

VIDEO_UNLABELED_TRAIN_RATIO = 0.70

# Original LV class in the video semantic labels
VIDEO_LV_LABEL = 10

# Output labels for nnU-Net
VIDEO_BINARY_LV_LABELS = {
    "background": 0,
    "LV": 1,
}


def collect_extra_unlabeled_video(data_root):
    rows = []

    data_root = Path(data_root)
    image_dir = data_root / "images"

    if not image_dir.exists():
        log_warn(f"Extra video image folder not found: {image_dir}")
        return rows

    image_paths = [
        p for p in sorted(image_dir.rglob("*"))
        if p.is_file() and p.name.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    rng = random.Random(42)
    rng.shuffle(image_paths)

    n_train = int(len(image_paths) * VIDEO_UNLABELED_TRAIN_RATIO)

    train_paths = image_paths[:n_train]
    test_paths = image_paths[n_train:]

    for p in train_paths:
        rows.append({
            "data_type": DATA_TYPE,
            "task_id": "t3_vid",
            "split": "TrU",
            "file_type": "image",
            "case_name": make_case_name(p, PREFIX, CASE_SUFFIXES),
            "file_path": str(p),
        })

    for p in test_paths:
        rows.append({
            "data_type": DATA_TYPE,
            "task_id": "t3_vid",
            "split": "Ts",
            "file_type": "image",
            "case_name": make_case_name(p, PREFIX, CASE_SUFFIXES),
            "file_path": str(p),
        })

    log_ok(f"Extra video images: {len(image_paths)}")
    typer.echo(f"  TrU: {len(train_paths)}")
    typer.echo(f"  Ts : {len(test_paths)}")

    return rows


def collect_video_files(data_root):
    data_root = Path(data_root)
    reference_dir = data_root / "reference_data"
    task_dirs = find_task_dirs(reference_dir, PATTERNS)

    rows = []

    if len(task_dirs) == 0:
        log_warn(f"No reference video folders found under {reference_dir}")

    for task_dir in task_dirs:
        for p in sorted((task_dir / "train").glob("*/*.png")):
            rows.append({
                "data_type": DATA_TYPE,
                "task_id": task_dir.name,
                "split": "TrL",
                "file_type": "image",
                "case_name": make_case_name(p, PREFIX, CASE_SUFFIXES),
                "file_path": str(p),
            })

        for p in sorted((task_dir / "train").glob("*/*_png_Label.tar")):
            rows.append({
                "data_type": DATA_TYPE,
                "task_id": task_dir.name,
                "split": "TrL",
                "file_type": "mask",
                "case_name": make_case_name(p, PREFIX, CASE_SUFFIXES),
                "file_path": str(p),
            })

        for p in sorted((task_dir / "val" / "images").glob("*/*.png")):
            rows.append({
                "data_type": DATA_TYPE,
                "task_id": task_dir.name,
                "split": "Ts",
                "file_type": "image",
                "case_name": make_case_name(p, PREFIX, CASE_SUFFIXES),
                "file_path": str(p),
            })

    rows.extend(collect_extra_unlabeled_video(data_root))

    return rows


def write_video_rgb_image_as_nnunet_channels(src_path, dst_path):
    img = Image.open(src_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32)

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    primary_channel_path = None

    for channel_idx in range(3):
        channel_arr = arr[:, :, channel_idx]

        channel_nii = nib.Nifti1Image(
            channel_arr,
            affine=np.eye(4),
        )

        channel_path = dst_path.parent / f"{dst_path.name}_{channel_idx:04d}.nii.gz"
        nib.save(channel_nii, str(channel_path))

        if channel_idx == 0:
            primary_channel_path = channel_path

    return primary_channel_path


def write_video_lv_binary_mask_from_tar(src_path, dst_path):
    """
    Read original multi-class video mask from *_png_Label.tar
    and convert it to binary LV mask.

    Original:
        0  = background
        10 = LV

    Output:
        0 = background
        1 = LV
    """

    try:
        with tarfile.open(src_path, "r:*") as tar:
            members = [
                m for m in tar.getmembers()
                if m.isfile() and m.name.lower().endswith(".nii.gz")
            ]

            if len(members) == 0:
                log_warn(f"No .nii.gz mask found in {src_path}")
                return False

            file_obj = tar.extractfile(members[0])

            if file_obj is None:
                log_warn(f"Could not read mask from {src_path}")
                return False

            nii_gz_bytes = file_obj.read()

        with gzip.GzipFile(fileobj=io.BytesIO(nii_gz_bytes), mode="rb") as gz:
            nii_bytes = gz.read()

        img = nib.Nifti1Image.from_bytes(nii_bytes)
        mask_raw = np.asanyarray(img.dataobj)
        mask_raw = np.squeeze(mask_raw)

        mask_binary = (mask_raw == VIDEO_LV_LABEL).astype(np.uint8)

        out_nii = nib.Nifti1Image(
            mask_binary,
            affine=img.affine,
        )

        out_nii.set_data_dtype(np.uint8)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(out_nii, str(dst_path))

        return True

    except Exception as e:
        log_warn(f"Could not extract binary LV mask from {src_path}: {e}")
        return False


def prepare_video_dataset(data_root, output_dir, test=False):
    log_section("Processing Task 3 VIDEO")

    rows = collect_video_files(data_root)

    if len(rows) == 0:
        log_warn("No files found for VIDEO")
        return

    file_df = pd.DataFrame(rows)
    print_group_counts(file_df, ["split", "file_type"], "Collected files")

    sample_df = build_sample_dataframe(rows)

    print_group_counts(sample_df, ["split", "is_labeled"], "Prepared samples")

    write_dataset(
        sample_df=sample_df,
        dataset_id=DATASET_ID,
        data_type=DATA_TYPE,
        output_dir=output_dir,
        test=test,
        fixed_labels=VIDEO_BINARY_LV_LABELS,
        write_image_fn=write_video_rgb_image_as_nnunet_channels,
        write_real_mask_fn=write_video_lv_binary_mask_from_tar,
        write_dummy_mask_fn=write_dummy_mask_from_image,
        description_extra=(
            "Task 3 VIDEO dataset. RGB frames are stored as three nnU-Net "
            "channels. Video masks are converted to binary LV masks."
        ),
    )


@app.command()
def main(
    data_root: Path = typer.Option(
        Path("dirs/data_storage/raw/MVAA"),
        "--data-root",
        help="MVAA root directory containing reference_data/ and images/.",
    ),
    output_dir: Path = typer.Option(
        Path("dirs/data_storage/raw/MVAA_nnUNET_SSL"),
        "--output-dir",
        help="Output directory for prepared nnU-Net dataset.",
    ),
    test: bool = typer.Option(
        False,
        "--test",
        help="Prepare only a few samples per split.",
    ),
):
    output_dir.mkdir(parents=True, exist_ok=True)

    log_info(f"data_root  : {data_root}")
    log_info(f"output_dir : {output_dir}")
    log_info(f"test mode  : {test}")

    prepare_video_dataset(
        data_root=data_root,
        output_dir=output_dir,
        test=test,
    )


if __name__ == "__main__":
    app()
