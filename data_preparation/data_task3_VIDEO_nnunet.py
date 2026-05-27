from pathlib import Path
import os
import shutil
import subprocess
import sys
import tarfile
import random
import io
import gzip

import numpy as np
import nibabel as nib
import pandas as pd
from PIL import Image
import typer

from data_preparation.utils import (
    log_section,
    log_info,
    log_ok,
    log_warn,
    print_group_counts,
    make_case_name,
    find_task_dirs,
    build_sample_dataframe,
    write_dataset,
    set_nnunet_env,
    reassign_training_case_names_sequentially,
)


app = typer.Typer(help="Prepare MVAA Task 3 VIDEO dataset in nnU-Net format.")



DATASET_ID = "Dataset003_MVAA_VIDEO_SSL"
PATTERNS = ["t3_vid", "*vid*", "*video*"]
CASE_SUFFIXES = [".png", ".jpg", ".jpeg", "_png_Label.tar"]
VIDEO_LV_LABEL = 10

VIDEO_BINARY_LV_LABELS = {
    "background": 0,
    "LV": 1,
}


def get_dataset_number(dataset_id):
    return int(dataset_id.replace("Dataset", "")[:3])

def write_dummy_mask_from_image(image_path, mask_path):
    img   = nib.load(str(image_path))
    dummy = np.zeros(img.shape, dtype=np.uint8)

    dummy_nii = nib.Nifti1Image(
        dummy,
        affine=img.affine,
        header=img.header,
    )

    dummy_nii.set_data_dtype(np.uint8)

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(dummy_nii, str(mask_path))


def run_nnunet_plan_and_preprocess(dataset_id, num_processes):
    dataset_number = get_dataset_number(dataset_id)

    exe = shutil.which("nnUNetv2_plan_and_preprocess")

    if exe is not None:
        cmd = [
            exe,
            "-d",
            str(dataset_number),
            "-np",
            str(num_processes),
            "-npfp",
            str(num_processes),
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
            "-d",
            str(dataset_number),
            "-np",
            str(num_processes),
            "-npfp",
            str(num_processes),
        ]

    cmd.append("--verify_dataset_integrity")

    log_info("Running nnU-Net plan and preprocess:")
    typer.echo(" ".join(cmd))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    assert process.stdout is not None

    for line in process.stdout:
        typer.echo(line.rstrip())

    return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

    log_ok(f"Finished nnU-Net preprocessing for {dataset_id}")


def collect_extra_unlabeled_video(data_root):
    rows      = []
    data_root = Path(data_root)
    image_dir = data_root / "images"

    if not image_dir.exists():
        log_warn(f"Extra video image folder not found: {image_dir}")
        return rows

    image_paths = [
        p
        for p in sorted(image_dir.rglob("*"))
        if p.is_file() and p.name.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for p in image_paths:
        rows.append(
            {
                "split": "TrU",
                "file_type": "image",
                "case_name": make_case_name(p, CASE_SUFFIXES),
                "file_path": str(p),
            }
        )

    log_ok(
        f"Extra unlabeled video images: total={len(image_paths)}"
    )
    return rows

def collect_video_files(data_root):
    reference_dir = Path(data_root) / "reference_data"
    task_dirs     = find_task_dirs(reference_dir, PATTERNS)

    rows = []

    if len(task_dirs) == 0:
        log_warn(f"No VIDEO folders found under {reference_dir}")

    for task_dir in task_dirs:
        train_dir = task_dir / "train"
        test_dir  = task_dir / "val" / "images"

        for p in sorted(train_dir.glob("*/*.png")):
            parent_name = p.parent.name

            rows.append(
                {
                    "split":       "TrL",
                    "file_type":   "image",
                    "case_name":   f"{make_case_name(p, CASE_SUFFIXES)}",
                    "file_path":   str(p),
                }
            )

        for p in sorted(train_dir.glob("*/*_png_Label.tar")):
            parent_name = p.parent.name

            rows.append(
                {
                    "split":       "TrL",
                    "file_type":   "mask",
                    "case_name":   f"{make_case_name(p, CASE_SUFFIXES)}",
                    "file_path":   str(p),
                }
            )

        for p in sorted(test_dir.glob("*/*.png")):
            rows.append(
                {
                    "split":       "Ts",
                    "file_type":   "image",
                    "case_name":   f"{make_case_name(p, CASE_SUFFIXES)}",
                    "file_path":   str(p),
                }
            )

            rows.append(
                {
                    "split":       "TrU",
                    "file_type":   "image",
                    "case_name":   f"{make_case_name(p, CASE_SUFFIXES)}",
                    "file_path":   str(p),
                }
            )


    rows.extend(collect_extra_unlabeled_video(data_root))

    return rows



def write_video_rgb_image_as_nnunet_channels(src_path, dst_path):
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(src_path).convert("RGB")
    arr = np.asarray(image)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got shape {arr.shape}: {src_path}")

    affine = np.eye(4)

    written_paths = []

    for c in range(3):
        channel = arr[:, :, c].astype(np.uint8)

        nii = nib.Nifti1Image(channel, affine=affine)
        nii.set_data_dtype(np.uint8)

        channel_path = dst_path.parent / f"{dst_path.name}_{c:04d}.nii.gz"
        nib.save(nii, str(channel_path))

        written_paths.append(channel_path)

    return written_paths[0]


def _read_first_nifti_from_tar(tar_path):
    tar_path = Path(tar_path)

    with tarfile.open(tar_path, "r:*") as tar:
        members = [
            m
            for m in tar.getmembers()
            if m.isfile() and m.name.lower().endswith(".nii.gz")
        ]

        if len(members) == 0:
            raise ValueError(f"No .nii.gz found inside tar mask: {tar_path}")

        file_obj = tar.extractfile(members[0])

        if file_obj is None:
            raise ValueError(f"Could not read .nii.gz inside tar mask: {tar_path}")

        nii_gz_bytes = file_obj.read()

    with gzip.GzipFile(fileobj=io.BytesIO(nii_gz_bytes), mode="rb") as gz:
        nii_bytes = gz.read()

    img = nib.Nifti1Image.from_bytes(nii_bytes)

    return img


def write_video_lv_binary_mask_from_tar(src_path, dst_path):
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    img = _read_first_nifti_from_tar(src_path)
    mask = np.asanyarray(img.dataobj)

    binary = (mask == VIDEO_LV_LABEL).astype(np.uint8)

    out = nib.Nifti1Image(binary, affine=img.affine, header=img.header)
    out.set_data_dtype(np.uint8)

    nib.save(out, str(dst_path))

    return True


def prepare_video_dataset(data_root, nnunet_raw, test=False, num_processes=None):
    log_section("Processing Task 3 VIDEO")

    rows = collect_video_files(data_root)

    if len(rows) == 0:
        log_warn("No files found for VIDEO")
        return None

    file_df = pd.DataFrame(rows)
    print_group_counts(file_df, ["split", "file_type"], "Collected files")
    
    sample_df = build_sample_dataframe(rows)
    sample_df = reassign_training_case_names_sequentially(sample_df)
    print_group_counts(sample_df, ["split"], "Prepared samples")

    dataset_dir = write_dataset(
        sample_df=sample_df,
        dataset_id=DATASET_ID,
        output_dir=nnunet_raw,
        data_type="video",
        test=test,
        fixed_labels=VIDEO_BINARY_LV_LABELS,
        write_image_fn=write_video_rgb_image_as_nnunet_channels,
        write_real_mask_fn=write_video_lv_binary_mask_from_tar,
        write_dummy_mask_fn=write_dummy_mask_from_image,
        description_extra="Task 3 VIDEO RGB dataset.",
        num_processes=num_processes,
    )

    return dataset_dir


@app.command()
def main(
    data_root: Path = typer.Option(
        Path("dirs/data_storage/raw/MVAA"),
        "--data-root",
        help="MVAA root directory containing reference_data/ and optional images/.",
    ),
    output_dir: Path = typer.Option(
        Path("dirs/data_storage/processed/MVAA_nnUNET"),
        "--output-dir",
        help="Final nnU-Net root containing nnUNet_raw, nnUNet_preprocessed, and nnUNet_results.",
    ),
    test: bool = typer.Option(
        False,
        "--test",
        help="Prepare only a few samples per split.",
    ),
    num_processes: int = typer.Option(
        os.cpu_count() or 1,
        "--num-processes",
        "-np",
        help="Number of workers for raw writing and nn-U-Net preprocessing.",
    ),
):
    output_dir = Path(output_dir).resolve()
    num_processes = max(1, int(num_processes))

    nnunet_raw = output_dir / "nnUNet_raw"
    nnunet_preprocessed = output_dir / "nnUNet_preprocessed"
    nnunet_results = output_dir / "nnUNet_results"

    output_dir.mkdir(parents=True, exist_ok=True)
    nnunet_raw.mkdir(parents=True, exist_ok=True)
    nnunet_preprocessed.mkdir(parents=True, exist_ok=True)
    nnunet_results.mkdir(parents=True, exist_ok=True)

    log_info(f"data_root           : {data_root}")
    log_info(f"output_dir          : {output_dir}")
    log_info(f"test mode           : {test}")
    log_info(f"num_processes       : {num_processes}")
    log_info("preprocess          : True")
    log_info("verify              : True")
    log_info("keep_nnunet_raw     : True")
    log_info("keep_nnunet_results : True")

    set_nnunet_env(
        nnunet_raw=nnunet_raw,
        nnunet_preprocessed=nnunet_preprocessed,
        nnunet_results=nnunet_results,
    )

    dataset_dir = prepare_video_dataset(
        data_root=data_root,
        nnunet_raw=nnunet_raw,
        test=test,
        num_processes=num_processes,
    )

    if dataset_dir is not None:
        run_nnunet_plan_and_preprocess(
            dataset_id=DATASET_ID,
            num_processes=num_processes,
        )

    log_ok(f"Raw nnU-Net dataset is in: {nnunet_raw / DATASET_ID}")
    log_ok(f"Preprocessed dataset is in: {nnunet_preprocessed / DATASET_ID}")
    log_ok(f"nnU-Net results folder is in: {nnunet_results}")


if __name__ == "__main__":
    app()