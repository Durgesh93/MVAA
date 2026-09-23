"""
Prepare the MVSeg2023 TEE dataset in nnU-Net format.

Replaces data_task2_TEE_nnunet.py, which sourced a lossy copy of this
same data (reference_data/t2_tee: the 105 labeled train cases plus 20 of
the 30 val *images* with their labels stripped) and then split it into
the SSL TrL/TrU/Ts scheme. The full MVSeg2023 release is fully labeled --
105 train + 30 val + 40 test, every case with a real mask -- so there is
no unlabeled partition to model and none of that scaffolding is needed.

Mapping (MVSeg2023 -> nnU-Net):
    imagesTr/labelsTr  <- train (105) + val (30) = 135
    imagesTs/labelsTs  <- test (40), fully labeled and scorable

Which split a case came from is preserved in dataset.json's `case_ids`,
which is how the datamodule recovers MVSeg2023's own train/val boundary.
That boundary is the only one the dataset defines, so nothing here
re-splits it.

Case names are the dataset's own (train_001, val_001, test_001), not
renumbered -- unlike the old script's sequential reassignment, which made
outputs impossible to trace back to a source volume without metadata.csv.

Labels (per the MVSeg2023 dataset card):
    0 background, 1 posterior leaflet, 2 anterior leaflet
"""

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import nibabel as nib
import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

from data_preparation.utils import (
    log_section,
    log_info,
    log_ok,
    log_warn,
    log_flag,
    print_group_counts,
    copy_file,
    set_nnunet_env,
)

app = typer.Typer(help="Prepare the MVSeg2023 TEE dataset in nnU-Net format.")


DATASET_ID = "Dataset002_MVAA_TEE_SSL_MVSeg"

# MVSeg2023 ships one channel (B-mode ultrasound). nnU-Net normalizes per
# channel using the scheme its planner picks from the fingerprint; the
# name is only a label.
CHANNEL_NAMES = {"0": "US"}

LABELS = {"background": 0, "posterior_leaflet": 1, "anterior_leaflet": 2}

# Splits pooled into imagesTr/labelsTr (train first).
TRAIN_SPLITS = ("train", "val")
TEST_SPLIT = "test"

IMAGE_SUFFIX = "-US.nii.gz"
LABEL_SUFFIX = "-label.nii.gz"

METADATA_COLS = [
    "case_name",
    "source_split",
    "nnunet_split",
    "shape",
    "spacing",
    "labels_in_mask",
    "pct_foreground",
    "image_path",
    "label_path",
]


def get_dataset_number(dataset_id):
    return int(dataset_id.replace("Dataset", "")[:3])


def collect_cases(src_root):
    """
    Discover every case under src_root/{train,val,test}, pairing each
    -US.nii.gz with its -label.nii.gz. Raises if any image has no
    matching label: unlike the SSL script this pipeline has no dummy-mask
    path, so a missing label is a real error, not a case to synthesize.
    """

    src_root = Path(src_root)

    rows = []

    for source_split in (*TRAIN_SPLITS, TEST_SPLIT):
        split_dir = src_root / source_split

        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Missing split folder: {split_dir}\n"
                f"Expected MVSeg2023 extracted as {src_root}/{{train,val,test}}/"
            )

        images = sorted(split_dir.glob(f"*{IMAGE_SUFFIX}"))

        if len(images) == 0:
            raise FileNotFoundError(f"No {IMAGE_SUFFIX} files in {split_dir}")

        for image_path in images:
            case_name = image_path.name[: -len(IMAGE_SUFFIX)]
            label_path = split_dir / f"{case_name}{LABEL_SUFFIX}"

            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for case '{case_name}': {label_path}")

            rows.append(
                {
                    "case_name": case_name,
                    "source_split": source_split,
                    "nnunet_split": "Ts" if source_split == TEST_SPLIT else "Tr",
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                }
            )

    return rows


def prepare_output_dirs(output_dir, dataset_id):
    """
    Create a clean raw dataset folder, labelsTs included.

    labelsTs is not part of nnU-Net's own training contract (it ignores
    it), but MVSeg2023's test split is fully labeled and keeping the
    masks next to the images is what makes the test set scorable later
    rather than prediction-only.
    """

    dataset_dir = Path(output_dir) / dataset_id

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (dataset_dir / sub).mkdir(parents=True, exist_ok=True)

    return dataset_dir


def write_one_case(row, dataset_dir):
    """
    Copy one image/label pair into the nnU-Net layout and report what was
    written. Files are copied verbatim (no re-encoding): image and label
    geometry is verified rather than rewritten, so whatever spacing and
    affine MVSeg2023 shipped is exactly what nnU-Net's planner fingerprints.
    """

    case_name = row["case_name"]
    image_dir = "imagesTr" if row["nnunet_split"] == "Tr" else "imagesTs"
    label_dir = "labelsTr" if row["nnunet_split"] == "Tr" else "labelsTs"

    # nnU-Net requires the _0000 channel suffix on images, and a bare
    # case name on the corresponding segmentation.
    dst_image = dataset_dir / image_dir / f"{case_name}_0000.nii.gz"
    dst_label = dataset_dir / label_dir / f"{case_name}.nii.gz"

    copy_file(row["image_path"], dst_image)
    copy_file(row["label_path"], dst_label)

    image = nib.load(str(dst_image))
    label = nib.load(str(dst_label))

    label_array = np.asanyarray(label.dataobj)
    values, counts = np.unique(label_array, return_counts=True)

    problems = []

    if image.shape != label.shape:
        problems.append(f"shape mismatch image={image.shape} label={label.shape}")

    if not np.allclose(image.affine, label.affine, atol=1e-4):
        problems.append("affine mismatch between image and label")

    unexpected = sorted(set(int(v) for v in values) - set(LABELS.values()))

    if unexpected:
        problems.append(f"unexpected label values {unexpected}")

    foreground = int(sum(c for v, c in zip(values, counts) if int(v) != 0))

    return {
        "case_name": case_name,
        "source_split": row["source_split"],
        "nnunet_split": row["nnunet_split"],
        "shape": "x".join(str(x) for x in image.shape),
        "spacing": "x".join(f"{s:.4f}" for s in image.header.get_zooms()[:3]),
        "labels_in_mask": "|".join(str(int(v)) for v in values),
        "pct_foreground": round(100.0 * foreground / label_array.size, 4),
        "image_path": str(Path(image_dir) / dst_image.name),
        "label_path": str(Path(label_dir) / dst_label.name),
        "problems": "; ".join(problems),
    }


def write_dataset(rows, dataset_id, output_dir, num_processes):
    dataset_dir = prepare_output_dirs(output_dir, dataset_id)

    log_info(f"Raw dataset writing workers: {num_processes}")

    metadata_rows = []

    if num_processes == 1:
        for row in tqdm(rows, desc=f"Writing {dataset_id}"):
            metadata_rows.append(write_one_case(row, dataset_dir))
    else:
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(write_one_case, row, dataset_dir) for row in rows]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Writing {dataset_id}"):
                metadata_rows.append(future.result())

    metadata_df = pd.DataFrame(metadata_rows).sort_values("case_name").reset_index(drop=True)

    flagged = metadata_df[metadata_df["problems"] != ""]

    for _, bad in flagged.iterrows():
        log_flag(f"{bad['case_name']}: {bad['problems']}")

    if len(flagged) > 0:
        raise RuntimeError(
            f"{len(flagged)} case(s) failed geometry/label validation -- see [FLAG] lines above. "
            "Refusing to write dataset.json for an inconsistent dataset."
        )

    n_train = int((metadata_df["nnunet_split"] == "Tr").sum())
    n_test = int((metadata_df["nnunet_split"] == "Ts").sum())

    dataset_json = {
        "channel_names": CHANNEL_NAMES,
        "labels": LABELS,
        "numTraining": n_train,
        "numTest": n_test,
        "file_ending": ".nii.gz",
        "name": dataset_id,
        "description": (
            "MVSeg2023 (MICCAI 2023) 3D TEE mitral valve leaflet segmentation. "
            "Fully supervised: every case has a real mask. "
            "imagesTr pools the official train (105) and val (30) splits, whose "
            "membership is recorded in case_ids below. imagesTs is the official "
            "test split (40), with labelsTs kept so it stays scorable. "
            "Labels: 1 = posterior leaflet, 2 = anterior leaflet. "
            "Source: https://www.synapse.org/MVSeg2023 (CC BY-NC-ND 4.0)."
        ),
        # Kept so the split a case came from is recoverable from
        # dataset.json alone, without re-reading metadata.csv.
        "case_ids": {
            split: metadata_df.loc[metadata_df["source_split"] == split, "case_name"].tolist()
            for split in (*TRAIN_SPLITS, TEST_SPLIT)
        },
    }

    metadata_df[METADATA_COLS].to_csv(dataset_dir / "metadata.csv", index=False)

    with open(dataset_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4, ensure_ascii=False)

    log_ok(f"Prepared dataset: {dataset_dir}")

    typer.echo()
    typer.secho(f"{dataset_id} summary", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  imagesTr / labelsTr : {n_train}")
    typer.echo(f"  imagesTs / labelsTs : {n_test}")

    for split in (*TRAIN_SPLITS, TEST_SPLIT):
        subset = metadata_df[metadata_df["source_split"] == split]
        typer.echo(
            f"    {split:<6s} : {len(subset):>3d} cases | "
            f"mean foreground {subset['pct_foreground'].mean():.2f}%"
        )

    typer.echo(f"  metadata            : {dataset_dir / 'metadata.csv'}")
    typer.echo(f"  dataset.json        : {dataset_dir / 'dataset.json'}")

    return dataset_dir


def run_nnunet_plan_and_preprocess(dataset_id, num_processes):
    dataset_number = get_dataset_number(dataset_id)

    exe = shutil.which("nnUNetv2_plan_and_preprocess")

    if exe is not None:
        cmd = [exe, "-d", str(dataset_number), "-np", str(num_processes), "-npfp", str(num_processes)]
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

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)

    assert process.stdout is not None

    for line in process.stdout:
        typer.echo(line.rstrip())

    return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

    log_ok(f"Finished nnU-Net preprocessing for {dataset_id}")


@app.command()
def main(
    src: Path = typer.Option(
        Path("dirs/data_storage/MVSeg2023"),
        "--src",
        help="Extracted MVSeg2023 root containing train/, val/, test/.",
    ),
    output_dir: Path = typer.Option(
        Path("dirs/data_storage/nnUNet"),
        "--output-dir",
        help="nnU-Net root containing nnUNet_raw, nnUNet_preprocessed, and nnUNet_results.",
    ),
    dataset_id: str = typer.Option(DATASET_ID, "--dataset-id", help="Target nnU-Net dataset folder name."),
    preprocess: bool = typer.Option(True, "--preprocess/--no-preprocess", help="Run nnUNetv2_plan_and_preprocess."),
    clean: bool = typer.Option(
        True,
        "--clean/--no-clean",
        help="Delete the existing preprocessed folder first (see clean_preprocessed). "
        "--no-clean keeps it, which can leave stale gt_segmentations behind.",
    ),
    num_processes: int = typer.Option(
        os.cpu_count() or 1,
        "--num-processes",
        "-np",
        help="Workers for raw writing and nnU-Net preprocessing.",
    ),
):
    log_section("Preparing MVSeg2023 (TEE) in nnU-Net format")

    src = Path(src).resolve()
    output_dir = Path(output_dir).resolve()
    num_processes = max(1, int(num_processes))

    nnunet_raw = output_dir / "nnUNet_raw"
    nnunet_preprocessed = output_dir / "nnUNet_preprocessed"
    nnunet_results = output_dir / "nnUNet_results"

    log_info(f"source          : {src}")
    log_info(f"output_dir      : {output_dir}")
    log_info(f"dataset_id      : {dataset_id}")
    log_info(f"preprocess      : {preprocess}")
    log_info(f"clean           : {clean}")
    log_info(f"num_processes   : {num_processes}")

    set_nnunet_env(
        nnunet_raw=nnunet_raw, nnunet_preprocessed=nnunet_preprocessed, nnunet_results=nnunet_results
    )

    rows = collect_cases(src)

    file_df = pd.DataFrame(rows)
    print_group_counts(file_df, ["source_split", "nnunet_split"], "Collected cases")

    write_dataset(rows=rows, dataset_id=dataset_id, output_dir=nnunet_raw, num_processes=num_processes)

    if preprocess:
        if clean:
            clean_preprocessed(nnunet_preprocessed=nnunet_preprocessed, dataset_id=dataset_id)

        run_nnunet_plan_and_preprocess(dataset_id=dataset_id, num_processes=num_processes)
    else:
        log_warn("Skipped plan_and_preprocess (--no-preprocess).")

    log_ok(f"Raw nnU-Net dataset is in: {nnunet_raw / dataset_id}")
    log_ok(f"Preprocessed dataset is in: {nnunet_preprocessed / dataset_id}")


if __name__ == "__main__":
    app()
