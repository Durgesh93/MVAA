from pathlib import Path
import os
import shutil
import subprocess
import sys

import pandas as pd
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
    write_nifti_image_copy,
    write_nifti_mask_copy,
    write_dummy_mask_from_image,
    set_nnunet_env,
    reassign_training_case_names_sequentially
)


app = typer.Typer(help="Prepare MVAA Task 2 TEE dataset in nnU-Net format.")



DATASET_ID = "Dataset002_MVAA_TEE_SSL"
PATTERNS = ["t2_tee", "*tee*"]
CASE_SUFFIXES = ["-US.nii.gz", "-label.nii.gz"]


def get_dataset_number(dataset_id):
    return int(dataset_id.replace("Dataset", "")[:3])




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

    log_ok(f"Finished nn-U-Net preprocessing for {dataset_id}")


def collect_tee_files(data_root):
    reference_dir = Path(data_root) / "reference_data"
    task_dirs     = find_task_dirs(reference_dir, PATTERNS)

    rows = []

    if len(task_dirs) == 0:
        log_warn(f"No TEE folders found under {reference_dir}")
        return rows

    for task_dir in task_dirs:
        train_dir = task_dir / "train"
        test_dir = task_dir / "val" / "images"

        for p in sorted(train_dir.glob("*-US.nii.gz")):
            rows.append(
                {
                    "split": "TrL",
                    "file_type": "image",
                    "case_name": make_case_name(p, CASE_SUFFIXES),
                    "file_path": str(p),
                }
            )

        for p in sorted(train_dir.glob("*-label.nii.gz")):
            rows.append(
                {
                    "split": "TrL",
                    "file_type": "mask",
                    "case_name": make_case_name(p, CASE_SUFFIXES),
                    "file_path": str(p),
                }
            )

        for p in sorted(test_dir.glob("*-US.nii.gz")):
            rows.append(
                {
                    "split": "TrU",
                    "file_type": "image",
                    "case_name": make_case_name(p, CASE_SUFFIXES),
                    "file_path": str(p),
                }
            )
            rows.append(
                {
                    "split": "Ts",
                    "file_type": "image",
                    "case_name": make_case_name(p, CASE_SUFFIXES),
                    "file_path": str(p),
                }
            )

    # ------------------------------------------------------------------
    # TEE has no real TrU images.
    # Add one TrU sample by reusing one TrL image without its mask.
    # write_dataset() will create a dummy zero mask for TrU.
    # ------------------------------------------------------------------
    has_tru = any(
        row["split"] == "TrU" and row["file_type"] == "image"
        for row in rows
    )

    if not has_tru:
        trl_images = [
            row for row in rows
            if row["split"] == "TrL" and row["file_type"] == "image"
        ]

        if len(trl_images) > 0:
            sampled = trl_images[0].copy()
            sampled["split"] = "TrU"
            sampled["case_name"] = sampled["case_name"] + "_unlabeled"

            rows.append(sampled)

            log_ok(
                "TEE has no TrU images. Added one TrU sample from TrL image: "
                f"{sampled['file_path']}"
            )
        else:
            log_warn("TEE has no TrL image available to create a TrU sample.")

    return rows


def prepare_tee_dataset(data_root, nnunet_raw, test=False, num_processes=None):
    log_section("Processing Task 2 TEE")

    rows = collect_tee_files(data_root)

    if len(rows) == 0:
        log_warn("No files found for TEE")
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
        data_type="tee",
        test=test,
        write_image_fn=write_nifti_image_copy,
        write_real_mask_fn=write_nifti_mask_copy,
        write_dummy_mask_fn=write_dummy_mask_from_image,
        description_extra="Task 2 TEE dataset. One TrU case is sampled from TrL image when no real TrU images exist.",
        num_processes=num_processes,
    )

    return dataset_dir


@app.command()
def main(
    data_root: Path = typer.Option(
        Path("dirs/data_storage/nnUNet/MVAA_nnUNET"),
        "--data-root",
        help="MVAA root directory containing reference_data/.",
    ),
    output_dir: Path = typer.Option(
        Path("dirs/data_storage/nnUNet"),
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

    dataset_dir = prepare_tee_dataset(
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
