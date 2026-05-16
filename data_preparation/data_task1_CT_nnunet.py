from pathlib import Path

import pandas as pd
import typer

from utils import (
    log_section,
    log_info,
    log_warn,
    print_group_counts,
    make_case_name,
    find_task_dirs,
    build_sample_dataframe,
    reassign_case_names_sequentially,
    write_dataset,
    write_nifti_image_copy,
    write_nifti_mask_copy,
    write_dummy_mask_from_image,
)


app = typer.Typer(help="Prepare MVAA Task 1 CT dataset in nnU-Net format.")


DATA_TYPE = "ct"
DATASET_ID = "Dataset001_MVAA_CT_SSL"
PREFIX = "ct"
PATTERNS = ["t1_ct", "*ct*"]
CASE_SUFFIXES = ["-seg.nii.gz", ".nii.gz"]


def collect_ct_files(data_root):
    reference_dir = Path(data_root) / "reference_data"
    task_dirs = find_task_dirs(reference_dir, PATTERNS)

    rows = []

    if len(task_dirs) == 0:
        log_warn(f"No CT folders found under {reference_dir}")
        return rows

    for task_dir in task_dirs:
        for p in sorted((task_dir / "train" / "labeled" / "images").glob("*.nii.gz")):
            rows.append({
                "data_type": DATA_TYPE,
                "task_id": task_dir.name,
                "split": "TrL",
                "file_type": "image",
                "case_name": make_case_name(p, PREFIX, CASE_SUFFIXES),
                "file_path": str(p),
            })

        for p in sorted((task_dir / "train" / "labeled" / "labels").glob("*-seg.nii.gz")):
            rows.append({
                "data_type": DATA_TYPE,
                "task_id": task_dir.name,
                "split": "TrL",
                "file_type": "mask",
                "case_name": make_case_name(p, PREFIX, CASE_SUFFIXES),
                "file_path": str(p),
            })

        for p in sorted((task_dir / "train" / "unlabeled").glob("*.nii.gz")):
            rows.append({
                "data_type": DATA_TYPE,
                "task_id": task_dir.name,
                "split": "TrU",
                "file_type": "image",
                "case_name": make_case_name(p, PREFIX, CASE_SUFFIXES),
                "file_path": str(p),
            })

        for p in sorted((task_dir / "val" / "images").glob("*.nii.gz")):
            rows.append({
                "data_type": DATA_TYPE,
                "task_id": task_dir.name,
                "split": "Ts",
                "file_type": "image",
                "case_name": make_case_name(p, PREFIX, CASE_SUFFIXES),
                "file_path": str(p),
            })

    return rows


def prepare_ct_dataset(data_root, output_dir, test=False):
    log_section("Processing Task 1 CT")

    rows = collect_ct_files(data_root)

    if len(rows) == 0:
        log_warn("No files found for CT")
        return

    file_df = pd.DataFrame(rows)
    print_group_counts(file_df, ["split", "file_type"], "Collected files")

    sample_df = build_sample_dataframe(rows)
    sample_df = reassign_case_names_sequentially(sample_df, prefix=PREFIX)

    print_group_counts(sample_df, ["split", "is_labeled"], "Prepared samples")

    write_dataset(
        sample_df=sample_df,
        dataset_id=DATASET_ID,
        data_type=DATA_TYPE,
        output_dir=output_dir,
        test=test,
        fixed_labels=None,
        write_image_fn=write_nifti_image_copy,
        write_real_mask_fn=write_nifti_mask_copy,
        write_dummy_mask_fn=write_dummy_mask_from_image,
        description_extra="Task 1 CT dataset.",
    )


@app.command()
def main(
    data_root: Path = typer.Option(
        Path("dirs/data_storage/raw/MVAA"),
        "--data-root",
        help="MVAA root directory containing reference_data/.",
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

    prepare_ct_dataset(
        data_root=data_root,
        output_dir=output_dir,
        test=test,
    )


if __name__ == "__main__":
    app()
