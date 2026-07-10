from pathlib import Path
import json
import shutil
import tarfile
import io
import gzip
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
import typer

TEST_N = 5

METADATA_COLS = [
    "dataid",
    "case_name",
    "split",
    "has_foreground",
    "labels_in_mask",
    "image_shape",
    "mask_shape",
    "image_path",
    "mask_path",
]


# =============================================================================
# Typer logging helpers
# =============================================================================


def log_section(title):
    typer.echo()
    typer.echo("=" * 80)
    typer.secho(title, fg=typer.colors.CYAN, bold=True)
    typer.echo("=" * 80)


def log_info(message):
    typer.secho(f"[INFO] {message}", fg=typer.colors.CYAN)


def log_ok(message):
    typer.secho(f"[OK] {message}", fg=typer.colors.GREEN)


def log_warn(message):
    typer.secho(f"[WARN] {message}", fg=typer.colors.YELLOW)


def log_flag(message):
    typer.secho(f"[FLAG] {message}", fg=typer.colors.MAGENTA)


def log_fix(message):
    typer.secho(f"[FIX] {message}", fg=typer.colors.BLUE)


def print_group_counts(df, group_cols, title):
    typer.echo()
    typer.secho(title, fg=typer.colors.CYAN, bold=True)

    if df.empty:
        log_warn("No rows to show.")
        return

    grouped = df.groupby(group_cols).size().reset_index(name="count")
    typer.echo(grouped.to_string(index=False))


# =============================================================================
# Common helpers
# =============================================================================


def prepare_output_dirs(output_dir, dataset_id):
    dataset_dir = Path(output_dir) / dataset_id

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    (dataset_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "imagesTs").mkdir(parents=True, exist_ok=True)

    return dataset_dir


def copy_file(src_path, dst_path):
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def nifti_shape(path):
    try:
        img = nib.load(str(path))
        return "x".join(str(x) for x in img.shape)
    except Exception:
        return ""


def make_case_name(path, case_suffixes):
    name = Path(path).name

    for suffix in case_suffixes:
        name = name.replace(suffix, "")

    return f"{name}"


def find_task_dirs(reference_dir, patterns):
    dirs = []

    for pattern in patterns:
        dirs.extend(Path(reference_dir).glob(pattern))

    return sorted({p for p in dirs if p.is_dir()})


def labels_from_values(label_values):
    label_values = sorted([int(v) for v in label_values])

    if len(label_values) == 0:
        label_values = [0, 1]

    if 0 not in label_values:
        label_values = [0] + label_values

    labels = {}

    for value in label_values:
        if value == 0:
            labels["background"] = 0
        else:
            labels[f"class_{value}"] = int(value)

    return labels


def read_labels_from_mask_path(mask_path):
    mask_path = Path(mask_path)

    if str(mask_path).lower().endswith(".nii.gz"):
        mask = np.asanyarray(nib.load(str(mask_path)).dataobj)
        return sorted([int(v) for v in np.unique(mask)])

    if str(mask_path).lower().endswith(".tar"):
        try:
            with tarfile.open(mask_path, "r:*") as tar:
                members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(".nii.gz")]

                if len(members) == 0:
                    log_warn(f"No .nii.gz found inside tar: {mask_path}")
                    return []

                file_obj = tar.extractfile(members[0])

                if file_obj is None:
                    log_warn(f"Could not read .nii.gz inside tar: {mask_path}")
                    return []

                nii_gz_bytes = file_obj.read()

            with gzip.GzipFile(fileobj=io.BytesIO(nii_gz_bytes), mode="rb") as gz:
                nii_bytes = gz.read()

            img = nib.Nifti1Image.from_bytes(nii_bytes)
            mask = np.asanyarray(img.dataobj)

            return sorted([int(v) for v in np.unique(mask)])

        except Exception as e:
            log_warn(f"Could not read labels from tar mask {mask_path}: {e}")
            return []

    log_warn(f"Unsupported mask format while detecting labels: {mask_path}")
    return []


def check_mask_foreground(mask_path):
    if mask_path is None or str(mask_path).strip() == "":
        return 0, ""

    labels = read_labels_from_mask_path(mask_path)

    if len(labels) == 0:
        return 0, ""

    has_foreground = int(any(v > 0 for v in labels))
    labels_in_mask = "|".join(str(v) for v in labels)

    return has_foreground, labels_in_mask


def detect_labels_from_sample_df(sample_df):
    labeled_df = sample_df[
        (sample_df["split"] == "TrL")
        & (sample_df["original_mask_path"].notna())
        & (sample_df["original_mask_path"].astype(str).str.strip().ne(""))
    ].copy()

    if labeled_df.empty:
        log_warn(f"No labeled masks found Falling back to binary labels.")
        return {"background": 0, "foreground": 1}

    all_seen_labels = set()
    checked_masks = 0
    masks_with_foreground = 0
    background_only_masks = []

    log_info(f"Scanning all real TrL masks for labels...")

    for _, row in tqdm(labeled_df.iterrows(), total=len(labeled_df), desc=f"Detecting labels"):
        mask_path = Path(row["original_mask_path"])

        if not mask_path.exists():
            log_warn(f"Mask not found: {mask_path}")
            continue

        unique_values = read_labels_from_mask_path(mask_path)

        if len(unique_values) == 0:
            continue

        checked_masks += 1
        all_seen_labels.update(unique_values)

        if any(v > 0 for v in unique_values):
            masks_with_foreground += 1
        else:
            background_only_masks.append(str(mask_path))

    all_seen_labels = sorted([int(v) for v in all_seen_labels])

    if len(all_seen_labels) == 0:
        log_warn(f"No labels detected, Falling back to binary labels.")
        all_seen_labels = [0, 1]

    if 0 not in all_seen_labels:
        log_warn(f"Label 0 not found for Adding background=0.")
        all_seen_labels = [0] + all_seen_labels

    labels = labels_from_values(all_seen_labels)

    log_ok(f"Complete label scan")
    log_info(f"  masks checked          : {checked_masks}")
    log_info(f"  masks with foreground  : {masks_with_foreground}")
    log_info(f"  background-only masks  : {len(background_only_masks)}")
    log_info(f"  raw labels found       : {all_seen_labels}")
    log_info("  detected labels:")

    for name, value in labels.items():
        log_info(f"    {name}: {value}")

    if background_only_masks:
        log_info("  background-only examples:")
        for p in background_only_masks[:10]:
            log_info(f"    {p}")

    return labels


def build_sample_dataframe(rows):
    file_df = pd.DataFrame(rows)

    image_df = file_df[file_df["file_type"] == "image"].copy()
    mask_df = file_df[file_df["file_type"] == "mask"].copy()

    image_df = image_df.rename(columns={"file_path": "original_image_path"})
    mask_df = mask_df.rename(columns={"file_path": "original_mask_path"})

    image_cols = ["split", "case_name", "original_image_path"]

    mask_cols = ["split", "case_name", "original_mask_path"]

    sample_df = image_df[image_cols].merge(mask_df[mask_cols], on=["split", "case_name"], how="left")

    sample_df["original_image_path"] = sample_df["original_image_path"].fillna("")
    sample_df["original_mask_path"] = sample_df["original_mask_path"].fillna("")
    return sample_df


def reassign_training_case_names_sequentially(sample_df):
    """
    Reassign case_name sequentially only for training cases.

    Prefix is detected automatically from existing case_name.
    TrL and TrU are renamed together.
    Ts keeps original case_name.
    """

    sample_df = sample_df.copy()

    split_order = {"TrL": 0, "TrU": 1, "Ts": 2}

    sample_df["_split_order"] = sample_df["split"].map(split_order)

    sample_df = sample_df.sort_values(["_split_order", "case_name"]).reset_index(drop=True)

    train_mask = sample_df["split"].isin(["TrL", "TrU"])
    n_train = train_mask.sum()

    train_case_names = sample_df.loc[train_mask, "case_name"].drop_duplicates().tolist()

    if len(train_case_names) == 0:
        raise RuntimeError("No TrL/TrU cases found for renaming.")

    sample_df.loc[train_mask, "case_name"] = [f"{i:04d}" for i in range(n_train)]

    sample_df = sample_df.drop(columns=["_split_order"])
    log_ok(f"Training case names reassigned sequentially.")
    log_info("  TrL first, then TrU.")
    log_info("  Ts case names kept unchanged.")
    return sample_df


def channel_names(data_type):
    if data_type == "video":
        return {"0": "red", "1": "green", "2": "blue"}

    return {"0": "image"}


def get_image_paths(dataset_dir, split, case_name, data_type):
    if split in ["TrL", "TrU"]:
        image_dir_rel = Path("imagesTr")
    elif split == "Ts":
        image_dir_rel = Path("imagesTs")
    else:
        raise ValueError(f"Unknown split: {split}")

    if data_type == "video":
        image_base_rel = image_dir_rel / case_name
        image_rel_for_metadata = image_dir_rel / f"{case_name}_0000.nii.gz"
        image_dst_for_writer = dataset_dir / image_base_rel
    else:
        image_rel_for_metadata = image_dir_rel / f"{case_name}_0000.nii.gz"
        image_dst_for_writer = dataset_dir / image_rel_for_metadata

    return image_rel_for_metadata, image_dst_for_writer


def fix_mask_shape_if_needed(image_path, mask_path):

    if mask_path is None or not Path(mask_path).exists():
        return ""

    image_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))

    image = np.squeeze(np.asanyarray(image_nii.dataobj))
    mask = np.squeeze(np.asanyarray(mask_nii.dataobj))

    if image.shape == mask.shape:
        return "x".join(str(x) for x in mask.shape)

    swapped_mask = mask.swapaxes(0, 1)

    if image.ndim == mask.ndim and image.shape == swapped_mask.shape:
        log_fix(f"Transposing mask to match image: " f"image={image.shape}, mask={mask.shape}, mask_path={mask_path}")

        fixed_mask = swapped_mask.astype(mask.dtype)
        fixed_nii = nib.Nifti1Image(fixed_mask, affine=image_nii.affine)
        nib.save(fixed_nii, str(mask_path))

        return "x".join(str(x) for x in fixed_mask.shape)

    raise ValueError(
        f"Image and mask shape mismatch cannot be fixed automatically: "
        f"image={image.shape}, mask={mask.shape}, mask_path={mask_path}"
    )


# =============================================================================
# Generic image/mask writers
# =============================================================================


def write_nifti_image_copy(src_path, dst_path):
    copy_file(src_path, dst_path)
    return dst_path


def write_nifti_mask_copy(src_path, dst_path):
    copy_file(src_path, dst_path)
    return True


def write_dummy_mask_from_image(image_path, mask_path):
    img = nib.load(str(image_path))
    dummy = np.zeros(img.shape, dtype=np.uint8)

    dummy_nii = nib.Nifti1Image(dummy, affine=img.affine, header=img.header)

    dummy_nii.set_data_dtype(np.uint8)

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(dummy_nii, str(mask_path))


# =============================================================================
# Generic dataset writer
# =============================================================================


def write_one_sample(
    dataid, row, dataset_dir, data_type, write_image_fn, write_real_mask_fn, write_dummy_mask_fn, fixed_labels=None
):
    split = row["split"]
    case_name = row["case_name"]

    src_image = Path(row["original_image_path"])

    src_mask = Path(row["original_mask_path"]) if row["original_mask_path"] != "" else None

    image_rel, dst_image_for_writer = get_image_paths(
        dataset_dir=dataset_dir, split=split, case_name=case_name, data_type=data_type
    )

    mask_rel = None
    dst_mask = None

    if split in ["TrL", "TrU"]:
        mask_rel = Path("labelsTr") / f"{case_name}.nii.gz"
        dst_mask = dataset_dir / mask_rel

    primary_image_path = write_image_fn(src_path=src_image, dst_path=dst_image_for_writer)

    image_shape = nifti_shape(primary_image_path)
    mask_shape = ""
    has_foreground = 0
    labels_in_mask = ""

    if split == "TrL":
        if src_mask is None or not src_mask.exists():
            log_warn(f"Missing mask for labeled case: {src_image}")
        else:
            ok = write_real_mask_fn(src_mask, dst_mask)

            if ok:
                mask_shape = fix_mask_shape_if_needed(image_path=primary_image_path, mask_path=dst_mask)

                has_foreground, labels_in_mask = check_mask_foreground(dst_mask)

                if fixed_labels is not None:
                    allowed_values = set(int(v) for v in fixed_labels.values())
                    mask_values = set(int(v) for v in labels_in_mask.split("|") if str(v).strip() != "")

                    unknown_values = sorted(mask_values - allowed_values)

                    if unknown_values:
                        log_flag(f"Mask has labels outside fixed_labels: " f"{unknown_values}, mask={dst_mask}")

                if has_foreground == 0:
                    log_flag(f"Background-only TrL mask: {dst_mask}")

    elif split == "TrU":
        write_dummy_mask_fn(primary_image_path, dst_mask)

        mask_shape = fix_mask_shape_if_needed(image_path=primary_image_path, mask_path=dst_mask)

        has_foreground = 0
        labels_in_mask = "0"

    elif split == "Ts":
        pass

    else:
        raise ValueError(f"Unknown split: {split}")

    return {
        "dataid": int(dataid),
        "data_type": data_type,
        "case_name": case_name,
        "split": split,
        "is_labeled": int(split == "TrL"),
        "has_foreground": has_foreground,
        "labels_in_mask": labels_in_mask,
        "image_shape": image_shape,
        "mask_shape": mask_shape,
        "image_path": str(image_rel),
        "mask_path": "" if mask_rel is None else str(mask_rel),
    }


def write_dataset(
    sample_df,
    dataset_id,
    output_dir,
    data_type,
    test=False,
    write_image_fn=None,
    write_real_mask_fn=None,
    write_dummy_mask_fn=None,
    description_extra="",
    num_processes=None,
    fixed_labels=None,
):
    dataset_dir = prepare_output_dirs(output_dir, dataset_id)

    if test:
        sample_df = sample_df.groupby("split", group_keys=False).head(TEST_N).reset_index(drop=True)

    if num_processes is None:
        num_processes = os.cpu_count() or 1

    num_processes = max(1, int(num_processes))

    log_info(f"Raw dataset writing workers: {num_processes}")

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------
    if fixed_labels is None:
        labels = detect_labels_from_sample_df(sample_df=sample_df)
    else:
        labels = fixed_labels

    metadata_rows = []
    rows_for_parallel = list(sample_df.iterrows())

    if num_processes == 1:
        for dataid, row in tqdm(rows_for_parallel, total=len(rows_for_parallel), desc=f"Writing {dataset_id}"):
            metadata_row = write_one_sample(
                dataid=dataid,
                row=row,
                dataset_dir=dataset_dir,
                data_type=data_type,
                write_image_fn=write_image_fn,
                write_real_mask_fn=write_real_mask_fn,
                write_dummy_mask_fn=write_dummy_mask_fn,
                fixed_labels=fixed_labels,
            )

            metadata_rows.append(metadata_row)

    else:
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(
                    write_one_sample,
                    dataid,
                    row,
                    dataset_dir,
                    data_type,
                    write_image_fn,
                    write_real_mask_fn,
                    write_dummy_mask_fn,
                    fixed_labels,
                )
                for dataid, row in rows_for_parallel
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Writing {dataset_id}"):
                metadata_rows.append(future.result())

    metadata_df = pd.DataFrame(metadata_rows)

    if metadata_df.empty:
        metadata_df = pd.DataFrame(columns=METADATA_COLS)
    else:
        metadata_df = metadata_df[METADATA_COLS]
        metadata_df = metadata_df.sort_values("dataid").reset_index(drop=True)

    counts = {"TrL": 0, "TrU": 0, "Ts": 0}

    ssl_case_ids = {"TrL": [], "TrU": [], "Ts": []}

    # ------------------------------------------------------------
    # Count unique cases, not image/mask rows
    # ------------------------------------------------------------
    for split in ["TrL", "TrU", "Ts"]:
        case_names = metadata_df.loc[metadata_df["split"] == split, "case_name"].drop_duplicates().tolist()

        counts[split] = len(case_names)
        ssl_case_ids[split] = case_names

    metadata_df.to_csv(dataset_dir / "metadata.csv", index=False)

    dataset_json = {
        "channel_names": channel_names(data_type),
        "labels": labels,
        "numTraining": int(counts["TrL"] + counts["TrU"]),
        "file_ending": ".nii.gz",
        "name": dataset_id,
        "description": (
            "MVAA semi-supervised nnU-Net dataset. "
            "TrL has real masks. TrU has dummy zero masks. "
            f"{description_extra}"
        ),
        "ssl_counts": {"TrL": int(counts["TrL"]), "TrU": int(counts["TrU"]), "Ts": int(counts["Ts"])},
        "ssl_case_ids": {"TrL": ssl_case_ids["TrL"], "TrU": ssl_case_ids["TrU"], "Ts": ssl_case_ids["Ts"]},
    }

    with open(dataset_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4, ensure_ascii=False)

    log_ok(f"Prepared dataset: {dataset_dir}")

    typer.echo()
    typer.secho(f"{dataset_id} summary", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  TrL labeled training   : {counts['TrL']}")
    typer.echo(f"  TrU unlabeled training : {counts['TrU']}")
    typer.echo(f"  Ts test                : {counts['Ts']}")
    typer.echo(f"  imagesTr              : {len(list((dataset_dir / 'imagesTr').glob('*')))}")
    typer.echo(f"  labelsTr              : {len(list((dataset_dir / 'labelsTr').glob('*')))}")
    typer.echo(f"  imagesTs              : {len(list((dataset_dir / 'imagesTs').glob('*')))}")
    typer.echo(f"  metadata              : {dataset_dir / 'metadata.csv'}")
    typer.echo(f"  dataset.json          : {dataset_dir / 'dataset.json'}")

    return dataset_dir


def set_nnunet_env(nnunet_raw, nnunet_preprocessed, nnunet_results):
    nnunet_raw = Path(nnunet_raw).resolve()
    nnunet_preprocessed = Path(nnunet_preprocessed).resolve()
    nnunet_results = Path(nnunet_results).resolve()

    nnunet_raw.mkdir(parents=True, exist_ok=True)
    nnunet_preprocessed.mkdir(parents=True, exist_ok=True)
    nnunet_results.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(nnunet_raw)
    os.environ["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    os.environ["nnUNet_results"] = str(nnunet_results)

    log_info(f"nnUNet_raw          : {os.environ['nnUNet_raw']}")
    log_info(f"nnUNet_preprocessed : {os.environ['nnUNet_preprocessed']}")
    log_info(f"nnUNet_results      : {os.environ['nnUNet_results']}")
