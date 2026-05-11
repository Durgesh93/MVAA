from pathlib import Path
import argparse
import json
import shutil
import tarfile
import zipfile
import tempfile

import nibabel as nib
import pandas as pd
from PIL import Image
from tqdm import tqdm


TEST_N = 5
VIDEO_UNLABELED_TRAIN_RATIO = 0.70


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare MVAA SSL dataset with TrL, TrU, Ts"
    )

    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("dirs/data_storage/raw/MVAA"),
        help="MVAA root directory containing reference_data/ and images/",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dirs/data_storage/processed/MVAA_ssl"),
        help="Output directory",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Prepare only {TEST_N} samples per split",
    )

    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create each dataset directly inside a zip file instead of folders",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "ct": {
        "dataset_id": "Dataset001_MVAA_CT_SSL",
        "patterns": ["t1_ct", "*ct*"],
        "prefix": "ct",
        "image_ending": ".nii.gz",
        "mask_ending": ".nii.gz",
    },
    "tee": {
        "dataset_id": "Dataset002_MVAA_TEE_SSL",
        "patterns": ["t2_tee", "*tee*"],
        "prefix": "tee",
        "image_ending": ".nii.gz",
        "mask_ending": ".nii.gz",
    },
    "video": {
        "dataset_id": "Dataset003_MVAA_VIDEO_SSL",
        "patterns": ["t3_vid", "*vid*", "*video*"],
        "prefix": "video",
        "image_ending": ".png",
        "mask_ending": ".nii.gz",
    },
}


METADATA_COLS = [
    "dataid",
    "data_type",
    "task_id",
    "case_id",
    "split",
    "image_shape",
    "mask_shape",
    "image_path",
    "mask_path",
]


VIDEO_LABELS = {
    "background": 0,
    "atrial_retractor": 1,
    "dissecting_forceps": 2,
    "scissors": 3,
    "needle_holder": 4,
    "sharp_knife": 5,
    "wire_organizer": 6,
    "suture": 7,
    "needle": 8,
    "atrial_inner_surface": 9,
    "mitral_valve": 10,
    "ventricle": 11,
    "blood": 12,
    "irrelevant_frame": 13,
    "artificial_valve": 14,
    "sizer": 15,
    "pledget": 16,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def path_has_part(path, part_name):
    return part_name.lower() in [p.lower() for p in path.parts]


def strip_nii_gz(filename):
    return filename.replace(".nii.gz", "")


def ct_case_id(filename):
    case_id = strip_nii_gz(filename)

    for token in ["-seg", "_seg", "-label", "_label", "-mask", "_mask"]:
        case_id = case_id.replace(token, "")

    return case_id


def tee_case_id(filename):
    """
    Examples:
        train_001-US.nii.gz     -> train_001
        train_001-label.nii.gz  -> train_001
        val_001-US.nii.gz       -> val_001
    """

    case_id = strip_nii_gz(filename)

    for token in ["-US", "_US", "-label", "_label", "-Label", "_Label"]:
        case_id = case_id.replace(token, "")

    return case_id


def video_case_id(filename):
    case_id = filename

    remove_tokens = [
        "_png_Label.tar",
        "_png_label.tar",
        "_Label.tar",
        "_label.tar",
        ".tar",
        ".png",
        ".jpg",
        ".jpeg",
    ]

    for token in remove_tokens:
        case_id = case_id.replace(token, "")

    return case_id


def make_case_name(prefix, dataid):
    return f"{prefix}_{int(dataid):06d}"


def get_task_dirs(reference_dir, data_type):
    task_dirs = []

    for pattern in DATASETS[data_type]["patterns"]:
        task_dirs.extend(reference_dir.glob(pattern))

    return sorted({p for p in task_dirs if p.is_dir()})


def get_split_dirs(task_dir):
    return [
        p for p in sorted(task_dir.iterdir())
        if p.is_dir() and p.name in ["train", "val", "test"]
    ]


def get_nifti_shape(path):
    try:
        nii = nib.load(str(path))
        return "x".join(str(x) for x in nii.shape)
    except Exception:
        return ""


def get_png_shape(path):
    try:
        img = Image.open(path)
        return "x".join(str(x) for x in img.size[::-1])
    except Exception:
        return ""


def get_nii_shape_from_bytes(nii_bytes):
    try:
        with tempfile.NamedTemporaryFile(suffix=".nii.gz") as tmp:
            tmp.write(nii_bytes)
            tmp.flush()
            return get_nifti_shape(tmp.name)
    except Exception:
        return ""


def save_metadata_csv(metadata_rows, output_path=None):
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df = metadata_df[METADATA_COLS]

    if output_path is not None:
        metadata_df.to_csv(output_path, index=False)

    return metadata_df


def build_dataset_json(dataset_name, data_type, counts):
    labels = VIDEO_LABELS if data_type == "video" else {
        "background": 0,
        "foreground": 1,
    }

    return {
        "name": dataset_name,
        "data_type": data_type,
        "description": "MVAA semi-supervised segmentation dataset",
        "format": "ssl_trl_tru_ts",
        "image_ending": DATASETS[data_type]["image_ending"],
        "mask_ending": DATASETS[data_type]["mask_ending"],
        "channel_names": {
            "0": "image"
        },
        "labels": labels,
        "splits": {
            "TrL": "labeled training images and masks",
            "TrU": "unlabeled training images",
            "Ts": "test or hidden-label images",
        },
        "counts": {
            "TrL": int(counts.get("TrL", 0)),
            "TrU": int(counts.get("TrU", 0)),
            "Ts": int(counts.get("Ts", 0)),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Video mask extraction from tar
# ─────────────────────────────────────────────────────────────────────────────

def extract_nii_from_tar(tar_path, dst_path):
    """
    Extract .nii.gz mask from video label tar.

    Example tar contents:
        xxx_png_Label.json
        xxx_png_Label.nii.gz
    """

    try:
        with tarfile.open(tar_path, "r:*") as tar:
            members = [
                m for m in tar.getmembers()
                if m.isfile() and m.name.lower().endswith(".nii.gz")
            ]

            if len(members) == 0:
                print(f"[WARN] No .nii.gz mask found in {tar_path}")
                return False

            member = members[0]
            file_obj = tar.extractfile(member)

            if file_obj is None:
                print(f"[WARN] Could not read {member.name} from {tar_path}")
                return False

            dst_path.parent.mkdir(parents=True, exist_ok=True)

            with open(dst_path, "wb") as f:
                shutil.copyfileobj(file_obj, f)

            return True

    except Exception as e:
        print(f"[WARN] Could not extract .nii.gz from {tar_path}: {e}")
        return False


def extract_nii_bytes_from_tar(tar_path):
    """
    Extract .nii.gz mask from video label tar and return bytes.
    """

    try:
        with tarfile.open(tar_path, "r:*") as tar:
            members = [
                m for m in tar.getmembers()
                if m.isfile() and m.name.lower().endswith(".nii.gz")
            ]

            if len(members) == 0:
                print(f"[WARN] No .nii.gz mask found in {tar_path}")
                return None

            member = members[0]
            file_obj = tar.extractfile(member)

            if file_obj is None:
                print(f"[WARN] Could not read {member.name} from {tar_path}")
                return None

            return file_obj.read()

    except Exception as e:
        print(f"[WARN] Could not extract .nii.gz from {tar_path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Collect CT
# ─────────────────────────────────────────────────────────────────────────────

def collect_ct_files(task_dir, split_dir):
    rows = []

    for p in sorted(split_dir.rglob("*.nii.gz")):
        filename = p.name

        if path_has_part(p, "labels"):
            file_type = "mask"
            subset = "labeled"

        elif path_has_part(p, "labeled") and path_has_part(p, "images"):
            file_type = "image"
            subset = "labeled"

        elif path_has_part(p, "unlabeled"):
            file_type = "image"
            subset = "unlabeled"

        elif split_dir.name == "val":
            file_type = "image"
            subset = "unlabeled"

        else:
            file_type = "image"
            subset = "unknown"

        rows.append({
            "data_type": "ct",
            "task_id": task_dir.name,
            "raw_split": split_dir.name,
            "subset": subset,
            "case_id": ct_case_id(filename),
            "file_path": str(p),
            "filename": filename,
            "file_type": file_type,
            "source": "reference_data",
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Collect TEE
# ─────────────────────────────────────────────────────────────────────────────

def collect_tee_files(task_dir, split_dir):
    """
    TEE:
        train_001-US.nii.gz      -> image
        train_001-label.nii.gz   -> mask
    """

    rows = []

    for p in sorted(split_dir.rglob("*.nii.gz")):
        filename = p.name
        lower = filename.lower()

        if "label" in lower:
            file_type = "mask"
            subset = "labeled"
        else:
            file_type = "image"
            subset = "unlabeled" if split_dir.name == "val" else "labeled"

        rows.append({
            "data_type": "tee",
            "task_id": task_dir.name,
            "raw_split": split_dir.name,
            "subset": subset,
            "case_id": tee_case_id(filename),
            "file_path": str(p),
            "filename": filename,
            "file_type": file_type,
            "source": "reference_data",
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Collect Video
# ─────────────────────────────────────────────────────────────────────────────

def collect_video_files(task_dir, split_dir):
    """
    Video:
        .png/.jpg/.jpeg = image
        .tar            = label package containing .nii.gz mask
    """

    rows = []

    for p in sorted(split_dir.rglob("*")):
        if not p.is_file():
            continue

        filename = p.name
        lower = filename.lower()

        if lower.endswith((".png", ".jpg", ".jpeg")):
            file_type = "image"
            subset = "labeled" if split_dir.name == "train" else "unlabeled"

        elif lower.endswith(".tar"):
            file_type = "mask"
            subset = "labeled"

        else:
            continue

        rows.append({
            "data_type": "video",
            "task_id": task_dir.name,
            "raw_split": split_dir.name,
            "subset": subset,
            "case_id": video_case_id(filename),
            "file_path": str(p),
            "filename": filename,
            "file_type": file_type,
            "source": "reference_data",
        })

    return rows


def collect_extra_video_unlabeled_files(data_root):
    """
    Extra video images are inside:
        data_root/images/

    These are split together with video val images:
        70% -> TrU
        30% -> Ts
    """

    rows = []
    extra_video_image_dir = Path(data_root) / "images"

    if not extra_video_image_dir.exists():
        print(f"[WARN] Extra video image dir not found: {extra_video_image_dir}")
        return rows

    for p in sorted(extra_video_image_dir.rglob("*")):
        if not p.is_file():
            continue

        filename = p.name
        lower = filename.lower()

        if not lower.endswith((".png", ".jpg", ".jpeg")):
            continue

        rows.append({
            "data_type": "video",
            "task_id": "t3_vid",
            "raw_split": "external",
            "subset": "unlabeled",
            "case_id": video_case_id(filename),
            "file_path": str(p),
            "filename": filename,
            "file_type": "image",
            "source": "extra_images",
        })

    print(f"[OK] Extra video unlabeled images found: {len(rows)}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Collection router
# ─────────────────────────────────────────────────────────────────────────────

def collect_files_for_data_type(data_root, data_type):
    rows = []
    reference_dir = Path(data_root) / "reference_data"

    task_dirs = get_task_dirs(reference_dir, data_type)

    if len(task_dirs) == 0:
        if data_type != "video":
            raise FileNotFoundError(
                f"No task folders found for {data_type} under {reference_dir}"
            )
        else:
            print(f"[WARN] No reference video task folders found under {reference_dir}")

    for task_dir in task_dirs:
        split_dirs = get_split_dirs(task_dir)

        for split_dir in split_dirs:
            if data_type == "ct":
                rows.extend(collect_ct_files(task_dir, split_dir))

            elif data_type == "tee":
                rows.extend(collect_tee_files(task_dir, split_dir))

            elif data_type == "video":
                rows.extend(collect_video_files(task_dir, split_dir))

    if data_type == "video":
        rows.extend(collect_extra_video_unlabeled_files(data_root))

    file_df = pd.DataFrame(rows)

    if file_df.empty:
        raise FileNotFoundError(f"No files found for {data_type}")

    return file_df


# ─────────────────────────────────────────────────────────────────────────────
# Build sample dataframe
# ─────────────────────────────────────────────────────────────────────────────

def build_sample_dataframe(file_df, data_type):
    image_df = file_df[file_df["file_type"] == "image"].copy()
    mask_df = file_df[file_df["file_type"] == "mask"].copy()

    image_df = image_df.rename(columns={
        "file_path": "original_image_path",
        "filename": "image_filename",
    })

    mask_df = mask_df.rename(columns={
        "file_path": "original_mask_path",
        "filename": "mask_filename",
    })

    image_cols = [
        "data_type",
        "task_id",
        "raw_split",
        "subset",
        "case_id",
        "original_image_path",
        "image_filename",
        "source",
    ]

    mask_cols = [
        "data_type",
        "task_id",
        "raw_split",
        "subset",
        "case_id",
        "original_mask_path",
        "mask_filename",
    ]

    if mask_df.empty:
        sample_df = image_df[image_cols].copy()
        sample_df["original_mask_path"] = ""
        sample_df["mask_filename"] = ""
    else:
        sample_df = image_df[image_cols].merge(
            mask_df[mask_cols],
            on=[
                "data_type",
                "task_id",
                "raw_split",
                "subset",
                "case_id",
            ],
            how="left",
        )

        sample_df["original_mask_path"] = sample_df["original_mask_path"].fillna("")
        sample_df["mask_filename"] = sample_df["mask_filename"].fillna("")

    sample_df["has_mask"] = sample_df["original_mask_path"].ne("")

    if data_type == "video":
        sample_df = assign_video_ssl_split(sample_df)
    else:
        sample_df = assign_medical_ssl_split(sample_df)

    sample_df = sample_df.sort_values(
        ["ssl_split", "task_id", "case_id", "image_filename"]
    ).reset_index(drop=True)

    return sample_df


def assign_medical_ssl_split(sample_df):
    sample_df = sample_df.copy()

    sample_df["ssl_split"] = "TrU"

    sample_df.loc[
        (sample_df["raw_split"] == "train") &
        (sample_df["subset"] == "labeled") &
        (sample_df["has_mask"]),
        "ssl_split"
    ] = "TrL"

    sample_df.loc[
        sample_df["raw_split"].isin(["val", "test"]),
        "ssl_split"
    ] = "Ts"

    return sample_df


def assign_video_ssl_split(sample_df):
    """
    Video:
        train + has mask              -> TrL
        val images + external images  -> 70% TrU, 30% Ts
        train images without mask     -> included in same unlabeled pool
    """

    sample_df = sample_df.copy()
    sample_df["ssl_split"] = ""

    labeled_mask = (
        (sample_df["raw_split"] == "train") &
        (sample_df["has_mask"])
    )
    sample_df.loc[labeled_mask, "ssl_split"] = "TrL"

    unlabeled_idx = sample_df.index[~labeled_mask].tolist()
    unlabeled_idx = sorted(unlabeled_idx)

    n_unlabeled = len(unlabeled_idx)
    n_train_u = int(round(n_unlabeled * VIDEO_UNLABELED_TRAIN_RATIO))

    train_u_idx = unlabeled_idx[:n_train_u]
    ts_idx = unlabeled_idx[n_train_u:]

    sample_df.loc[train_u_idx, "ssl_split"] = "TrU"
    sample_df.loc[ts_idx, "ssl_split"] = "Ts"

    return sample_df


# ─────────────────────────────────────────────────────────────────────────────
# Folder writer
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataset_dirs(dataset_dir):
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    dirs = [
        dataset_dir / "TrL" / "images",
        dataset_dir / "TrL" / "masks",
        dataset_dir / "TrU" / "images",
        dataset_dir / "Ts" / "images",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def copy_file(src_path, dst_path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def write_samples_to_folder(sample_df, output_dir, data_type, test=False):
    config = DATASETS[data_type]

    dataset_id = config["dataset_id"]
    prefix = config["prefix"]

    dataset_dir = output_dir / dataset_id
    prepare_dataset_dirs(dataset_dir)

    sample_df = sample_df.copy()

    if test:
        sample_df = (
            sample_df
            .groupby(["ssl_split"], group_keys=False)
            .head(TEST_N)
            .reset_index(drop=True)
        )

    metadata_rows = []
    counts = {"TrL": 0, "TrU": 0, "Ts": 0}

    for idx, row in tqdm(
        sample_df.iterrows(),
        total=len(sample_df),
        desc=f"Writing folder {dataset_id}"
    ):
        dataid = int(idx)
        ssl_split = row["ssl_split"]
        case_name = make_case_name(prefix, dataid)

        original_image_path = Path(row["original_image_path"])
        original_mask_path = (
            Path(row["original_mask_path"])
            if row["original_mask_path"] != ""
            else None
        )

        if data_type == "video":
            image_out_name = f"{case_name}.png"
            mask_out_name = f"{case_name}.nii.gz"
        else:
            image_out_name = f"{case_name}_0000.nii.gz"
            mask_out_name = f"{case_name}.nii.gz"

        if ssl_split == "TrL":
            image_rel_path = Path("TrL") / "images" / image_out_name
            mask_rel_path = Path("TrL") / "masks" / mask_out_name

        elif ssl_split == "TrU":
            image_rel_path = Path("TrU") / "images" / image_out_name
            mask_rel_path = ""

        elif ssl_split == "Ts":
            image_rel_path = Path("Ts") / "images" / image_out_name
            mask_rel_path = ""

        else:
            raise ValueError(f"Unknown ssl_split: {ssl_split}")

        image_out_path = dataset_dir / image_rel_path
        mask_out_path = dataset_dir / mask_rel_path if mask_rel_path != "" else None

        copy_file(original_image_path, image_out_path)

        if data_type == "video":
            image_shape = get_png_shape(image_out_path)
        else:
            image_shape = get_nifti_shape(image_out_path)

        mask_shape = ""

        if ssl_split == "TrL":
            if original_mask_path is None or not original_mask_path.exists():
                print(f"[WARN] TrL image has no mask: {original_image_path}")

            else:
                if data_type == "video":
                    ok = extract_nii_from_tar(original_mask_path, mask_out_path)
                    if ok:
                        mask_shape = get_nifti_shape(mask_out_path)
                    else:
                        print(f"[WARN] Could not extract video mask: {original_mask_path}")
                else:
                    copy_file(original_mask_path, mask_out_path)
                    mask_shape = get_nifti_shape(mask_out_path)

        counts[ssl_split] += 1

        metadata_rows.append({
            "dataid": dataid,
            "data_type": data_type,
            "task_id": row["task_id"],
            "case_id": row["case_id"],
            "split": ssl_split,
            "image_shape": image_shape,
            "mask_shape": mask_shape,
            "image_path": str(image_rel_path),
            "mask_path": str(mask_rel_path),
        })

    metadata_df = save_metadata_csv(
        metadata_rows,
        output_path=dataset_dir / "metadata.csv",
    )

    dataset_json = build_dataset_json(
        dataset_name=dataset_id,
        data_type=data_type,
        counts=counts,
    )

    with open(dataset_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4, ensure_ascii=False)

    print(f"\n[OK] Prepared folder → {dataset_dir}")
    print(f"  TrL: {counts['TrL']}")
    print(f"  TrU: {counts['TrU']}")
    print(f"  Ts : {counts['Ts']}")
    print(f"  metadata columns: {list(metadata_df.columns)}")

    return dataset_dir


# ─────────────────────────────────────────────────────────────────────────────
# Zip writer
# ─────────────────────────────────────────────────────────────────────────────

def write_samples_to_zip(sample_df, output_dir, data_type, test=False):
    config = DATASETS[data_type]

    dataset_id = config["dataset_id"]
    prefix = config["prefix"]

    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / f"{dataset_id}.zip"

    if zip_path.exists():
        zip_path.unlink()

    sample_df = sample_df.copy()

    if test:
        sample_df = (
            sample_df
            .groupby(["ssl_split"], group_keys=False)
            .head(TEST_N)
            .reset_index(drop=True)
        )

    metadata_rows = []
    counts = {"TrL": 0, "TrU": 0, "Ts": 0}

    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=4,
    ) as zipf:

        for idx, row in tqdm(
            sample_df.iterrows(),
            total=len(sample_df),
            desc=f"Writing zip {dataset_id}"
        ):
            dataid = int(idx)
            ssl_split = row["ssl_split"]
            case_name = make_case_name(prefix, dataid)

            original_image_path = Path(row["original_image_path"])
            original_mask_path = (
                Path(row["original_mask_path"])
                if row["original_mask_path"] != ""
                else None
            )

            if data_type == "video":
                image_out_name = f"{case_name}.png"
                mask_out_name = f"{case_name}.nii.gz"
            else:
                image_out_name = f"{case_name}_0000.nii.gz"
                mask_out_name = f"{case_name}.nii.gz"

            if ssl_split == "TrL":
                image_rel_path = Path("TrL") / "images" / image_out_name
                mask_rel_path = Path("TrL") / "masks" / mask_out_name

            elif ssl_split == "TrU":
                image_rel_path = Path("TrU") / "images" / image_out_name
                mask_rel_path = ""

            elif ssl_split == "Ts":
                image_rel_path = Path("Ts") / "images" / image_out_name
                mask_rel_path = ""

            else:
                raise ValueError(f"Unknown ssl_split: {ssl_split}")

            image_zip_path = Path(dataset_id) / image_rel_path
            mask_zip_path = Path(dataset_id) / mask_rel_path if mask_rel_path != "" else ""

            zipf.write(original_image_path, arcname=str(image_zip_path))

            if data_type == "video":
                image_shape = get_png_shape(original_image_path)
            else:
                image_shape = get_nifti_shape(original_image_path)

            mask_shape = ""

            if ssl_split == "TrL":
                if original_mask_path is None or not original_mask_path.exists():
                    print(f"[WARN] TrL image has no mask: {original_image_path}")

                else:
                    if data_type == "video":
                        mask_bytes = extract_nii_bytes_from_tar(original_mask_path)

                        if mask_bytes is not None:
                            zipf.writestr(str(mask_zip_path), mask_bytes)
                            mask_shape = get_nii_shape_from_bytes(mask_bytes)
                        else:
                            print(f"[WARN] Could not extract video mask: {original_mask_path}")

                    else:
                        zipf.write(original_mask_path, arcname=str(mask_zip_path))
                        mask_shape = get_nifti_shape(original_mask_path)

            counts[ssl_split] += 1

            metadata_rows.append({
                "dataid": dataid,
                "data_type": data_type,
                "task_id": row["task_id"],
                "case_id": row["case_id"],
                "split": ssl_split,
                "image_shape": image_shape,
                "mask_shape": mask_shape,
                "image_path": str(image_rel_path),
                "mask_path": str(mask_rel_path),
            })

        metadata_df = save_metadata_csv(metadata_rows)
        metadata_csv = metadata_df.to_csv(index=False)

        dataset_json = build_dataset_json(
            dataset_name=dataset_id,
            data_type=data_type,
            counts=counts,
        )

        zipf.writestr(
            str(Path(dataset_id) / "metadata.csv"),
            metadata_csv,
        )

        zipf.writestr(
            str(Path(dataset_id) / "dataset.json"),
            json.dumps(dataset_json, indent=4, ensure_ascii=False),
        )

    print(f"\n[OK] Created zip → {zip_path}")
    print(f"  TrL: {counts['TrL']}")
    print(f"  TrU: {counts['TrU']}")
    print(f"  Ts : {counts['Ts']}")
    print(f"  metadata columns: {METADATA_COLS}")

    return zip_path


# ─────────────────────────────────────────────────────────────────────────────
# Process one type
# ─────────────────────────────────────────────────────────────────────────────

def process_data_type(data_root, output_dir, data_type, test=False, make_zip=False):
    print("\n" + "=" * 80)
    print(f"Processing {data_type}")
    print("=" * 80)

    try:
        file_df = collect_files_for_data_type(
            data_root=data_root,
            data_type=data_type,
        )
    except FileNotFoundError as e:
        print(f"[WARN] {e}")
        return

    print("\nCollected counts:")
    print(file_df.groupby(["raw_split", "file_type"]).size())

    sample_df = build_sample_dataframe(file_df, data_type=data_type)

    print("\nSample counts:")
    print(sample_df.groupby(["ssl_split", "has_mask"]).size())

    if make_zip:
        write_samples_to_zip(
            sample_df=sample_df,
            output_dir=output_dir,
            data_type=data_type,
            test=test,
        )
    else:
        write_samples_to_folder(
            sample_df=sample_df,
            output_dir=output_dir,
            data_type=data_type,
            test=test,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for data_type in ["ct", "tee", "video"]:
        process_data_type(
            data_root=args.data_root,
            output_dir=args.output_dir,
            data_type=data_type,
            test=args.test,
            make_zip=args.zip,
        )


if __name__ == "__main__":
    main()