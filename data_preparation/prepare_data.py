from pathlib import Path
from abc import ABC, abstractmethod
import argparse
import json
import shutil
import tarfile
import random
import io
import gzip

import numpy as np
import nibabel as nib
import pandas as pd
from PIL import Image
from tqdm import tqdm


TEST_N = 5
VIDEO_UNLABELED_TRAIN_RATIO = 0.70


VIDEO_LABELS = {
    "background": 0,
    "class_1": 1,
    "class_2": 2,
    "class_3": 3,
    "class_4": 4,
    "class_5": 5,
    "class_6": 6,
    "class_7": 7,
    "class_8": 8,
    "class_9": 9,
    "class_10": 10,
    "class_11": 11,
    "class_12": 12,
    "class_13": 13,
    "class_14": 14,
    "class_15": 15,
    "class_16": 16,
    "class_17": 17,
    "class_18": 18,
    "class_19": 19,
    "class_20": 20,
}


METADATA_COLS = [
    "dataid",
    "data_type",
    "task_id",
    "case_name",
    "original_case_name",
    "split",
    "is_labeled",
    "has_foreground",
    "labels_in_mask",
    "image_shape",
    "mask_shape",
    "image_path",
    "mask_path",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare MVAA semi-supervised dataset in nnU-Net format"
    )

    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("dirs/data_storage/raw/MVAA"),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dirs/data_storage/raw/MVAA_nnUNET_SSL"),
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Prepare only {TEST_N} samples per split",
    )

    return parser.parse_args()


class MVAADataPreparerBase(ABC):
    data_type = None
    dataset_id = None
    prefix = None
    patterns = None
    case_suffixes = None
    fixed_labels = None

    def __init__(self, data_root, output_dir, test=False):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.test = test
        self.reference_dir = self.data_root / "reference_data"

    @abstractmethod
    def collect_files(self):
        pass

    @abstractmethod
    def write_image(self, src_path, dst_path):
        pass

    @abstractmethod
    def write_real_mask(self, src_path, dst_path):
        pass

    @abstractmethod
    def write_dummy_mask(self, image_path, mask_path):
        pass

    def case_name(self, path):
        name = Path(path).name

        for suffix in self.case_suffixes:
            name = name.replace(suffix, "")

        return f"{self.prefix}_{name}"

    def task_dirs(self):
        dirs = []

        for pattern in self.patterns:
            dirs.extend(self.reference_dir.glob(pattern))

        return sorted({p for p in dirs if p.is_dir()})

    def copy_file(self, src_path, dst_path):
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

    def nifti_shape(self, path):
        try:
            img = nib.load(str(path))
            return "x".join(str(x) for x in img.shape)
        except Exception:
            return ""

    def prepare_output_dirs(self):
        dataset_dir = self.output_dir / self.dataset_id

        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        (dataset_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "imagesTs").mkdir(parents=True, exist_ok=True)

        return dataset_dir

    def build_sample_dataframe(self, rows):
        file_df = pd.DataFrame(rows)

        image_df = file_df[file_df["file_type"] == "image"].copy()
        mask_df = file_df[file_df["file_type"] == "mask"].copy()

        image_df = image_df.rename(columns={"file_path": "original_image_path"})
        mask_df = mask_df.rename(columns={"file_path": "original_mask_path"})

        image_cols = [
            "data_type",
            "task_id",
            "split",
            "case_name",
            "original_image_path",
        ]

        mask_cols = [
            "data_type",
            "task_id",
            "split",
            "case_name",
            "original_mask_path",
        ]

        sample_df = image_df[image_cols].merge(
            mask_df[mask_cols],
            on=["data_type", "task_id", "split", "case_name"],
            how="left",
        )

        sample_df["original_mask_path"] = sample_df["original_mask_path"].fillna("")
        sample_df["is_labeled"] = sample_df["split"].eq("TrL").astype(int)
        sample_df["original_case_name"] = sample_df["case_name"]

        split_order = {
            "TrL": 0,
            "TrU": 1,
            "Ts": 2,
        }

        sample_df["_split_order"] = sample_df["split"].map(split_order)

        sample_df = sample_df.sort_values(
            ["_split_order", "task_id", "case_name"]
        ).reset_index(drop=True)

        sample_df = sample_df.drop(columns=["_split_order"])

        duplicated = sample_df["case_name"].duplicated(keep=False)

        if duplicated.any():
            print("[WARN] Duplicate case names found:")
            print(sample_df.loc[duplicated, ["case_name", "split", "original_image_path"]])

        return sample_df

    def labels_from_values(self, label_values):
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

    def read_labels_from_mask_path(self, mask_path):
        mask_path = Path(mask_path)

        if str(mask_path).lower().endswith(".nii.gz"):
            mask = np.asanyarray(nib.load(str(mask_path)).dataobj)
            return sorted([int(v) for v in np.unique(mask)])

        if str(mask_path).lower().endswith(".tar"):
            try:
                with tarfile.open(mask_path, "r:*") as tar:
                    members = [
                        m for m in tar.getmembers()
                        if m.isfile() and m.name.lower().endswith(".nii.gz")
                    ]

                    if len(members) == 0:
                        print(f"[WARN] No .nii.gz found inside tar: {mask_path}")
                        return []

                    file_obj = tar.extractfile(members[0])

                    if file_obj is None:
                        print(f"[WARN] Could not read .nii.gz inside tar: {mask_path}")
                        return []

                    nii_gz_bytes = file_obj.read()

                with gzip.GzipFile(fileobj=io.BytesIO(nii_gz_bytes), mode="rb") as gz:
                    nii_bytes = gz.read()

                img = nib.Nifti1Image.from_bytes(nii_bytes)
                mask = np.asanyarray(img.dataobj)

                return sorted([int(v) for v in np.unique(mask)])

            except Exception as e:
                print(f"[WARN] Could not read labels from tar mask {mask_path}: {e}")
                return []

        print(f"[WARN] Unsupported mask format while detecting labels: {mask_path}")
        return []

    def check_mask_foreground(self, mask_path):
        if mask_path is None or str(mask_path).strip() == "":
            return 0, ""

        labels = self.read_labels_from_mask_path(mask_path)

        if len(labels) == 0:
            return 0, ""

        has_foreground = int(any(v > 0 for v in labels))
        labels_in_mask = "|".join(str(v) for v in labels)

        return has_foreground, labels_in_mask

    def detect_labels_from_sample_df(self, sample_df):
        labeled_df = sample_df[
            (sample_df["split"] == "TrL") &
            (sample_df["is_labeled"] == 1) &
            (sample_df["original_mask_path"].notna()) &
            (sample_df["original_mask_path"].astype(str).str.strip().ne(""))
        ].copy()

        if labeled_df.empty:
            print(f"[WARN] No labeled masks found for {self.data_type}. Falling back to binary labels.")
            return {
                "background": 0,
                "foreground": 1,
            }

        all_seen_labels = set()
        checked_masks = 0
        masks_with_foreground = 0
        background_only_masks = []

        print(f"\n[INFO] Scanning all real TrL masks for {self.data_type} labels...")

        for _, row in tqdm(
            labeled_df.iterrows(),
            total=len(labeled_df),
            desc=f"Detecting labels {self.data_type}",
        ):
            mask_path = Path(row["original_mask_path"])

            if not mask_path.exists():
                print(f"[WARN] Mask not found: {mask_path}")
                continue

            unique_values = self.read_labels_from_mask_path(mask_path)

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
            print(f"[WARN] No labels detected for {self.data_type}. Falling back to binary labels.")
            all_seen_labels = [0, 1]

        if 0 not in all_seen_labels:
            print(f"[WARN] Label 0 not found for {self.data_type}. Adding background=0.")
            all_seen_labels = [0] + all_seen_labels

        labels = self.labels_from_values(all_seen_labels)

        print(f"\n[OK] Complete label scan for {self.data_type}")
        print(f"     masks checked          : {checked_masks}")
        print(f"     masks with foreground  : {masks_with_foreground}")
        print(f"     background-only masks  : {len(background_only_masks)}")
        print(f"     raw labels found       : {all_seen_labels}")
        print("     detected labels:")

        for name, value in labels.items():
            print(f"       {name}: {value}")

        if background_only_masks:
            print("     background-only examples:")
            for p in background_only_masks[:10]:
                print(f"       {p}")

        return labels

    def get_labels_for_dataset_json(self, sample_df):
        detected_labels = self.detect_labels_from_sample_df(sample_df)

        if self.fixed_labels is None:
            return detected_labels

        detected_values = sorted([int(v) for v in detected_labels.values()])
        fixed_values = sorted([int(v) for v in self.fixed_labels.values()])
        missing_from_detected = sorted(set(fixed_values) - set(detected_values))
        extra_detected = sorted(set(detected_values) - set(fixed_values))

        print(f"\n[OK] Using fixed labels for {self.data_type} in dataset.json.")
        print(f"     detected raw label values : {detected_values}")
        print(f"     fixed label values        : {fixed_values}")
        print(f"     missing in current masks  : {missing_from_detected}")
        print(f"     extra detected labels     : {extra_detected}")
        print("     final dataset.json labels:")

        for name, value in self.fixed_labels.items():
            print(f"       {name}: {value}")

        return self.fixed_labels

    def channel_names(self):
        if self.data_type == "video":
            return {
                "0": "red",
                "1": "green",
                "2": "blue",
            }

        return {
            "0": "image",
        }

    def fix_mask_shape_if_needed(self, image_path, mask_path):
        if mask_path is None or not Path(mask_path).exists():
            return ""

        image_nii = nib.load(str(image_path))
        mask_nii = nib.load(str(mask_path))

        image = np.squeeze(np.asanyarray(image_nii.dataobj))
        mask = np.squeeze(np.asanyarray(mask_nii.dataobj))

        if image.shape == mask.shape:
            return "x".join(str(x) for x in mask.shape)

        if image.ndim == 2 and mask.ndim == 2 and image.shape == mask.T.shape:
            print(
                f"[FIX] Transposing mask to match image: "
                f"image={image.shape}, mask={mask.shape}, mask_path={mask_path}"
            )

            fixed_mask = mask.T.astype(mask.dtype)

            fixed_nii = nib.Nifti1Image(
                fixed_mask,
                affine=image_nii.affine,
            )

            nib.save(fixed_nii, str(mask_path))

            return "x".join(str(x) for x in fixed_mask.shape)

        if image.ndim == 3 and mask.ndim == 3 and image.shape == mask.transpose(1, 0, 2).shape:
            print(
                f"[FIX] Transposing 3D mask axes to match image: "
                f"image={image.shape}, mask={mask.shape}, mask_path={mask_path}"
            )

            fixed_mask = mask.transpose(1, 0, 2).astype(mask.dtype)

            fixed_nii = nib.Nifti1Image(
                fixed_mask,
                affine=image_nii.affine,
            )

            nib.save(fixed_nii, str(mask_path))

            return "x".join(str(x) for x in fixed_mask.shape)

        raise ValueError(
            f"Image and mask shape mismatch cannot be fixed automatically: "
            f"image={image.shape}, mask={mask.shape}, mask_path={mask_path}"
        )

    def get_image_paths(self, dataset_dir, split, case_name):
        if split in ["TrL", "TrU"]:
            image_dir_rel = Path("imagesTr")
        elif split == "Ts":
            image_dir_rel = Path("imagesTs")
        else:
            raise ValueError(f"Unknown split: {split}")

        if self.data_type == "video":
            image_base_rel = image_dir_rel / case_name
            image_rel_for_metadata = image_dir_rel / f"{case_name}_0000.nii.gz"
            image_dst_for_writer = dataset_dir / image_base_rel
        else:
            image_rel_for_metadata = image_dir_rel / f"{case_name}_0000.nii.gz"
            image_dst_for_writer = dataset_dir / image_rel_for_metadata

        return image_rel_for_metadata, image_dst_for_writer

    def write_dataset(self, sample_df):
        dataset_dir = self.prepare_output_dirs()

        if self.test:
            sample_df = (
                sample_df
                .groupby("split", group_keys=False)
                .head(TEST_N)
                .reset_index(drop=True)
            )

        labels = self.get_labels_for_dataset_json(sample_df)

        metadata_rows = []

        counts = {
            "TrL": 0,
            "TrU": 0,
            "Ts": 0,
        }

        for dataid, row in tqdm(
            sample_df.iterrows(),
            total=len(sample_df),
            desc=f"Writing {self.dataset_id}",
        ):
            split = row["split"]
            case_name = row["case_name"]

            src_image = Path(row["original_image_path"])

            src_mask = (
                Path(row["original_mask_path"])
                if row["original_mask_path"] != ""
                else None
            )

            image_rel, dst_image_for_writer = self.get_image_paths(
                dataset_dir=dataset_dir,
                split=split,
                case_name=case_name,
            )

            mask_rel = None
            dst_mask = None

            if split in ["TrL", "TrU"]:
                mask_rel = Path("labelsTr") / f"{case_name}.nii.gz"
                dst_mask = dataset_dir / mask_rel

            primary_image_path = self.write_image(
                src_path=src_image,
                dst_path=dst_image_for_writer,
            )

            image_shape = self.nifti_shape(primary_image_path)
            mask_shape = ""
            has_foreground = 0
            labels_in_mask = ""

            if split == "TrL":
                if src_mask is None or not src_mask.exists():
                    print(f"[WARN] Missing mask for labeled case: {src_image}")
                else:
                    ok = self.write_real_mask(src_mask, dst_mask)

                    if ok:
                        mask_shape = self.fix_mask_shape_if_needed(
                            image_path=primary_image_path,
                            mask_path=dst_mask,
                        )

                        has_foreground, labels_in_mask = self.check_mask_foreground(dst_mask)

                        if has_foreground == 0:
                            print(f"[FLAG] Background-only TrL mask: {dst_mask}")

            elif split == "TrU":
                self.write_dummy_mask(primary_image_path, dst_mask)

                mask_shape = self.fix_mask_shape_if_needed(
                    image_path=primary_image_path,
                    mask_path=dst_mask,
                )

                has_foreground = 0
                labels_in_mask = "0"

            elif split == "Ts":
                pass

            else:
                raise ValueError(f"Unknown split: {split}")

            counts[split] += 1

            metadata_rows.append({
                "dataid": int(dataid),
                "data_type": self.data_type,
                "task_id": row["task_id"],
                "case_name": case_name,
                "original_case_name": row["original_case_name"],
                "split": split,
                "is_labeled": int(row["is_labeled"]),
                "has_foreground": has_foreground,
                "labels_in_mask": labels_in_mask,
                "image_shape": image_shape,
                "mask_shape": mask_shape,
                "image_path": str(image_rel),
                "mask_path": "" if mask_rel is None else str(mask_rel),
            })

        metadata_df = pd.DataFrame(metadata_rows)

        if metadata_df.empty:
            metadata_df = pd.DataFrame(columns=METADATA_COLS)
        else:
            metadata_df = metadata_df[METADATA_COLS]

        metadata_df.to_csv(dataset_dir / "metadata.csv", index=False)

        dataset_json = {
            "channel_names": self.channel_names(),
            "labels": labels,
            "numTraining": int(counts["TrL"] + counts["TrU"]),
            "file_ending": ".nii.gz",
            "name": self.dataset_id,
            "description": (
                "MVAA semi-supervised nnU-Net dataset. "
                "TrL has real masks. TrU has dummy zero masks. "
                "For video, RGB frames are stored as three nnU-Net channels."
            ),
            "ssl_counts": {
                "TrL": int(counts["TrL"]),
                "TrU": int(counts["TrU"]),
                "Ts": int(counts["Ts"]),
            },
        }

        with open(dataset_dir / "dataset.json", "w", encoding="utf-8") as f:
            json.dump(dataset_json, f, indent=4, ensure_ascii=False)

        print(f"\n[OK] Prepared dataset: {dataset_dir}")
        print(f"  TrL labeled training   : {counts['TrL']}")
        print(f"  TrU unlabeled training : {counts['TrU']}")
        print(f"  Ts test                : {counts['Ts']}")
        print(f"  imagesTr              : {len(list((dataset_dir / 'imagesTr').glob('*')))}")
        print(f"  labelsTr              : {len(list((dataset_dir / 'labelsTr').glob('*')))}")
        print(f"  imagesTs              : {len(list((dataset_dir / 'imagesTs').glob('*')))}")
        print(f"  metadata              : {dataset_dir / 'metadata.csv'}")
        print(f"  dataset.json          : {dataset_dir / 'dataset.json'}")

    def process(self):
        print("\n" + "=" * 80)
        print(f"Processing {self.data_type.upper()}")
        print("=" * 80)

        rows = self.collect_files()

        if len(rows) == 0:
            print(f"[WARN] No files found for {self.data_type}")
            return

        file_df = pd.DataFrame(rows)

        print("\nCollected files:")
        print(file_df.groupby(["split", "file_type"]).size())

        sample_df = self.build_sample_dataframe(rows)

        print("\nPrepared samples:")
        print(sample_df.groupby(["split", "is_labeled"]).size())

        self.write_dataset(sample_df)


class CTDataPreparer(MVAADataPreparerBase):
    data_type = "ct"
    dataset_id = "Dataset001_MVAA_CT_SSL"
    prefix = "ct"
    patterns = ["t1_ct", "*ct*"]
    case_suffixes = ["-seg.nii.gz", ".nii.gz"]

    def build_sample_dataframe(self, rows):
        sample_df = super().build_sample_dataframe(rows)

        split_order = {
            "TrL": 0,
            "TrU": 1,
            "Ts": 2,
        }

        sample_df["_split_order"] = sample_df["split"].map(split_order)

        sample_df = sample_df.sort_values(
            ["_split_order", "task_id", "original_case_name"]
        ).reset_index(drop=True)

        sample_df["case_name"] = [
            f"{self.prefix}_{i + 1:04d}"
            for i in range(len(sample_df))
        ]

        sample_df = sample_df.drop(columns=["_split_order"])

        print("[OK] CT final case names reassigned sequentially.")
        print("     TrL first, then TrU, then Ts.")

        return sample_df

    def collect_files(self):
        rows = []
        task_dirs = self.task_dirs()

        if len(task_dirs) == 0:
            print(f"[WARN] No CT folders found under {self.reference_dir}")
            return rows

        for task_dir in task_dirs:
            for p in sorted((task_dir / "train" / "labeled" / "images").glob("*.nii.gz")):
                rows.append({
                    "data_type": self.data_type,
                    "task_id": task_dir.name,
                    "split": "TrL",
                    "file_type": "image",
                    "case_name": self.case_name(p),
                    "file_path": str(p),
                })

            for p in sorted((task_dir / "train" / "labeled" / "labels").glob("*-seg.nii.gz")):
                rows.append({
                    "data_type": self.data_type,
                    "task_id": task_dir.name,
                    "split": "TrL",
                    "file_type": "mask",
                    "case_name": self.case_name(p),
                    "file_path": str(p),
                })

            for p in sorted((task_dir / "train" / "unlabeled").glob("*.nii.gz")):
                rows.append({
                    "data_type": self.data_type,
                    "task_id": task_dir.name,
                    "split": "TrU",
                    "file_type": "image",
                    "case_name": self.case_name(p),
                    "file_path": str(p),
                })

            for p in sorted((task_dir / "val" / "images").glob("*.nii.gz")):
                rows.append({
                    "data_type": self.data_type,
                    "task_id": task_dir.name,
                    "split": "Ts",
                    "file_type": "image",
                    "case_name": self.case_name(p),
                    "file_path": str(p),
                })

        return rows

    def write_image(self, src_path, dst_path):
        self.copy_file(src_path, dst_path)
        return dst_path

    def write_real_mask(self, src_path, dst_path):
        self.copy_file(src_path, dst_path)
        return True

    def write_dummy_mask(self, image_path, mask_path):
        img = nib.load(str(image_path))
        dummy = np.zeros(img.shape, dtype=np.uint8)

        dummy_nii = nib.Nifti1Image(
            dummy,
            affine=img.affine,
            header=img.header,
        )

        mask_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(dummy_nii, str(mask_path))


class TEEDataPreparer(MVAADataPreparerBase):
    data_type = "tee"
    dataset_id = "Dataset002_MVAA_TEE_SSL"
    prefix = "tee"
    patterns = ["t2_tee", "*tee*"]
    case_suffixes = ["-US.nii.gz", "-label.nii.gz"]

    def collect_files(self):
        rows = []
        task_dirs = self.task_dirs()

        if len(task_dirs) == 0:
            print(f"[WARN] No TEE folders found under {self.reference_dir}")
            return rows

        for task_dir in task_dirs:
            train_dir = task_dir / "train"

            for p in sorted(train_dir.glob("*-US.nii.gz")):
                rows.append({
                    "data_type": self.data_type,
                    "task_id": task_dir.name,
                    "split": "TrL",
                    "file_type": "image",
                    "case_name": self.case_name(p),
                    "file_path": str(p),
                })

            for p in sorted(train_dir.glob("*-label.nii.gz")):
                rows.append({
                    "data_type": self.data_type,
                    "task_id": task_dir.name,
                    "split": "TrL",
                    "file_type": "mask",
                    "case_name": self.case_name(p),
                    "file_path": str(p),
                })

            for p in sorted((task_dir / "val" / "images").glob("*-US.nii.gz")):
                rows.append({
                    "data_type": self.data_type,
                    "task_id": task_dir.name,
                    "split": "Ts",
                    "file_type": "image",
                    "case_name": self.case_name(p),
                    "file_path": str(p),
                })

        return rows

    def write_image(self, src_path, dst_path):
        self.copy_file(src_path, dst_path)
        return dst_path

    def write_real_mask(self, src_path, dst_path):
        self.copy_file(src_path, dst_path)
        return True

    def write_dummy_mask(self, image_path, mask_path):
        img = nib.load(str(image_path))
        dummy = np.zeros(img.shape, dtype=np.uint8)

        dummy_nii = nib.Nifti1Image(
            dummy,
            affine=img.affine,
            header=img.header,
        )

        mask_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(dummy_nii, str(mask_path))


class VideoDataPreparer(MVAADataPreparerBase):
    data_type = "video"
    dataset_id = "Dataset003_MVAA_VIDEO_SSL"
    prefix = "video"
    patterns = ["t3_vid", "*vid*", "*video*"]
    case_suffixes = [".png", ".jpg", ".jpeg", "_png_Label.tar"]
    fixed_labels = VIDEO_LABELS

    def collect_files(self):
        rows = []
        task_dirs = self.task_dirs()

        if len(task_dirs) == 0:
            print(f"[WARN] No reference video folders found under {self.reference_dir}")

        for task_dir in task_dirs:
            for p in sorted((task_dir / "train").glob("*/*.png")):
                rows.append({
                    "data_type": self.data_type,
                    "task_id": task_dir.name,
                    "split": "TrL",
                    "file_type": "image",
                    "case_name": self.case_name(p),
                    "file_path": str(p),
                })

            for p in sorted((task_dir / "train").glob("*/*_png_Label.tar")):
                rows.append({
                    "data_type": self.data_type,
                    "task_id": task_dir.name,
                    "split": "TrL",
                    "file_type": "mask",
                    "case_name": self.case_name(p),
                    "file_path": str(p),
                })

            for p in sorted((task_dir / "val" / "images").glob("*/*.png")):
                rows.append({
                    "data_type": self.data_type,
                    "task_id": task_dir.name,
                    "split": "Ts",
                    "file_type": "image",
                    "case_name": self.case_name(p),
                    "file_path": str(p),
                })

        rows.extend(self.collect_extra_unlabeled_video())

        return rows

    def collect_extra_unlabeled_video(self):
        rows = []
        image_dir = self.data_root / "images"

        if not image_dir.exists():
            print(f"[WARN] Extra video image folder not found: {image_dir}")
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
                "data_type": self.data_type,
                "task_id": "t3_vid",
                "split": "TrU",
                "file_type": "image",
                "case_name": self.case_name(p),
                "file_path": str(p),
            })

        for p in test_paths:
            rows.append({
                "data_type": self.data_type,
                "task_id": "t3_vid",
                "split": "Ts",
                "file_type": "image",
                "case_name": self.case_name(p),
                "file_path": str(p),
            })

        print(f"[OK] Extra video images: {len(image_paths)}")
        print(f"     TrU: {len(train_paths)}")
        print(f"     Ts : {len(test_paths)}")

        return rows

    def write_image(self, src_path, dst_path):
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

    def write_real_mask(self, src_path, dst_path):
        try:
            with tarfile.open(src_path, "r:*") as tar:
                members = [
                    m for m in tar.getmembers()
                    if m.isfile() and m.name.lower().endswith(".nii.gz")
                ]

                if len(members) == 0:
                    print(f"[WARN] No .nii.gz mask found in {src_path}")
                    return False

                file_obj = tar.extractfile(members[0])

                if file_obj is None:
                    print(f"[WARN] Could not read mask from {src_path}")
                    return False

                dst_path.parent.mkdir(parents=True, exist_ok=True)

                with open(dst_path, "wb") as f:
                    shutil.copyfileobj(file_obj, f)

                return True

        except Exception as e:
            print(f"[WARN] Could not extract mask from {src_path}: {e}")
            return False

    def write_dummy_mask(self, image_path, mask_path):
        img = nib.load(str(image_path))
        dummy = np.zeros(img.shape, dtype=np.uint8)

        dummy_nii = nib.Nifti1Image(
            dummy,
            affine=img.affine,
        )

        mask_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(dummy_nii, str(mask_path))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    preparers = [
        CTDataPreparer(args.data_root, args.output_dir, args.test),
        TEEDataPreparer(args.data_root, args.output_dir, args.test),
        VideoDataPreparer(args.data_root, args.output_dir, args.test),
    ]

    for preparer in preparers:
        preparer.process()


if __name__ == "__main__":
    main()