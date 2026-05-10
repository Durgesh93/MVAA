#!/usr/bin/env python3
"""Dataset and dataloader utilities for task1 semi-supervised training."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Dict, List, Sequence

import numpy as np
import torch
from monai.data import DataLoader, Dataset, list_data_collate
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandSpatialCropSamplesd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)

_AFFINE_EPS = 1e-8


def _to_numpy_affine(affine) -> np.ndarray | None:
    if affine is None:
        return None
    if isinstance(affine, torch.Tensor):
        arr = affine.detach().cpu().numpy()
    else:
        arr = np.asarray(affine)
    if arr.shape != (4, 4):
        return None
    return arr.astype(np.float64, copy=False)


def _is_valid_affine(affine) -> bool:
    arr = _to_numpy_affine(affine)
    if arr is None:
        return False
    if not np.isfinite(arr).all():
        return False
    det = float(np.linalg.det(arr[:3, :3]))
    return abs(det) > _AFFINE_EPS


def _identity_like(affine):
    if isinstance(affine, torch.Tensor):
        return torch.eye(4, dtype=affine.dtype, device=affine.device)
    return np.eye(4, dtype=np.float64)


class FixInvalidAffineD:
    """Fix invalid/non-invertible affine matrices before spacing resample."""

    def __init__(self, keys: Sequence[str]):
        self.keys = tuple(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            meta_key = f"{key}_meta_dict"
            meta = d.get(meta_key)
            if isinstance(meta, dict):
                for aff_name in ("affine", "original_affine"):
                    if aff_name in meta and not _is_valid_affine(meta[aff_name]):
                        meta[aff_name] = np.eye(4, dtype=np.float64)

            img = d.get(key)
            if hasattr(img, "affine"):
                aff = getattr(img, "affine")
                if not _is_valid_affine(aff):
                    setattr(img, "affine", _identity_like(aff))
        return d


def _collect_nii_sorted(folder: str | Path) -> List[Path]:
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {p}")
    files = sorted(p.glob("*.nii.gz"))
    return [f for f in files if f.is_file()]


def discover_labeled_pairs(images_dir: str | Path, labels_dir: str | Path) -> List[Dict[str, str]]:
    images = _collect_nii_sorted(images_dir)
    labels_root = Path(labels_dir)
    pairs: List[Dict[str, str]] = []

    for image_path in images:
        stem = image_path.name.replace(".nii.gz", "")
        label_path = labels_root / f"{stem}-seg.nii.gz"
        if label_path.exists():
            pairs.append(
                {
                    "case_id": stem,
                    "image": str(image_path),
                    "label": str(label_path),
                }
            )
    return pairs


def discover_unlabeled_files(images_dir: str | Path) -> List[Dict[str, str]]:
    images = _collect_nii_sorted(images_dir)
    return [
        {
            "case_id": p.name.replace(".nii.gz", ""),
            "image": str(p),
        }
        for p in images
    ]


def get_labeled_train_transforms(
    roi_size: Sequence[int],
    num_samples: int,
    enable_spacing_resample: bool = False,
    target_spacing: Sequence[float] = (0.5, 0.5, 0.5),
):
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
    ]
    if enable_spacing_resample:
        transforms.append(FixInvalidAffineD(keys=["image", "label"]))
        transforms.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(target_spacing),
                mode=("bilinear", "nearest"),
                diagonal=True,
            )
        )
    transforms.extend(
        [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000.0,
                a_max=1000.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=tuple(roi_size)),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=tuple(roi_size),
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
                allow_smaller=True,
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=tuple(roi_size)),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return Compose(transforms)


def get_unlabeled_train_transforms(
    roi_size: Sequence[int],
    num_samples: int,
    enable_spacing_resample: bool = False,
    target_spacing: Sequence[float] = (0.5, 0.5, 0.5),
):
    transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
    ]
    if enable_spacing_resample:
        transforms.append(FixInvalidAffineD(keys=["image"]))
        transforms.append(
            Spacingd(
                keys=["image"],
                pixdim=tuple(target_spacing),
                mode=("bilinear",),
                diagonal=True,
            )
        )
    transforms.extend(
        [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000.0,
                a_max=1000.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            SpatialPadd(keys=["image"], spatial_size=tuple(roi_size)),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=tuple(roi_size),
                num_samples=num_samples,
                random_center=True,
                random_size=False,
            ),
            ResizeWithPadOrCropd(keys=["image"], spatial_size=tuple(roi_size)),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image"], prob=0.2, max_k=3),
            EnsureTyped(keys=["image"]),
        ]
    )
    return Compose(transforms)


def get_eval_transforms(
    enable_spacing_resample: bool = False,
    target_spacing: Sequence[float] = (0.5, 0.5, 0.5),
):
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
    ]
    if enable_spacing_resample:
        transforms.append(FixInvalidAffineD(keys=["image", "label"]))
        transforms.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(target_spacing),
                mode=("bilinear", "nearest"),
                diagonal=True,
            )
        )
    transforms.extend(
        [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000.0,
                a_max=1000.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return Compose(transforms)


def get_task1_dataloaders(
    root_dir: str | Path,
    roi_size: Sequence[int],
    batch_size: int,
    num_samples: int,
    num_workers: int,
    enable_spacing_resample: bool = False,
    target_spacing: Sequence[float] = (0.5, 0.5, 0.5),
    max_labeled_cases: int = 0,
    max_unlabeled_cases: int = 0,
    max_val_cases: int = 0,
    val_ratio: float = 0.1,
    val_count: int = 0,
    split_seed: int = 42,
):
    root = Path(root_dir)

    train_labeled_all = discover_labeled_pairs(
        root / "train" / "labeled" / "images",
        root / "train" / "labeled" / "labels",
    )
    train_unlabeled = discover_unlabeled_files(root / "train" / "unlabeled")

    if max_labeled_cases > 0:
        train_labeled_all = train_labeled_all[:max_labeled_cases]
    if max_unlabeled_cases > 0:
        train_unlabeled = train_unlabeled[:max_unlabeled_cases]

    if len(train_labeled_all) == 0:
        raise RuntimeError("No labeled train pairs found.")
    if len(train_unlabeled) == 0:
        raise RuntimeError("No unlabeled train images found.")

    if val_count > 0:
        requested_val = val_count
    else:
        requested_val = int(round(len(train_labeled_all) * max(0.0, min(0.5, val_ratio))))

    if len(train_labeled_all) >= 2:
        requested_val = max(1, min(requested_val, len(train_labeled_all) - 1))
    else:
        requested_val = 0

    shuffled_labeled = list(train_labeled_all)
    random.Random(split_seed).shuffle(shuffled_labeled)
    val_files = shuffled_labeled[:requested_val]
    train_labeled = shuffled_labeled[requested_val:]

    if max_val_cases > 0:
        val_files = val_files[:max_val_cases]

    labeled_ds = Dataset(
        train_labeled,
        transform=get_labeled_train_transforms(
            roi_size=roi_size,
            num_samples=num_samples,
            enable_spacing_resample=enable_spacing_resample,
            target_spacing=target_spacing,
        ),
    )
    unlabeled_ds = Dataset(
        train_unlabeled,
        transform=get_unlabeled_train_transforms(
            roi_size=roi_size,
            num_samples=num_samples,
            enable_spacing_resample=enable_spacing_resample,
            target_spacing=target_spacing,
        ),
    )
    val_ds = Dataset(
        val_files,
        transform=get_eval_transforms(
            enable_spacing_resample=enable_spacing_resample,
            target_spacing=target_spacing,
        ),
    )
    labeled_loader = DataLoader(
        labeled_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    unlabeled_loader = DataLoader(
        unlabeled_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, min(2, num_workers)),
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    split_info = {
        "train_labeled_count": len(train_labeled),
        "train_unlabeled_count": len(train_unlabeled),
        "val_count": len(val_files),
        "train_labeled_cases": [x["case_id"] for x in train_labeled],
        "train_unlabeled_cases": [x["case_id"] for x in train_unlabeled],
        "val_cases": [x["case_id"] for x in val_files],
    }

    return labeled_loader, unlabeled_loader, val_loader, split_info
