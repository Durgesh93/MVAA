#!/usr/bin/env python3
"""Dataset and dataloader utilities for NIfTI cardiac segmentation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from monai.data import DataLoader, Dataset, list_data_collate
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandAdjustContrastd,
    RandAffined,
    RandCropByLabelClassesd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandShiftIntensityd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
)

CASE_RE = re.compile(r"^(train_\d+)-US\.nii(\.gz)?$")


def _case_sort_key(case_id: str) -> int:
    num = case_id.split("_")[-1]
    return int(num)


def discover_case_pairs(data_dir: str | Path) -> List[Dict[str, str]]:
    """Find paired files in the format train_xxx-US.nii.gz and train_xxx-label.nii.gz."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data_dir not found: {root}")

    us_files = sorted(root.glob("*-US.nii.gz"))
    pairs: List[Tuple[str, Path, Path]] = []

    for us_path in us_files:
        m = CASE_RE.match(us_path.name)
        if not m:
            continue
        case_id = m.group(1)
        label_path = root / f"{case_id}-label.nii.gz"
        if label_path.exists():
            pairs.append((case_id, us_path, label_path))

    pairs = sorted(pairs, key=lambda x: _case_sort_key(x[0]))
    return [
        {"case_id": case_id, "image": str(image), "label": str(label)}
        for case_id, image, label in pairs
    ]


def split_train_val(
    files: Sequence[Dict[str, str]],
    val_count: int = 20,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Use the last val_count cases as validation set."""
    if len(files) <= val_count:
        raise ValueError(
            f"Not enough files for split: total={len(files)}, val_count={val_count}"
        )

    train_files = list(files[:-val_count])
    val_files = list(files[-val_count:])
    return train_files, val_files


def get_train_transforms(
    roi_size: Sequence[int],
    num_samples: int,
    num_classes: int,
    enable_spacing_resample: bool = False,
    target_spacing: Sequence[float] = (0.5, 0.5, 0.5),
):
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
    ]
    if enable_spacing_resample:
        transforms.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(target_spacing),
                mode=("bilinear", "nearest"),
            )
        )
    transforms.extend(
        [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0.0,
                a_max=255.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=tuple(roi_size),
                ratios=tuple([1.0] + [2.0] * max(0, num_classes - 1)),
                num_classes=num_classes,
                num_samples=num_samples,
                allow_smaller=True,
            ),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=tuple(roi_size)),
            RandAffined(
                keys=["image", "label"],
                prob=0.2,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.03),
            RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.7, 1.5)),
            RandShiftIntensityd(keys=["image"], prob=0.2, offsets=0.1),
            RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.25, 1.0), sigma_y=(0.25, 1.0), sigma_z=(0.25, 1.0)),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return Compose(transforms)


def get_val_transforms(
    enable_spacing_resample: bool = False,
    target_spacing: Sequence[float] = (0.5, 0.5, 0.5),
):
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
    ]
    if enable_spacing_resample:
        transforms.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(target_spacing),
                mode=("bilinear", "nearest"),
            )
        )
    transforms.extend(
        [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0.0,
                a_max=255.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return Compose(transforms)


def get_dataloaders(
    data_dir: str | Path,
    val_count: int,
    roi_size: Sequence[int],
    batch_size: int,
    num_samples: int,
    num_workers: int,
    max_train_cases: int = 0,
    max_val_cases: int = 0,
    num_classes: int = 3,
    enable_spacing_resample: bool = False,
    target_spacing: Sequence[float] = (0.5, 0.5, 0.5),
):
    all_files = discover_case_pairs(data_dir)
    train_files, val_files = split_train_val(all_files, val_count=val_count)
    if max_train_cases > 0:
        train_files = train_files[:max_train_cases]
    if max_val_cases > 0:
        val_files = val_files[:max_val_cases]

    train_ds = Dataset(
        train_files,
        transform=get_train_transforms(
            roi_size=roi_size,
            num_samples=num_samples,
            num_classes=num_classes,
            enable_spacing_resample=enable_spacing_resample,
            target_spacing=target_spacing,
        ),
    )
    val_ds = Dataset(
        val_files,
        transform=get_val_transforms(
            enable_spacing_resample=enable_spacing_resample,
            target_spacing=target_spacing,
        ),
    )

    train_loader = DataLoader(
        train_ds,
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

    return train_loader, val_loader, train_files, val_files
