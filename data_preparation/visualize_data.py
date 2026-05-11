from pathlib import Path
import argparse

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize MVAA nnU-Net samples from metadata.csv"
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dirs/data_storage/raw/MVAA_nnUNET_SSL/Dataset001_MVAA_CT_SSL"),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="TrL",
        choices=["TrL", "TrU", "Ts"],
    )

    parser.add_argument(
        "--n_random",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/mvaa_visualization"),
    )

    parser.add_argument(
        "--axis",
        type=int,
        default=2,
        choices=[0, 1, 2],
    )

    parser.add_argument(
        "--max_slices",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--only_mask_slices",
        action="store_true",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
    )

    return parser.parse_args()


# =============================================================================
# Loaders
# =============================================================================

def load_nifti(path):
    nii = nib.load(str(path))
    return np.asanyarray(nii.dataobj)


def load_rgb_video_image(dataset_dir, image_path):
    """
    Load video RGB image saved in nnU-Net channel format.

    metadata image_path points to:
        case_0000.nii.gz

    This function also loads:
        case_0001.nii.gz
        case_0002.nii.gz

    Returns:
        H x W x 3 RGB image
    """
    image_path = dataset_dir / image_path

    if not image_path.exists():
        raise FileNotFoundError(f"Image channel 0 not found: {image_path}")

    image_path_str = str(image_path)

    if not image_path_str.endswith("_0000.nii.gz"):
        raise ValueError(f"Expected video image path ending with _0000.nii.gz: {image_path}")

    r_path = image_path
    g_path = Path(image_path_str.replace("_0000.nii.gz", "_0001.nii.gz"))
    b_path = Path(image_path_str.replace("_0000.nii.gz", "_0002.nii.gz"))

    for p in [r_path, g_path, b_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing RGB channel: {p}")

    r = np.squeeze(load_nifti(r_path)).astype(np.float32)
    g = np.squeeze(load_nifti(g_path)).astype(np.float32)
    b = np.squeeze(load_nifti(b_path)).astype(np.float32)

    if r.shape != g.shape or r.shape != b.shape:
        raise ValueError(
            f"RGB channel shape mismatch: R={r.shape}, G={g.shape}, B={b.shape}"
        )

    rgb = np.stack([r, g, b], axis=-1)
    return rgb


def load_image_from_metadata(row, dataset_dir):
    """
    CT/TEE:
        load one NIfTI image channel as 2D/3D grayscale.

    Video:
        load three NIfTI channels as RGB.
    """
    if row["data_type"] == "video":
        return load_rgb_video_image(dataset_dir, row["image_path"])

    image_path = dataset_dir / row["image_path"]

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return np.squeeze(load_nifti(image_path))


# =============================================================================
# Image helpers
# =============================================================================

def normalize_image(arr):
    arr = np.asarray(arr).astype(np.float32)

    if arr.size == 0:
        return arr

    if np.all(arr == arr.flat[0]):
        return np.zeros_like(arr, dtype=np.float32)

    vmin, vmax = np.percentile(arr, [1, 99])
    arr = np.clip(arr, vmin, vmax)
    arr = (arr - vmin) / (vmax - vmin + 1e-8)

    return arr


def normalize_rgb(arr):
    arr = np.asarray(arr).astype(np.float32)

    out = np.zeros_like(arr, dtype=np.float32)

    for c in range(3):
        out[..., c] = normalize_image(arr[..., c])

    return out


def squeeze_mask(mask):
    mask = np.squeeze(np.asarray(mask))

    if mask.ndim not in [2, 3]:
        raise ValueError(f"Expected 2D or 3D mask. Got shape={mask.shape}")

    return mask


def is_rgb_image(image):
    return image.ndim == 3 and image.shape[-1] == 3


def is_3d_gray_image(image):
    return image.ndim == 3 and image.shape[-1] != 3


def get_slice(arr, axis, idx):
    if axis == 0:
        return arr[idx, :, :]
    if axis == 1:
        return arr[:, idx, :]
    if axis == 2:
        return arr[:, :, idx]

    raise ValueError(f"Invalid axis: {axis}")


def select_slice_indices(mask, axis, max_slices, only_mask_slices):
    if mask.ndim == 2:
        return [None]

    n_slices = mask.shape[axis]

    if only_mask_slices:
        slice_ids = []

        for i in range(n_slices):
            mask_slice = get_slice(mask, axis, i)

            if np.any(mask_slice > 0):
                slice_ids.append(i)

        if len(slice_ids) == 0:
            print("[WARN] No foreground mask slices found. Falling back to all slices.")
            slice_ids = list(range(n_slices))
    else:
        slice_ids = list(range(n_slices))

    if len(slice_ids) > max_slices:
        keep = np.linspace(0, len(slice_ids) - 1, max_slices).astype(int)
        slice_ids = [slice_ids[i] for i in keep]

    return slice_ids


# =============================================================================
# Color helpers
# =============================================================================

def label_to_color(label):
    colors = {
        1: np.array([1.0, 0.0, 0.0]),
        2: np.array([0.0, 1.0, 0.0]),
        3: np.array([0.0, 0.0, 1.0]),
        4: np.array([1.0, 1.0, 0.0]),
        5: np.array([1.0, 0.0, 1.0]),
        6: np.array([0.0, 1.0, 1.0]),
        7: np.array([1.0, 0.5, 0.0]),
        8: np.array([0.5, 0.0, 1.0]),
        9: np.array([0.0, 0.5, 1.0]),
        10: np.array([0.5, 1.0, 0.0]),
        11: np.array([1.0, 0.4, 0.7]),
        12: np.array([0.6, 0.3, 0.0]),
        13: np.array([0.7, 0.7, 0.7]),
        14: np.array([0.0, 0.8, 0.4]),
        15: np.array([0.4, 0.4, 1.0]),
        16: np.array([0.8, 0.2, 0.2]),
    }

    label = int(label)

    if label in colors:
        return colors[label]

    rng = np.random.default_rng(label)
    return rng.random(3)


def make_colored_mask(mask_2d):
    mask_2d = np.asarray(mask_2d)
    rgb = np.zeros((*mask_2d.shape, 3), dtype=np.float32)

    labels = sorted([int(v) for v in np.unique(mask_2d) if int(v) != 0])

    for label in labels:
        rgb[mask_2d == label] = label_to_color(label)

    return rgb


def make_overlay(image_2d, mask_2d, alpha):
    """
    Works for:
        grayscale image: H x W
        RGB image:       H x W x 3
    """
    if image_2d.ndim == 2:
        image_norm = normalize_image(image_2d)
        rgb = np.stack([image_norm, image_norm, image_norm], axis=-1)
    elif image_2d.ndim == 3 and image_2d.shape[-1] == 3:
        rgb = normalize_rgb(image_2d)
    else:
        raise ValueError(f"Expected 2D grayscale or RGB image. Got shape={image_2d.shape}")

    overlay = rgb.copy()

    labels = sorted([int(v) for v in np.unique(mask_2d) if int(v) != 0])

    for label in labels:
        mask_bool = mask_2d == label
        color = label_to_color(label)

        overlay[mask_bool] = (
            (1 - alpha) * overlay[mask_bool] +
            alpha * color
        )

    return overlay


# =============================================================================
# Visualization
# =============================================================================

def visualize_case(
    row,
    dataset_dir,
    output_dir,
    axis,
    max_slices,
    only_mask_slices,
    alpha,
):
    image = load_image_from_metadata(row, dataset_dir)

    mask_path_value = str(row.get("mask_path", "")).strip()

    if mask_path_value:
        mask_path = dataset_dir / mask_path_value

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        mask = squeeze_mask(load_nifti(mask_path))
    else:
        mask_path = None

        if is_rgb_image(image):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask = np.zeros_like(image, dtype=np.uint8)

    if is_rgb_image(image):
        image_spatial_shape = image.shape[:2]
    else:
        image = np.squeeze(image)
        image_spatial_shape = image.shape

    if image_spatial_shape != mask.shape:
        raise ValueError(
            f"Shape mismatch for {row['case_name']}: "
            f"image={image_spatial_shape}, mask={mask.shape}"
        )

    print("\nSample:")
    print(f"  case_name : {row['case_name']}")
    print(f"  data_type : {row['data_type']}")
    print(f"  split     : {row['split']}")
    print(f"  labeled   : {row['is_labeled']}")
    print(f"  image shp : {image.shape}")
    print(f"  mask      : {mask_path}")
    print(f"  mask shp  : {mask.shape}")
    print(f"  labels    : {np.unique(mask)}")

    slice_ids = select_slice_indices(
        mask=mask,
        axis=axis,
        max_slices=max_slices,
        only_mask_slices=only_mask_slices,
    )

    n = len(slice_ids)

    fig_height = max(4, n * 2.4)
    fig, axes = plt.subplots(n, 3, figsize=(12, fig_height))

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, slice_idx in enumerate(slice_ids):
        if is_rgb_image(image):
            image_2d = image
            mask_2d = mask
            image_show = normalize_rgb(image_2d)
            slice_title = "RGB"
        elif image.ndim == 2:
            image_2d = image
            mask_2d = mask
            image_show = normalize_image(image_2d)
            slice_title = "2D"
        else:
            image_2d = get_slice(image, axis, slice_idx)
            mask_2d = get_slice(mask, axis, slice_idx)
            image_show = normalize_image(image_2d)
            slice_title = f"slice {slice_idx}"

        mask_show = make_colored_mask(mask_2d)
        overlay = make_overlay(image_2d, mask_2d, alpha)

        if image_show.ndim == 2:
            axes[row_idx, 0].imshow(image_show, cmap="gray")
        else:
            axes[row_idx, 0].imshow(image_show)

        axes[row_idx, 0].set_title(f"Image | {slice_title}")

        axes[row_idx, 1].imshow(mask_show, interpolation="nearest")
        axes[row_idx, 1].set_title(f"Mask | {slice_title}")

        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title(f"Overlay | {slice_title}")

        for col in range(3):
            axes[row_idx, col].axis("off")

    fig.suptitle(
        f"{row.get('data_type', '')} | {row['split']} | {row['case_name']}",
        fontsize=14,
    )

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{row['case_name']}_{row['split']}.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    metadata_path = args.dataset / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    required_cols = [
        "dataid",
        "data_type",
        "case_name",
        "split",
        "is_labeled",
        "image_path",
        "mask_path",
    ]

    missing_cols = [c for c in required_cols if c not in metadata.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in metadata.csv: {missing_cols}")

    df = metadata[metadata["split"] == args.split].copy()

    if args.split == "TrL":
        df = df[df["is_labeled"] == 1].copy()

    if args.split in ["TrL", "TrU"]:
        df = df[
            df["mask_path"].notna() &
            df["mask_path"].astype(str).str.strip().ne("")
        ].copy()

    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No samples found for split={args.split}")

    n_samples = min(args.n_random, len(df))

    selected_df = df.sample(
        n=n_samples,
        random_state=args.seed,
    ).reset_index(drop=True)

    print(f"\nDataset: {args.dataset}")
    print(f"Selected {n_samples} random samples from split={args.split}")

    show_cols = [
        c for c in [
            "dataid",
            "data_type",
            "case_name",
            "split",
            "is_labeled",
            "image_shape",
            "mask_shape",
        ]
        if c in selected_df.columns
    ]

    print(selected_df[show_cols])

    output_dir = args.output_dir / args.dataset.name / args.split

    for _, row in selected_df.iterrows():
        visualize_case(
            row=row,
            dataset_dir=args.dataset,
            output_dir=output_dir,
            axis=args.axis,
            max_slices=args.max_slices,
            only_mask_slices=args.only_mask_slices,
            alpha=args.alpha,
        )

    print("\nDone.")
    print(f"Figures saved in: {output_dir}")


if __name__ == "__main__":
    main()