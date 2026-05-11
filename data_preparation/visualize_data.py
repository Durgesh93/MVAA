from pathlib import Path
import argparse

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset folder, e.g. Dataset001_MVAA_CT_SSL",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="TrL",
        choices=["TrL", "TrU", "Ts"],
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Downsample factor for faster browser rendering",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/volume.html"),
    )

    return parser.parse_args()


def load_nifti(path):
    nii = nib.load(str(path))
    arr = np.asanyarray(nii.dataobj)
    return arr


def normalize_volume(vol):
    vol = vol.astype(np.float32)

    vmin, vmax = np.percentile(vol, [1, 99])

    vol = np.clip(vol, vmin, vmax)
    vol = (vol - vmin) / (vmax - vmin + 1e-8)

    return vol


def main():
    args = parse_args()

    metadata = pd.read_csv(args.dataset / "metadata.csv")
    split_df = metadata[metadata["split"] == args.split].reset_index(drop=True)

    if split_df.empty:
        raise ValueError(f"No samples found for split={args.split}")

    row = split_df.iloc[args.index]

    image_path = args.dataset / row["image_path"]

    image = load_nifti(image_path)

    print("Image:")
    print(f"  path : {image_path}")
    print(f"  shape: {image.shape}")
    print(f"  dtype: {image.dtype}")

    if image.ndim != 3:
        raise ValueError(f"This script is for 3D volumes only. Got shape={image.shape}")

    step = args.downsample
    vol = image[::step, ::step, ::step]
    vol = normalize_volume(vol)

    x, y, z = np.mgrid[
        0:vol.shape[0],
        0:vol.shape[1],
        0:vol.shape[2],
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=vol.flatten(),
            opacity=0.08,
            surface_count=20,
            name="image",
        )
    )

    has_mask = (
        isinstance(row["mask_path"], str)
        and row["mask_path"].strip() != ""
    )

    if has_mask:
        mask_path = args.dataset / row["mask_path"]

        if mask_path.exists():
            mask = load_nifti(mask_path)
            print("Mask:")
            print(f"  path : {mask_path}")
            print(f"  shape: {mask.shape}")
            print(f"  labels: {np.unique(mask)}")

            mask_ds = mask[::step, ::step, ::step]
            mask_points = np.argwhere(mask_ds > 0)

            # limit number of mask points to keep browser responsive
            max_points = 50000
            if len(mask_points) > max_points:
                idx = np.linspace(0, len(mask_points) - 1, max_points).astype(int)
                mask_points = mask_points[idx]

            fig.add_trace(
                go.Scatter3d(
                    x=mask_points[:, 0],
                    y=mask_points[:, 1],
                    z=mask_points[:, 2],
                    mode="markers",
                    marker=dict(
                        size=2,
                        opacity=0.35,
                        color="red",
                    ),
                    name="mask",
                )
            )

    fig.update_layout(
        title=(
            f"{row['data_type']} | {row['split']} | "
            f"case={row['case_id']} | dataid={row['dataid']}"
        ),
        width=1000,
        height=850,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.output))

    print(f"\nSaved interactive HTML  {args.output}")
    print("Open this HTML file in a browser.")


if __name__ == "__main__":
    main()
