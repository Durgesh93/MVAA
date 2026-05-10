#!/usr/bin/env python3
"""Model and loss factory for full-supervised cardiac segmentation."""

from __future__ import annotations

from typing import Sequence

from monai.losses import DiceCELoss
from monai.networks.nets import UNet


def get_model(
    name: str = "unet3d",
    model_size: str = "large",
    in_channels: int = 1,
    out_channels: int = 3,
    channels: Sequence[int] | None = None,
    strides: Sequence[int] = (2, 2, 2, 2),
    num_res_units: int | None = None,
):
    """Create segmentation model by name."""
    name = name.lower()
    model_size = model_size.lower()

    if channels is None or num_res_units is None:
        if model_size == "small":
            channels = (16, 32, 64, 128, 256)
            num_res_units = 2
        elif model_size == "base":
            channels = (24, 48, 96, 192, 384)
            num_res_units = 2
        elif model_size == "large":
            channels = (24, 48, 96, 192, 384)
            num_res_units = 3
        else:
            raise ValueError(f"Unsupported model_size: {model_size}")

    if name in {"unet", "unet3d"}:
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            act="PRELU",
            norm="INSTANCE",
            dropout=0.0,
        )

    raise ValueError(f"Unsupported model name: {name}")


def get_loss_fn():
    """Dice + CE for stable supervised training."""
    return DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=0.8, lambda_ce=0.2)
