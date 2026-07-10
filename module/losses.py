"""
Compound loss for the supervised trainer: nnU-Net's own default Dice +
CE loss (DC_and_CE_loss, built from MemoryEfficientSoftDiceLoss and
RobustCrossEntropyLoss -- see nnUNetTrainer._build_loss upstream),
with an optional third BoundaryLoss (Kervadec et al., 2019) term
mixed in via a convex combination once module/nnunet.py's epoch hook
ramps its weight above 0.

The Tversky / Focal / Focal-Tversky region+pixel variants tried across
the phase 1-3 experiment branches didn't outperform plain Dice+CE, so
this file no longer reimplements that family or exposes a configurable
loss_type -- CompoundLoss always uses nnU-Net's default loss, with
BoundaryLoss as the only optional add-on.

BoundaryLoss has no floor forcing any foreground prediction on its
own (an all-background prediction can still score 0), so
CompoundLoss only mixes it in at a small boundary_weight -- see
litmodule.use_boundary / boundary_weight_max / boundary_ramp_epochs in
the experiment configs and module/nnunet.py's epoch hook.
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import nn, Tensor

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1

# Worst-case finite stand-in for a NaN/Inf/blown-up loss (e.g. dice/CE
# exploding on a degenerate batch). nan_to_num first, since plain clamp
# leaves NaN untouched (NaN compares False to both bounds); clamp after
# to actually bound merely-huge-but-finite values too, not just
# literal inf. nan_to_num also zeroes the gradient at the positions it
# replaces, so a bad step is skipped there rather than corrupting the
# model weights -- clamp does the same for the values it bounds.
LOSS_CLIP_VALUE = 1e6


def _clip_loss(loss: Tensor) -> Tensor:
    loss = torch.nan_to_num(loss, nan=LOSS_CLIP_VALUE, posinf=LOSS_CLIP_VALUE, neginf=-LOSS_CLIP_VALUE)
    return torch.clamp(loss, min=-LOSS_CLIP_VALUE, max=LOSS_CLIP_VALUE)


def _onehot_target(x: Tensor, y: Tensor, do_bg: bool) -> Tensor:
    """
    One-hot-encode y to match x's channel layout (pass through as-is if
    y is already one-hot), dropping the background channel unless
    do_bg.
    """

    if x.ndim != y.ndim:
        y = y.view((y.shape[0], 1, *y.shape[1:]))

    if x.shape == y.shape:
        # gt is probably already a one hot encoding
        y_onehot = y.to(torch.float32)
    else:
        y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
        y_onehot.scatter_(1, y.long(), 1)

    if not do_bg:
        y_onehot = y_onehot[:, 1:]

    return y_onehot


def _signed_distance_map(posmask: np.ndarray) -> np.ndarray:
    """
    Signed Euclidean distance transform of one binary foreground mask
    (Kervadec et al., 2019): negative inside the foreground, positive
    outside, ~0 right at the boundary. Empty/full masks (no boundary
    to speak of) map to all-zero, so they contribute nothing.
    """

    if not posmask.any() or posmask.all():
        return np.zeros_like(posmask, dtype=np.float32)

    negmask = ~posmask

    return (distance_transform_edt(negmask) * negmask - (distance_transform_edt(posmask) - 1) * posmask).astype(
        np.float32
    )


class BoundaryLoss(nn.Module):
    """
    Boundary loss (Kervadec et al., 2019): mean_c sum_q phi_G(q) *
    s_theta(q), where phi_G is the signed distance map of the
    ground-truth mask (see _signed_distance_map) and s_theta is the
    predicted softmax foreground probability. Linear in s_theta (phi_G
    is a fixed, non-differentiable target computed under no_grad), so
    unlike Dice it doesn't saturate as predictions approach the true
    mask -- it keeps pushing on whichever pixels are still far from
    the boundary on the wrong side.

    Computed on-the-fly per batch via scipy's distance_transform_edt
    (CPU, one call per batch item per class) rather than precomputed,
    since augmentation changes the mask every step.

    Has no floor forcing any foreground prediction on its own (an
    all-background prediction can still score 0), so CompoundLoss only
    mixes it in at a small boundary_weight, keeping most of the convex
    combination on Dice+CE, which anchors the mask -- see
    CompoundLoss.boundary_weight / set_boundary_weight.
    """

    def __init__(self, apply_nonlin=None, do_bg: bool = False):
        super().__init__()

        self.apply_nonlin = apply_nonlin
        self.do_bg = do_bg

    def forward(self, x: Tensor, y: Tensor, loss_mask=None) -> Tensor:
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        with torch.no_grad():
            y_onehot = _onehot_target(x, y, self.do_bg)
            y_onehot_np = y_onehot.cpu().numpy().astype(bool)

            dist_np = np.stack(
                [
                    _signed_distance_map(y_onehot_np[b, c])
                    for b in range(y_onehot_np.shape[0])
                    for c in range(y_onehot_np.shape[1])
                ]
            ).reshape(y_onehot_np.shape)

            dist = torch.from_numpy(dist_np).to(device=x.device, dtype=x.dtype)

        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is not None:
            num_valid = loss_mask.expand_as(x).sum().clamp_min(1)
            return (dist * x * loss_mask).sum() / num_valid

        return (dist * x).mean()


class CompoundLoss(nn.Module):
    """
    nnU-Net's default Dice + CE loss (DC_and_CE_loss), with an optional
    BoundaryLoss term mixed in via a convex combination:
    (1 - boundary_weight) * dice_ce + boundary_weight * boundary.

    boundary_cls: BoundaryLoss, or None (no boundary term -- the
    common case). Mixed in at self.boundary_weight, which starts at 0.0
    and is updated externally via set_boundary_weight (module/nnunet.py
    ramps it up over epochs when litmodule.use_boundary is set).
    """

    def __init__(self, batch_dice: bool, ddp: bool, ignore_label=None, boundary_cls=None, boundary_kwargs=None):
        super().__init__()

        self.ignore_label = ignore_label
        self.boundary_weight = 0.0

        self.dice_ce = DC_and_CE_loss(
            {"batch_dice": batch_dice, "smooth": 1e-5, "do_bg": False, "ddp": ddp},
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        self.boundary = (
            boundary_cls(apply_nonlin=softmax_helper_dim1, **(boundary_kwargs or {}))
            if boundary_cls is not None
            else None
        )

    def set_boundary_weight(self, weight: float) -> None:
        self.boundary_weight = weight

    def forward(self, net_output: Tensor, target: Tensor) -> Tensor:
        """
        target must be b, c, x, y(, z) with c=1
        """

        dice_ce = self.dice_ce(net_output, target)

        if self.boundary is None or self.boundary_weight <= 0:
            return _clip_loss(dice_ce)

        if self.ignore_label is not None:
            mask = (target != self.ignore_label).bool()
            target_region = torch.where(mask, target, torch.zeros_like(target))

            if mask.sum() == 0:
                return _clip_loss(dice_ce)
        else:
            mask = None
            target_region = target

        boundary = self.boundary(net_output, target_region, loss_mask=mask)

        return _clip_loss((1 - self.boundary_weight) * dice_ce + self.boundary_weight * boundary)
