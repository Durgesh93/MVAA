"""
Compound losses for the supervised trainer: a Dice/Tversky-family
region loss paired with a CE/Focal pixel-wise term. Selected via
litmodule.loss_type in the experiment config -- module/nnunet.py's
_nnu_build_loss picks which classes to hand to CompoundLoss below,
passing the correct alpha/beta/gamma coefficients explicitly for
whichever special case it wants (Dice, Tversky, Focal Tversky, CE,
Focal CE).

Every class here also works standalone as a single loss
(FocalTverskyLoss, FocalLoss); CompoundLoss is only needed when
combining two of them.

Two base implementations, each covering its whole family via
parameters instead of separate classes per special case:

    FocalTverskyLoss(alpha, beta, gamma) -- region loss, built on
        _soft_tversky_index.
        alpha=beta=0.5, gamma=1 -> Dice
        alpha, beta, gamma=1    -> Tversky (Salehi et al., 2017)
        alpha, beta, gamma      -> Focal Tversky (Abraham & Khan, 2019)

    FocalLoss(gamma) -- per-voxel pixel loss.
        gamma=0 -> plain CE
        gamma   -> Focal CE (Lin et al., 2017)

FocalTverskyLoss operates on a region-level aggregate (one Tversky
index per class, summed over every voxel first); FocalLoss operates
per voxel (each voxel's own predicted probability for its true class).
Different inputs, different granularity -- they are genuinely two
implementations, not one formula with a switch.

There are no wrapper classes for special cases (Dice, plain Tversky,
plain CE) -- callers just construct FocalTverskyLoss/FocalLoss with
the coefficients for whichever special case they want (e.g.
FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=1) for Dice, FocalLoss
(gamma=0) for plain CE).

CompoundLoss combines one region-loss instance (first_cls) and one
pixel-loss instance (second_cls) into a compound loss -- either side
may be None to skip it entirely (not just weight it to 0), but not
both. There are no named preset classes for specific combinations
(e.g. "Dice + CE") -- module/nnunet.py constructs CompoundLoss
directly with whichever first_cls/second_cls and coefficients the
configured litmodule.loss_type calls for.

Note: FocalTverskyLoss returns the positive (1-TI)^(1/gamma) quantity
rather than an older bare -TI convention, so the region term is offset
by a constant +1 versus before this file was unified onto one
region-loss class. Gradient-identical (constant offsets don't affect
optimization), cosmetic only -- it just shifts the logged/reported
loss value.

BoundaryLoss (Kervadec et al., 2019) is a third, optional term, mixed
in via a convex combination against whichever first_cls/second_cls
combination is already selected: (1 - boundary_weight) * compound +
boundary_weight * boundary. It has no floor forcing any foreground
prediction on its own, so CompoundLoss keeps boundary_weight small and
(typically ramped-up-from-zero) via set_boundary_weight -- see
litmodule.use_boundary / boundary_weight_max / boundary_ramp_epochs in
the experiment configs and module/nnunet.py's epoch hook.
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import nn, Tensor

from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.utilities.helpers import softmax_helper_dim1


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


def _soft_tversky_index(
    x: Tensor,
    y: Tensor,
    alpha: float,
    beta: float,
    do_bg: bool,
    batch_dice: bool,
    smooth: float,
    ddp: bool,
    loss_mask=None,
) -> Tensor:
    """
    Soft Tversky index per class, mirroring nnunetv2's
    MemoryEfficientSoftDiceLoss (sum_pred/sum_gt/intersect instead of
    full per-voxel tp/fp/fn tensors -- fp = sum_pred - intersect,
    fn = sum_gt - intersect).

    x must already be post-nonlinearity (softmax) probabilities.

    TI = TP / (TP + alpha*FP + beta*FN). alpha == beta == 0.5 recovers
    Dice exactly: TI = intersect / (0.5*sum_pred + 0.5*sum_gt)
                     = 2*intersect / (sum_pred + sum_gt) = Dice.
    """

    axes = tuple(range(2, x.ndim))

    with torch.no_grad():
        y_onehot = _onehot_target(x, y, do_bg)

        sum_gt = y_onehot.sum(axes, dtype=torch.float32) if loss_mask is None \
            else (y_onehot * loss_mask).sum(axes, dtype=torch.float32)

    if not do_bg:
        x = x[:, 1:]

    if loss_mask is None:
        intersect = (x * y_onehot).sum(axes, dtype=torch.float32)
        sum_pred = x.sum(axes, dtype=torch.float32)
    else:
        intersect = (x * y_onehot * loss_mask).sum(axes, dtype=torch.float32)
        sum_pred = (x * loss_mask).sum(axes, dtype=torch.float32)

    if batch_dice:
        if ddp:
            intersect = AllGatherGrad.apply(intersect).sum(0, dtype=torch.float32)
            sum_pred = AllGatherGrad.apply(sum_pred).sum(0, dtype=torch.float32)
            sum_gt = AllGatherGrad.apply(sum_gt).sum(0, dtype=torch.float32)

        intersect = intersect.sum(0, dtype=torch.float32)
        sum_pred = sum_pred.sum(0, dtype=torch.float32)
        sum_gt = sum_gt.sum(0, dtype=torch.float32)

    tversky = (intersect + smooth) / (
        intersect
        + alpha * (sum_pred - intersect)
        + beta * (sum_gt - intersect)
        + smooth
    ).clamp_min(1e-8)

    return tversky


class FocalTverskyLoss(nn.Module):
    """
    Region loss, built on _soft_tversky_index: (1 - TI)^(1/gamma).

    The single region-loss implementation in this file -- Dice and
    plain Tversky are both special cases via alpha/beta/gamma:

        alpha=beta=0.5, gamma=1  -> Dice
        alpha, beta, gamma=1     -> Tversky (Salehi et al., 2017)
        alpha, beta, gamma       -> Focal Tversky (Abraham & Khan, 2019)

    gamma > 1 raises the loss (and its gradient) relatively more for
    classes with LOW Tversky index -- i.e. whichever class the model
    is currently doing worst on gets relatively more attention, on top
    of the FP/FN reweighting alpha/beta already does. gamma == 1 turns
    this off ((1-TI)^1 = 1-TI).

    Returns the positive (1-TI)^(1/gamma) quantity (the standard Focal
    Tversky convention) rather than a bare -TI -- see the module
    docstring for the constant-offset implication at gamma=1.
    """

    def __init__(
        self,
        apply_nonlin=None,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 4 / 3,
        batch_dice: bool = False,
        do_bg: bool = False,
        smooth: float = 1e-5,
        ddp: bool = True,
    ):
        super().__init__()

        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x: Tensor, y: Tensor, loss_mask=None) -> Tensor:
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tversky = _soft_tversky_index(
            x, y,
            alpha=self.alpha,
            beta=self.beta,
            do_bg=self.do_bg,
            batch_dice=self.batch_dice,
            smooth=self.smooth,
            ddp=self.ddp,
            loss_mask=loss_mask,
        )

        focal_tversky = (1 - tversky).clamp_min(0) ** (1 / self.gamma)

        return focal_tversky.mean()


class FocalLoss(nn.Module):
    """
    Per-voxel multiclass focal loss (Lin et al., 2017), operating on
    logits.

    The single pixel-loss implementation in this file -- plain CE is
    the gamma=0 special case ((1-pt)^0 = 1, so loss reduces to exactly
    ce_loss).

    Same input convention as nnunetv2's RobustCrossEntropyLoss: target
    may be (b, x, y(, z)) or (b, 1, x, y(, z)).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight=None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.gamma = gamma
        self.ignore_index = ignore_index

        self.ce = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction="none",
            label_smoothing=label_smoothing,
        )

    def forward(self, inp: Tensor, target: Tensor) -> Tensor:
        if target.ndim == inp.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]

        target = target.long()

        ce_loss = self.ce(inp, target)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss

        valid = target != self.ignore_index
        num_valid = valid.sum().clamp(min=1)

        return loss.sum() / num_valid


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

    return (
        distance_transform_edt(negmask) * negmask
        - (distance_transform_edt(posmask) - 1) * posmask
    ).astype(np.float32)


class BoundaryLoss(nn.Module):
    """
    Boundary loss (Kervadec et al., 2019): mean_c sum_q phi_G(q) *
    s_theta(q), where phi_G is the signed distance map of the
    ground-truth mask (see _signed_distance_map) and s_theta is the
    predicted softmax foreground probability. Linear in s_theta (phi_G
    is a fixed, non-differentiable target computed under no_grad), so
    unlike Dice/Tversky it doesn't saturate as predictions approach
    the true mask -- it keeps pushing on whichever pixels are still
    far from the boundary on the wrong side.

    Computed on-the-fly per batch via scipy's distance_transform_edt
    (CPU, one call per batch item per class) rather than precomputed,
    since augmentation changes the mask every step.

    Has no floor forcing any foreground prediction on its own (an
    all-background prediction can still score 0), so CompoundLoss only
    mixes it in at a small boundary_weight, keeping most of the convex
    combination on the region loss that anchors the mask -- see
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
    Generic region-loss + pixel-loss compound, with an optional third
    boundary-loss term mixed in via a convex combination:
    (1 - boundary_weight) * (first + second) + boundary_weight * boundary.

    first_cls: the region loss -- FocalTverskyLoss, or None
    (no region term -- skips building it entirely, not just weighting
    it to 0).
    second_cls: the pixel loss -- FocalLoss, or None (no pixel term,
    same deal). At least one of first_cls/second_cls must be given.
    boundary_cls: BoundaryLoss, or None (no boundary term -- the
    common case). Mixed in at self.boundary_weight, which starts at 0.0
    and is updated externally via set_boundary_weight (module/nnunet.py
    ramps it up over epochs when litmodule.use_boundary is set).

    No named presets -- module/nnunet.py constructs this directly with
    whichever first_cls/second_cls and coefficients the configured
    litmodule.loss_type calls for (e.g. FocalTverskyLoss(alpha=0.5,
    beta=0.5, gamma=1) + FocalLoss(gamma=0) for Dice + plain CE).
    """

    def __init__(
        self,
        first_cls,
        first_kwargs,
        second_cls,
        second_kwargs,
        ignore_label=None,
        boundary_cls=None,
        boundary_kwargs=None,
    ):
        super().__init__()

        assert first_cls is not None or second_cls is not None, (
            "CompoundLoss needs at least one of first_cls/second_cls"
        )

        self.ignore_label = ignore_label
        self.boundary_weight = 0.0

        self.first = (
            first_cls(apply_nonlin=softmax_helper_dim1, **first_kwargs)
            if first_cls is not None else None
        )

        if second_cls is not None:
            second_kwargs = dict(second_kwargs)

            if ignore_label is not None:
                second_kwargs["ignore_index"] = ignore_label

            self.second = second_cls(**second_kwargs)
        else:
            self.second = None

        self.boundary = (
            boundary_cls(apply_nonlin=softmax_helper_dim1, **(boundary_kwargs or {}))
            if boundary_cls is not None else None
        )

    def set_boundary_weight(self, weight: float) -> None:
        self.boundary_weight = weight

    def forward(self, net_output: Tensor, target: Tensor) -> Tensor:
        """
        target must be b, c, x, y(, z) with c=1
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target "
                f"variables ({type(self).__name__})"
            )

            mask = (target != self.ignore_label).bool()
            target_region = torch.clone(target)
            target_region[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_region = target
            mask = None
            num_fg = None

        valid = self.ignore_label is None or num_fg > 0

        compound = 0

        if self.first is not None:
            compound = compound + self.first(net_output, target_region, loss_mask=mask)

        if self.second is not None and valid:
            compound = compound + self.second(net_output, target)

        if self.boundary is not None and valid and self.boundary_weight > 0:
            boundary = self.boundary(net_output, target_region, loss_mask=mask)
            return (1 - self.boundary_weight) * compound + self.boundary_weight * boundary

        return compound
