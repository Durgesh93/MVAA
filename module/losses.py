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
"""

import torch
from torch import nn, Tensor

from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.utilities.helpers import softmax_helper_dim1


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


class CompoundLoss(nn.Module):
    """
    Generic region-loss + pixel-loss compound.

    first_cls: the region loss -- FocalTverskyLoss, or None
    (no region term -- skips building it entirely, not just weighting
    it to 0).
    second_cls: the pixel loss -- FocalLoss, or None (no pixel term,
    same deal). At least one of the two must be given.

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
    ):
        super().__init__()

        assert first_cls is not None or second_cls is not None, (
            "CompoundLoss needs at least one of first_cls/second_cls"
        )

        self.ignore_label = ignore_label

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

        loss = 0

        if self.first is not None:
            loss = loss + self.first(net_output, target_region, loss_mask=mask)

        if self.second is not None and (self.ignore_label is None or num_fg > 0):
            loss = loss + self.second(net_output, target)

        return loss
