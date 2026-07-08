"""
Compound losses for the supervised trainer: Dice/Tversky-family region
losses paired with a CE/Focal/TopK pixel-wise term, plus Unified Focal
Loss. Selected via litmodule.loss_type in the experiment config.

DiceLoss wraps nnunetv2's own MemoryEfficientSoftDiceLoss so the
compound losses below share one Dice implementation/config surface,
instead of each one wiring up nnU-Net's dice module separately (which
is how nnunetv2.training.loss.compound_losses.DC_and_CE_loss and the
focal branch's losses.py each did it independently). TverskyLoss and
FocalTverskyLoss share the same one-hot/tp-fp-fn/DDP machinery via
_soft_tversky_index, generalizing Dice (alpha == beta == 0.5 recovers
it exactly).

FocalLoss, TverskyLoss, FocalTverskyLoss and the Unified Focal Loss
components have no nnunetv2 equivalent (it only ships DC_and_CE_loss /
DC_and_BCE_loss / DC_and_topk_loss), so they're hand-rolled here. The
Unified Focal Loss classes are this file's implementation of the
published construction (Yeung et al., 2021) built from this file's own
building blocks, not a verified byte-for-byte reproduction of the
authors' code -- cross-check against it if exact numbers matter.
"""

import torch
from torch import nn, Tensor

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.utilities.helpers import softmax_helper_dim1


class DiceLoss(nn.Module):
    """
    Thin wrapper around nnU-Net's MemoryEfficientSoftDiceLoss.

    Exists so DC_and_CE_loss and DC_and_Focal_loss below configure and
    call Dice the same way, rather than each importing/wiring nnU-Net's
    dice module on their own.
    """

    def __init__(
        self,
        batch_dice: bool = False,
        smooth: float = 1e-5,
        do_bg: bool = False,
        ddp: bool = True,
    ):
        super().__init__()

        self.dc = MemoryEfficientSoftDiceLoss(
            apply_nonlin=softmax_helper_dim1,
            batch_dice=batch_dice,
            smooth=smooth,
            do_bg=do_bg,
            ddp=ddp,
        )

    def forward(self, net_output: Tensor, target: Tensor, loss_mask=None) -> Tensor:
        return self.dc(net_output, target, loss_mask=loss_mask)


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


class TverskyLoss(nn.Module):
    """
    Soft Tversky loss (Salehi et al., 2017). Generalizes Dice with
    separate false-positive (alpha) / false-negative (beta) weights.

    Raising beta above 0.5 (default here: alpha=0.3, beta=0.7)
    penalizes missed foreground harder than false alarms -- useful
    when the target class is small relative to the frame and recall
    matters more than precision.

    Same -index sign convention as DiceLoss/nnU-Net's own Dice loss
    (lower is better as a training loss).
    """

    def __init__(
        self,
        apply_nonlin=None,
        alpha: float = 0.3,
        beta: float = 0.7,
        batch_dice: bool = False,
        do_bg: bool = False,
        smooth: float = 1e-5,
        ddp: bool = True,
    ):
        super().__init__()

        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.beta = beta
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

        return -tversky.mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss (Abraham & Khan, 2019): (1 - TI)^(1/gamma),
    gamma in [1, 3].

    gamma > 1 raises the loss (and its gradient) relatively more for
    classes with LOW Tversky index -- i.e. whichever class the model
    is currently doing worst on gets relatively more attention, on top
    of the FP/FN reweighting Tversky already does via alpha/beta.

    Unlike TverskyLoss/DiceLoss (which return -index, matching nnU-Net's
    own "lower is better similarity" convention), this returns the
    literal positive (1-TI)^(1/gamma) quantity, the standard Focal
    Tversky convention. Both are just monotonic reparameterizations of
    the same index chosen for a minimize-this-loss objective, so either
    composes fine when summed with a CE/Focal term in a compound loss.
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
    Multiclass focal loss (Lin et al., 2017), operating on logits.

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


class SymmetricFocalCELoss(nn.Module):
    """
    Symmetric focal cross-entropy term from the Unified Focal Loss
    framework (Yeung et al., 2021).

    Unlike this file's plain FocalLoss (Lin et al., 2017 -- which only
    down-weights the *correct* class's already-confident predictions),
    this applies the same (1 - p_c)^gamma focal weighting uniformly to
    every class, and additionally reweights each voxel's contribution
    by delta (foreground classes) / (1 - delta) (background) to
    directly counter class imbalance rather than relying on the focal
    term alone.

    "Foreground" here means any class label != 0 -- a simplification
    of the paper's framing (often binary lesion-vs-background) to the
    multi-class case used in this project.
    """

    def __init__(
        self,
        delta: float = 0.6,
        gamma: float = 0.5,
        ignore_index: int = -100,
    ):
        super().__init__()

        self.delta = delta
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inp: Tensor, target: Tensor) -> Tensor:
        if target.ndim == inp.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]

        target = target.long()

        valid = target != self.ignore_index

        safe_target = target.clone()
        safe_target[~valid] = 0

        log_probs = torch.log_softmax(inp, dim=1)
        probs = log_probs.exp()

        target_onehot = torch.zeros_like(probs).scatter_(
            1, safe_target.unsqueeze(1), 1
        )

        p_t = (probs * target_onehot).sum(1)
        log_p_t = (log_probs * target_onehot).sum(1)

        is_bg = safe_target == 0

        class_weight = torch.where(
            is_bg,
            torch.full_like(p_t, 1 - self.delta),
            torch.full_like(p_t, self.delta),
        )

        loss = -class_weight * ((1 - p_t) ** self.gamma) * log_p_t
        loss = loss[valid]

        if loss.numel() == 0:
            return log_probs.sum() * 0.0

        return loss.mean()


class UnifiedFocalLoss(nn.Module):
    """
    Symmetric Unified Focal Loss (Yeung et al., 2021): a weighted sum
    of a symmetric focal CE term (SymmetricFocalCELoss) and a
    symmetric focal Tversky term, sharing delta/gamma between the two
    so one pair of hyperparameters controls both halves.

    delta in (0, 1): FN weight in the Tversky half (beta=delta,
    alpha=1-delta) and per-class weight in the CE half (delta for
    foreground, 1-delta for background). Higher delta biases both
    halves toward penalizing missed foreground harder.

    gamma in [0, 1): focal sharpening exponent. The Tversky half uses
    (1-TI)^(1-gamma); the CE half uses (1-p_t)^gamma. gamma near 0
    reduces close to plain Dice+CE; gamma near 1 sharpens focus onto
    the hardest classes/voxels.

    lambda_weight balances the two halves: loss = lambda*CE_half +
    (1-lambda)*Tversky_half.

    Built on this file's _soft_tversky_index (the same tp/fp/fn/DDP
    machinery TverskyLoss and FocalTverskyLoss use) and
    SymmetricFocalCELoss, rather than a separate from-scratch
    computation, to reduce the chance of a subtly-wrong new tensor op.
    """

    def __init__(
        self,
        lambda_weight: float = 0.5,
        delta: float = 0.6,
        gamma: float = 0.5,
        batch_dice: bool = False,
        do_bg: bool = False,
        smooth: float = 1e-5,
        ddp: bool = True,
        ignore_label=None,
    ):
        super().__init__()

        assert 0 <= gamma < 1, "gamma must be in [0, 1) for UnifiedFocalLoss"

        self.lambda_weight = lambda_weight
        self.delta = delta
        self.gamma = gamma
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.ddp = ddp
        self.ignore_label = ignore_label

        self.focal_ce = SymmetricFocalCELoss(
            delta=delta,
            gamma=gamma,
            ignore_index=ignore_label if ignore_label is not None else -100,
        )

    def _focal_tversky_term(self, net_output: Tensor, target_tversky: Tensor, mask) -> Tensor:
        probs = softmax_helper_dim1(net_output)

        tversky = _soft_tversky_index(
            probs, target_tversky,
            alpha=1 - self.delta,
            beta=self.delta,
            do_bg=self.do_bg,
            batch_dice=self.batch_dice,
            smooth=self.smooth,
            ddp=self.ddp,
            loss_mask=mask,
        )

        return ((1 - tversky).clamp_min(0) ** (1 - self.gamma)).mean()

    def forward(self, net_output: Tensor, target: Tensor) -> Tensor:
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target "
                "variables (UnifiedFocalLoss)"
            )

            mask = (target != self.ignore_label).bool()
            target_tversky = torch.clone(target)
            target_tversky[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_tversky = target
            mask = None
            num_fg = None

        ft_loss = self._focal_tversky_term(net_output, target_tversky, mask)

        ce_loss = self.focal_ce(net_output, target) \
            if (self.ignore_label is None or num_fg > 0) \
            else net_output.sum() * 0.0

        return self.lambda_weight * ce_loss + (1 - self.lambda_weight) * ft_loss


class DC_and_CE_loss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        weight_ce: float = 1,
        weight_dice: float = 1,
        ignore_label=None,
    ):
        super().__init__()

        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = DiceLoss(**soft_dice_kwargs)

    def forward(self, net_output: Tensor, target: Tensor) -> Tensor:
        """
        target must be b, c, x, y(, z) with c=1
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target "
                "variables (DC_and_CE_loss)"
            )

            mask = (target != self.ignore_label).bool()
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
            num_fg = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0

        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) \
            else 0

        return self.weight_dice * dc_loss + self.weight_ce * ce_loss


class DC_and_Focal_loss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        focal_kwargs,
        weight_focal: float = 1,
        weight_dice: float = 1,
        ignore_label=None,
    ):
        super().__init__()

        if ignore_label is not None:
            focal_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.ignore_label = ignore_label

        self.focal = FocalLoss(**focal_kwargs)
        self.dc = DiceLoss(**soft_dice_kwargs)

    def forward(self, net_output: Tensor, target: Tensor) -> Tensor:
        """
        target must be b, c, x, y(, z) with c=1
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target "
                "variables (DC_and_Focal_loss)"
            )

            mask = (target != self.ignore_label).bool()
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
            num_fg = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0

        focal_loss = self.focal(net_output, target) \
            if self.weight_focal != 0 and (self.ignore_label is None or num_fg > 0) \
            else 0

        return self.weight_dice * dc_loss + self.weight_focal * focal_loss


class Tversky_and_CE_loss(nn.Module):
    def __init__(
        self,
        tversky_kwargs,
        ce_kwargs,
        weight_ce: float = 1,
        weight_tversky: float = 1,
        ignore_label=None,
    ):
        super().__init__()

        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_tversky = weight_tversky
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.tversky = TverskyLoss(apply_nonlin=softmax_helper_dim1, **tversky_kwargs)

    def forward(self, net_output: Tensor, target: Tensor) -> Tensor:
        """
        target must be b, c, x, y(, z) with c=1
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target "
                "variables (Tversky_and_CE_loss)"
            )

            mask = (target != self.ignore_label).bool()
            target_tversky = torch.clone(target)
            target_tversky[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_tversky = target
            mask = None
            num_fg = None

        tversky_loss = self.tversky(net_output, target_tversky, loss_mask=mask) \
            if self.weight_tversky != 0 else 0

        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) \
            else 0

        return self.weight_tversky * tversky_loss + self.weight_ce * ce_loss


class FocalTversky_and_CE_loss(nn.Module):
    def __init__(
        self,
        focal_tversky_kwargs,
        ce_kwargs,
        weight_ce: float = 1,
        weight_focal_tversky: float = 1,
        ignore_label=None,
    ):
        super().__init__()

        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_focal_tversky = weight_focal_tversky
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.focal_tversky = FocalTverskyLoss(
            apply_nonlin=softmax_helper_dim1, **focal_tversky_kwargs
        )

    def forward(self, net_output: Tensor, target: Tensor) -> Tensor:
        """
        target must be b, c, x, y(, z) with c=1
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target "
                "variables (FocalTversky_and_CE_loss)"
            )

            mask = (target != self.ignore_label).bool()
            target_tversky = torch.clone(target)
            target_tversky[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_tversky = target
            mask = None
            num_fg = None

        focal_tversky_loss = self.focal_tversky(net_output, target_tversky, loss_mask=mask) \
            if self.weight_focal_tversky != 0 else 0

        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) \
            else 0

        return self.weight_focal_tversky * focal_tversky_loss + self.weight_ce * ce_loss


class DC_and_TopK_loss(nn.Module):
    """
    Dice + TopK CE, where TopK CE only backpropagates through the k%
    hardest voxels per batch (hard-example mining). TopKLoss is
    nnunetv2's own (nnunetv2.training.loss.robust_ce_loss.TopKLoss),
    reused directly here; only DiceLoss is this file's own wrapper, so
    this combo uses the same Dice component as DC_and_CE_loss rather
    than nnunetv2's older SoftDiceLoss (which is what
    nnunetv2.training.loss.compound_losses.DC_and_topk_loss uses).
    """

    def __init__(
        self,
        soft_dice_kwargs,
        topk_kwargs,
        weight_ce: float = 1,
        weight_dice: float = 1,
        ignore_label=None,
    ):
        super().__init__()

        if ignore_label is not None:
            topk_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**topk_kwargs)
        self.dc = DiceLoss(**soft_dice_kwargs)

    def forward(self, net_output: Tensor, target: Tensor) -> Tensor:
        """
        target must be b, c, x, y(, z) with c=1
        """

        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target "
                "variables (DC_and_TopK_loss)"
            )

            mask = (target != self.ignore_label).bool()
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
            num_fg = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0

        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) \
            else 0

        return self.weight_dice * dc_loss + self.weight_ce * ce_loss
