"""
Compound loss for the supervised trainer: Dice (MemoryEfficientSoftDiceLoss,
do_bg=False) + a foreground-weighted CE (_foreground_weighted_ce, not
nnU-Net's own DC_and_CE_loss/RobustCrossEntropyLoss, which averages CE per
voxel with no class weighting -- see _foreground_weighted_ce's docstring
for why that's worth overriding for a task like CT where background is
~98% of a crop), with an optional third BoundaryLoss (Kervadec et al.,
2019) term mixed in via a convex combination once module/nnunet.py's
epoch hook ramps its weight above 0.

The Tversky / Focal / Focal-Tversky region+pixel variants tried across
the phase 1-3 experiment branches didn't outperform plain Dice+CE, so
this file doesn't reimplement that family or expose a configurable
loss_type -- CompoundLoss always uses this Dice+CE combination, with
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


def _foreground_weighted_ce(net_output: Tensor, target: Tensor, loss_mask=None, foreground_weight: float = 1.0) -> Tensor:
    """
    Standard per-voxel cross-entropy with a static class weight (background
    at 1.0, every other class at foreground_weight), not renormalized by
    each class's own voxel count. Ground-truth labels are trustworthy here
    (real annotations, not pseudo-labels), so biasing the gradient toward
    the rare class (e.g. CT's mitral valve, ~2% of voxels) is safe -- unlike
    the pseudo-label branch (WeakStrongPseudoLabelLoss), there's no risk of
    amplifying a wrong self-generated guess.

    A per-class-averaged-then-uniformly-combined version of this was tried
    and reverted (2026-07-23 session): renormalizing by each class's own
    voxel count made the rare class's contribution high-variance batch to
    batch. A static weight avoids that -- it scales the gradient without
    renormalizing, so it doesn't get noisier just because a batch happens
    to have few foreground voxels.

    target: (B, 1, ...) integer class-id map. loss_mask: same shape,
    boolean/0-1, voxels to include (ignore_label support).
    """

    target_flat = target[:, 0].long()

    num_classes = net_output.shape[1]
    class_weight = torch.ones(num_classes, device=net_output.device, dtype=net_output.dtype)
    class_weight[1:] = foreground_weight

    per_voxel_ce = nn.functional.cross_entropy(net_output, target_flat, weight=class_weight, reduction="none")

    valid = loss_mask[:, 0].bool() if loss_mask is not None else torch.ones_like(target_flat, dtype=torch.bool)

    if not valid.any():
        return torch.zeros((), device=net_output.device, dtype=net_output.dtype)

    return per_voxel_ce[valid].mean()


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
    nnU-Net's default Dice + CE combination, but with CE made foreground-
    weighted (_foreground_weighted_ce) instead of nnU-Net's own
    DC_and_CE_loss (plain per-voxel CE) -- for a task like CT where
    background is ~98% of a crop and the real class is ~2%, a plain
    per-voxel CE average is dominated by background. Dice was already
    immune to this (do_bg=False, per-class by construction); CE wasn't, so
    this is the matching fix on that side. foreground_weight defaults to
    1.0 (plain CE, no-op) -- set it via config for tasks where the rare
    class needs a boost.

    Optional BoundaryLoss term mixed in via a convex combination:
    (1 - boundary_weight) * dice_ce + boundary_weight * boundary.

    boundary_cls: BoundaryLoss, or None (no boundary term -- the
    common case). Mixed in at self.boundary_weight, which starts at 0.0
    and is updated externally via set_boundary_weight (module/nnunet.py
    ramps it up over epochs when litmodule.use_boundary is set).
    """

    def __init__(
        self,
        batch_dice: bool,
        ddp: bool,
        ignore_label=None,
        boundary_cls=None,
        boundary_kwargs=None,
        foreground_weight: float = 1.0,
    ):
        super().__init__()

        self.ignore_label = ignore_label
        self.boundary_weight = 0.0
        self.foreground_weight = foreground_weight

        self.dc = MemoryEfficientSoftDiceLoss(
            apply_nonlin=softmax_helper_dim1, batch_dice=batch_dice, smooth=1e-5, do_bg=False, ddp=ddp
        )

        self.boundary = (
            boundary_cls(apply_nonlin=softmax_helper_dim1, **(boundary_kwargs or {}))
            if boundary_cls is not None
            else None
        )

    def set_boundary_weight(self, weight: float) -> None:
        self.boundary_weight = weight

    def _dice_ce(self, net_output: Tensor, target: Tensor) -> Tensor:
        if self.ignore_label is not None:
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, torch.zeros_like(target))
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask)
        ce_loss = _foreground_weighted_ce(net_output, target, loss_mask=mask, foreground_weight=self.foreground_weight)

        return dc_loss + ce_loss

    def forward(self, net_output: Tensor, target: Tensor) -> Tensor:
        """
        target must be b, c, x, y(, z) with c=1
        """

        dice_ce = self._dice_ce(net_output, target)

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


class AdaptiveConfidenceThreshold(nn.Module):
    """
    FreeMatch-style (Wang et al. 2023, arxiv 2205.07246) self-adaptive
    per-class confidence threshold, replacing a single flat cutoff.

    Maintains a global EMA of the batch's mean max-confidence (regardless
    of predicted class) and a per-class EMA of confidence conditioned on
    argmax == c, both updated from the *entire* unlabeled batch every step
    -- not just pixels that already cleared some bar -- since the point is
    to track how confidence is trending even for classes not clearing it
    yet. The per-class threshold is the global EMA scaled by that class's
    confidence relative to the best-tracked class (MaxNorm): a class the
    network is relatively unsure about gets a lower bar than a flat
    threshold would give it, while never exceeding the global EMA. Both
    EMAs start at 1/num_classes (uniform-prior confidence), so thresholds
    start low and rise as the network's real confidence rises -- this
    supplements rather than replaces the hard `pseudo_warmup_epochs` gate
    in lightning_module.py, which still fully zeroes the loss early on.

    Buffers are lazily (re)initialized on first use / on a class-count
    change so this doesn't need num_classes at construction time.
    """

    def __init__(self, momentum: float = 0.999):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("global_ema", torch.zeros(()))
        self.register_buffer("class_ema", torch.zeros(0))

    def _maybe_init(self, num_classes, device, dtype):
        if self.class_ema.numel() != num_classes:
            init_value = 1.0 / num_classes
            self.global_ema = torch.full((), init_value, device=device, dtype=dtype)
            self.class_ema = torch.full((num_classes,), init_value, device=device, dtype=dtype)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # class_ema/global_ema are lazily shaped by num_classes (see
        # _maybe_init), so a freshly constructed module -- e.g. for
        # predict/retrain, which load a checkpoint before any unlabeled batch
        # has run -- still has them at their construction-time shape. Resize
        # to whatever shape the checkpoint has *before* the default load
        # logic runs its shape check, so the real EMA values get restored
        # instead of tripping a size mismatch.
        class_key, global_key = prefix + "class_ema", prefix + "global_ema"

        if class_key in state_dict and state_dict[class_key].shape != self.class_ema.shape:
            self.class_ema = torch.empty_like(state_dict[class_key])
            self.global_ema = torch.empty_like(state_dict[global_key])

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def update(self, confidence: Tensor, pseudo_label: Tensor, num_classes: int):
        self._maybe_init(num_classes, confidence.device, confidence.dtype)

        m = self.momentum
        self.global_ema.mul_(m).add_(confidence.mean(), alpha=1 - m)

        for c in range(num_classes):
            class_mask = pseudo_label == c

            if class_mask.any():
                self.class_ema[c].mul_(m).add_(confidence[class_mask].mean(), alpha=1 - m)

    def per_class_threshold(self, num_classes: int) -> Tensor:
        self._maybe_init(num_classes, self.class_ema.device, self.class_ema.dtype)
        normalized = self.class_ema / self.class_ema.max().clamp_min(1e-12)
        return self.global_ema * normalized


class WeakStrongPseudoLabelLoss(nn.Module):
    """
    FixMatch-style confidence-thresholded consistency loss for TrU samples.

    Confidence-thresholded pseudo-label cross-entropy (Lee, 2013), sourced
    from a weak/strong view pair rather than a single self-trained view: the
    pseudo-label and confidence mask come from the weak view's own
    prediction (geometric-only, no intensity aug -- computed under no_grad
    by the caller), while the loss is the masked CE between that
    pseudo-label and one or more strong (weak + intensity aug) views'
    predictions. Two differently-augmented views of the same voxel-aligned
    patch give a genuine augmentation-invariance signal, rather than a
    network training to agree with its own single-view guess. Independent
    of CompoundLoss/ignore_label (none of the 3 MVAA datasets define one) --
    pixels below the confidence threshold are excluded entirely rather than
    distilled from a likely-wrong guess.

    Confidence threshold is per-class and EMA-tracked via
    AdaptiveConfidenceThreshold -- see that class's docstring.

    The CE itself is a single flat masked mean over confident voxels, no
    class weighting or per-class averaging of any kind (tried twice --
    per-class-averaged-then-uniformly-combined, then a static foreground
    weight -- and reverted both times, 2026-07-23 session). Unlike
    CompoundLoss's ground-truth-labeled CE, this loss's targets are the
    model's own weak-view predictions -- deliberately *not* biasing it
    toward foreground, since doing so on self-generated pseudo-labels
    risks reinforcing whatever the model currently over/under-predicts
    for the rare class rather than correcting it. The (already
    class-conditioned) AdaptiveConfidenceThreshold is the only place this
    loss discriminates between classes -- filtering *which* voxels count,
    not how much each one is weighted once it does.
    """

    def __init__(self, adaptive_momentum: float = 0.999):
        super().__init__()
        self.adaptive_threshold = AdaptiveConfidenceThreshold(momentum=adaptive_momentum)

    def forward(self, weak_logits: Tensor, strong_logits_list):
        """
        Returns (loss, confident_frac, per_class_confident_frac).

        confident_frac is the aggregate fraction of weak-view pixels that
        cleared the threshold (num_confident / confident_mask.numel()) --
        but that aggregate is computed over every pixel regardless of
        class, so for a class occupying only a couple percent of a
        volume (e.g. CT's mitral valve, ~2% of voxels) it's almost
        entirely a background-confidence artifact and stays pinned near
        1.0 whether or not the rare foreground class's pseudo-labels are
        actually trustworthy. per_class_confident_frac breaks the same
        ratio out per class (indexed by class id, NaN where a class
        doesn't appear in this batch's pseudo-labels at all -- torchmetrics'
        MeanMetric silently drops NaN updates, so this is safe to feed
        straight into per-step logging without a batch necessarily
        containing every class) so callers can see whether the rare
        classes are actually being filtered sensibly, not just the
        dominant one. See lightning_module.py's _pseudo_loss.
        """

        with torch.no_grad():
            probs = torch.softmax(weak_logits, dim=1)
            confidence, pseudo_label = probs.max(dim=1)

            num_classes = weak_logits.shape[1]
            self.adaptive_threshold.update(confidence, pseudo_label, num_classes)
            per_class_threshold = self.adaptive_threshold.per_class_threshold(num_classes)
            threshold = per_class_threshold[pseudo_label]

            confident_mask = confidence >= threshold

            per_class_confident_frac = torch.full((num_classes,), float("nan"), device=weak_logits.device)
            for c in range(num_classes):
                class_mask = pseudo_label == c
                class_count = class_mask.sum()
                if class_count > 0:
                    per_class_confident_frac[c] = (confident_mask & class_mask).sum().float() / class_count.float()

        num_confident = confident_mask.sum()
        confident_frac = num_confident.float() / confident_mask.numel()

        if num_confident == 0:
            zero = torch.zeros((), device=weak_logits.device, dtype=weak_logits.dtype)
            return zero, confident_frac, per_class_confident_frac

        view_losses = []

        for strong_logits in strong_logits_list:
            per_pixel_ce = nn.functional.cross_entropy(strong_logits, pseudo_label, reduction="none")
            view_losses.append(per_pixel_ce[confident_mask].mean())

        return torch.stack(view_losses).mean(), confident_frac, per_class_confident_frac
