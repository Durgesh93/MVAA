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

A FixMatch-style WeakStrongPseudoLabelLoss and its FreeMatch
AdaptiveConfidenceThreshold lived here too. Both were removed: the
ssl_sweep_bnd_weakstrong grid found lambda_pseudo=0.1 indistinguishable
from 0 and lambda_pseudo>=0.5 actively worse, so the term did not earn its
keep.

ClusteringCMLoss replaces it with a different semi-supervised idea --
SuperCM-style clustering of the per-voxel embeddings that
module/adapter.py exposes, rather than consistency between augmented
views. It is combined with CompoundLoss by the LightningModule, not
wrapped here, because CompoundLoss is already inside nnU-Net's
DeepSupervisionWrapper by the time training_step sees it.
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
    the rare class (e.g. CT's mitral valve, ~2% of voxels) is safe: there
    is no self-generated guess here that a heavier weight could amplify.

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


# =============================================================================
# SuperCM-style clustering regularizer
# =============================================================================


class ClusteringCMLoss(nn.Module):
    """
    Soft-k-means ("clustering module") regularizer on per-voxel embeddings.

    Penalises mean_voxels sum_k g_k * ||x - mu_k||^2, where x is the
    embedding_dim-dimensional embedding a voxel carries (see
    module/adapter.py), g is the network's own softmax over the class
    logits at that voxel, and mu_k is a class centroid. Minimising it pulls
    every voxel's embedding toward the centroid of whichever class the
    network already believes it belongs to, which tightens the class
    clusters without needing a label at that voxel -- the SuperCM idea.

    The centroids are NOT learned by gradient. They are re-estimated in
    closed form every step as the assignment-weighted mean
    mu_k = sum_p y_pk x_p / sum_p y_pk, then EMA-blended into the stored
    value. mu is therefore a buffer, not a parameter, and carries no
    gradient: the only thing this loss trains is the embedding itself.

    cm_mode picks where the assignments y for that estimate come from:
        "l"   : ground truth one-hot on the labeled crops. The honest
                first experiment -- it asks whether tightening clusters
                around TRUE class means helps at all, with no
                pseudo-labelling anywhere in the loop.
        "u"   : confident pseudo-labels on the unlabeled crops.
        "l+u" : both, concatenated. The actual semi-supervised claim.
    "l" needs no unlabeled forward pass at all, so it costs one extra
    einsum per step and nothing else -- see needs_unlabeled.

    Cross-entropy is deliberately NOT part of this module. The supervised
    term here is nnU-Net's own DeepSupervisionWrapper(CompoundLoss), which
    the LightningModule already owns; folding a second CE in (as the 2D
    supercm project's SuperCMLoss does) would just shadow it. The
    LightningModule forms total = sup_loss + weight_at(step) * energy.

    Shapes: embeddings (B, D, *spatial), logits (B, C, *spatial), target
    (B, 1, *spatial) integer class ids. Under deep supervision the caller
    passes the highest-resolution stage only -- the coarser stages are
    downsampled views of the same embedding space, and pooling them into
    one set of centroids would weight a 7x6x8 map as heavily as the
    full-resolution one.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        cm_mode: str = "l",
        weight: float = 0.1,
        warmup_steps: int = 500,
        ema_weight: float = 0.95,
        num_voxels=32768,
        conf_thr: float = 0.9,
        topk_frac: float = 0.15,
        min_keep: int = 64,
        normalize: bool = False,
        ignore_label=None,
        eps: float = 1e-6,
    ):
        super().__init__()

        if cm_mode not in ("l", "u", "l+u"):
            raise ValueError(f"cm_mode must be one of 'l', 'u', 'l+u', got {cm_mode!r}")

        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.cm_mode = cm_mode
        self.weight = float(weight)
        self.warmup_steps = int(warmup_steps)
        self.ema_weight = float(ema_weight)
        self.num_voxels = None if num_voxels is None else int(num_voxels)
        self.conf_thr = float(conf_thr)
        self.topk_frac = float(topk_frac)
        self.min_keep = int(min_keep)
        self.normalize = bool(normalize)
        self.ignore_label = ignore_label
        self.eps = float(eps)

        # Buffers, not parameters: closed-form estimated, never optimised.
        # Registered so they ride along in the checkpoint -- a resumed run
        # that re-initialised mu from scratch would hand the embedding a
        # step-change in its target on the first batch after the restore.
        self.register_buffer("mu", torch.zeros(self.num_classes, self.embedding_dim))

        # Per-class, not a single flag. A class absent from the first batch
        # (easy with two thin leaflets and a 1-2 case labeled pool) would
        # otherwise have its centroid marked "initialised" at the zero
        # vector and then EMA-crawl toward the truth for hundreds of steps,
        # dragging every voxel of that class toward the origin meanwhile.
        self.register_buffer("mu_initialized", torch.zeros(self.num_classes, dtype=torch.bool))

    # -------------------------------------------------------------------------
    # Scheduling
    # -------------------------------------------------------------------------
    @property
    def needs_unlabeled(self) -> bool:
        """True when cm_mode sources centroids from the unlabeled stream."""
        return self.cm_mode in ("u", "l+u")

    def weight_at(self, step: int) -> float:
        """
        Linearly ramped loss weight, 0 -> self.weight over warmup_steps.

        Driven by the caller's step (Lightning's global_step) rather than an
        internal counter. The 2D supercm project incremented its own buffer
        inside this function and ALSO used that buffer to decide
        hard-copy-vs-EMA for the centroids, so a warmup of 0 silently
        disabled the EMA; keeping the two concerns separate avoids that.

        The ramp matters because mu is meaningless before it has seen any
        data: at weight 0 on step 0 the loss cannot pull embeddings toward
        a zero vector.
        """

        if self.warmup_steps <= 0:
            return self.weight

        return self.weight * min(float(step) / float(self.warmup_steps), 1.0)

    # -------------------------------------------------------------------------
    # Flattening helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _flatten_spatial(tensor: Tensor) -> Tensor:
        """(B, C, *spatial) -> (B, C, P). A view when the input is contiguous."""
        return tensor.flatten(2)

    def _valid_mask(self, labels: Tensor) -> Tensor:
        """
        (B, P) bool of voxels that may contribute, from the ignore label.

        nnU-Net's RemoveLabelTansform maps the -1 padding to 0 before the
        loss sees it, so this is only about a genuine configured
        ignore_label; without one, everything is valid.
        """

        if self.ignore_label is None:
            return torch.ones_like(labels, dtype=torch.bool)

        return labels != int(self.ignore_label)

    def _gt_assignments(self, target: Tensor):
        """
        Ground-truth one-hot assignments from an integer label map.

        Returns (y, valid) with y (B, C, P) float and valid (B, P) bool.

        NOTE this is where a verbatim port of the 2D supercm project breaks:
        its _gt_onehot branches on the target's CHANNEL count, and an
        nnU-Net target is (B, 1, *spatial), so it would take the binary
        path and read the integer id 2 ("anterior_leaflet") as a
        foreground probability of 2.0. The label map has to be expanded
        against num_classes, not against its own channel dim.
        """

        labels = target[:, 0].reshape(target.shape[0], -1).long()
        valid = self._valid_mask(labels)

        # Ignored voxels are clamped into range purely so one_hot cannot
        # fault on them; `valid` is what actually excludes them.
        safe = labels.clamp(0, self.num_classes - 1)
        y = nn.functional.one_hot(safe, num_classes=self.num_classes)
        y = y.permute(0, 2, 1).to(torch.float32)

        return y * valid.unsqueeze(1), valid

    def _probabilities(self, logits: Tensor) -> Tensor:
        """
        (B, C, P) softmax over classes.

        float32 regardless of autocast: these feed a centroid estimate that
        is accumulated over ~10^6 voxels, where fp16 summation drifts.
        """
        return torch.softmax(self._flatten_spatial(logits).float(), dim=1)

    # -------------------------------------------------------------------------
    # Confident pseudo-label selection (unlabeled centroid source)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _confident_assignments(self, probs: Tensor):
        """
        Hard pseudo-label assignments restricted to confident voxels.

        Selection is PER PREDICTED CLASS, not over the patch as a whole.
        A global top-k would be dominated by background: the leaflets are a
        couple of percent of a crop, so topk_frac of all voxels is tens of
        thousands of background voxels and possibly no leaflet voxel at
        all, leaving the foreground centroids estimated from nothing.

        Returns (y, keep) with y (B, C, P) one-hot float and keep (B, P) bool.
        """

        confidence, predicted = probs.max(dim=1)

        y = nn.functional.one_hot(predicted, num_classes=self.num_classes)
        y = y.permute(0, 2, 1).to(torch.float32)

        flat_keep = torch.zeros(confidence.numel(), dtype=torch.bool, device=confidence.device)

        # One sync for all class counts rather than one per class.
        counts = torch.bincount(predicted.reshape(-1), minlength=self.num_classes).tolist()

        flat_confidence = confidence.reshape(-1)
        flat_predicted = predicted.reshape(-1)

        for class_id, count in enumerate(counts):
            if count == 0:
                continue

            is_class = flat_predicted == class_id

            # -1 sorts below every real probability, so topk can only pick
            # voxels predicted as this class.
            scores = torch.where(is_class, flat_confidence, torch.full_like(flat_confidence, -1.0))

            k = min(count, max(self.min_keep, int(round(self.topk_frac * count))))

            selected = torch.topk(scores, k=k, largest=True).indices

            flat_keep[selected] = True

        keep = flat_keep.view_as(confidence)

        # Intersect with the absolute threshold, but only if that leaves
        # enough voxels to estimate anything -- early in training nothing
        # clears 0.9 and an empty centroid update is worse than a noisy one.
        thresholded = keep & (confidence >= self.conf_thr)

        if int(thresholded.sum()) >= self.min_keep:
            keep = thresholded

        return y * keep.unsqueeze(1), keep

    # -------------------------------------------------------------------------
    # Centroid estimation
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def update_centroids(self, streams):
        """
        Re-estimate mu from (embeddings_flat, assignments) pairs and EMA it in.

        streams: list of (x, y) with x (B, D, P) and y (B, C, P) already
        zeroed at excluded voxels, so no separate mask is needed here.

        Runs on the FULL voxel set, not the subsample the energy uses: this
        is under no_grad, so the einsum keeps nothing for backward and the
        extra accuracy in the class means is free. Only the energy term
        needs to be economical.

        Under DDP the numerator and denominator are all-reduced before the
        division, so every rank ends up with the SAME centroids. Estimating
        them per rank would make the loss a different function on each rank
        -- gradients would still sync, but they would be gradients of eight
        different objectives.
        """

        numerator = torch.zeros_like(self.mu)
        denominator = torch.zeros(self.num_classes, device=self.mu.device, dtype=self.mu.dtype)

        for x, y in streams:
            x = x.to(self.mu.dtype)
            y = y.to(self.mu.dtype)
            numerator += torch.einsum("bdp,bcp->cd", x, y)
            denominator += y.sum(dim=(0, 2))

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(numerator)
            torch.distributed.all_reduce(denominator)

        present = denominator > self.eps

        if not bool(present.any()):
            return

        estimate = numerator[present] / denominator[present].unsqueeze(1)

        rows = present.nonzero(as_tuple=True)[0]
        fresh = ~self.mu_initialized[rows]

        # First sight of a class: take the estimate outright. An EMA from
        # the zero vector would need ~100 steps at ema_weight=0.95 to get
        # anywhere near the true mean.
        blended = torch.where(
            fresh.unsqueeze(1),
            estimate,
            self.ema_weight * self.mu[rows] + (1.0 - self.ema_weight) * estimate,
        )

        self.mu[rows] = blended
        self.mu_initialized[rows] = True

    # -------------------------------------------------------------------------
    # Energy
    # -------------------------------------------------------------------------
    def _subsample_indices(self, num_voxels: int, device):
        """Shared voxel subset for the energy term, or None to use all of them."""

        if self.num_voxels is None or self.num_voxels >= num_voxels:
            return None

        return torch.randperm(num_voxels, device=device)[: self.num_voxels]

    def _energy(self, x: Tensor, g: Tensor, valid: Tensor) -> Tensor:
        """
        mean over valid voxels of sum_k g_k ||x - mu_k||^2.

        Expanded rather than computed from an explicit difference, so the
        largest intermediate is (B, C, P) with C=3 instead of (B, D, P)
        with D=128. The 2D supercm project's permute(...).reshape(-1, D)
        materialises a full (B*P, D) copy that is then held for backward --
        at a 112x96x128 patch that is ~350 MB per sample of avoidable
        activation memory.

        sum_k g_k == 1 (g is a softmax), which is what lets the ||x||^2
        term come out of the class sum as a single (B, P) map.

        The result is divided by embedding_dim, so it is a mean squared
        distance PER DIMENSION rather than a raw squared distance in
        128-space. That is only a reparameterisation of the loss weight,
        but it is the difference between a weight that means something and
        one that does not: unscaled, this energy starts around D (=128) at
        initialisation, so the 0.1-ish auxiliary weight that the
        ssl_sweep_bnd_weakstrong grid found workable against CompoundLoss
        (itself O(1)) would have put the clustering term at ~14x the
        supervised one. Dividing by D makes the weight independent of
        embedding_dim and directly comparable to that sweep's lambda scale.

        normalize=True is exempt: spherical k-means already yields
        2 - 2cos(x, mu) in [0, 4], so it needs no rescaling and keeps the
        same weight ballpark.
        """

        mu = self.mu.to(x.dtype)

        if self.normalize:
            x = nn.functional.normalize(x, dim=1)
            mu = nn.functional.normalize(mu, dim=1)

        g = g.to(x.dtype)

        x_squared = x.pow(2).sum(dim=1)
        x_dot_mu = torch.einsum("bdp,cd->bcp", x, mu)
        mu_squared = mu.pow(2).sum(dim=1).view(1, -1, 1)

        per_voxel = x_squared - 2.0 * (g * x_dot_mu).sum(dim=1) + (g * mu_squared).sum(dim=1)

        if not self.normalize:
            per_voxel = per_voxel / float(self.embedding_dim)

        weights = valid.to(per_voxel.dtype)

        return (per_voxel * weights).sum() / weights.sum().clamp_min(1.0)

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(self, embeddings: Tensor, logits: Tensor, target: Tensor, u_embeddings=None, u_logits=None):
        """
        Returns (energy, stats).

        energy is UNWEIGHTED -- the caller scales it by weight_at(step).
        stats holds detached scalars for logging.

        The centroid update happens here, before the energy, so the energy
        is measured against centroids that have seen this batch. That
        ordering matches the 2D supercm project and matters most on the
        very first steps, where an un-updated mu is still the zero vector.
        """

        if self.needs_unlabeled and (u_embeddings is None or u_logits is None):
            raise ValueError(f"cm_mode={self.cm_mode!r} needs the unlabeled stream, but none was passed.")

        x_labeled = self._flatten_spatial(embeddings)
        g_labeled = self._probabilities(logits)
        y_labeled, valid_labeled = self._gt_assignments(target)

        streams = []
        stats = {}

        if self.cm_mode in ("l", "l+u"):
            streams.append((x_labeled, y_labeled))

        x_unlabeled = g_unlabeled = None

        if u_embeddings is not None and u_logits is not None:
            x_unlabeled = self._flatten_spatial(u_embeddings)
            g_unlabeled = self._probabilities(u_logits)

            if self.cm_mode in ("u", "l+u"):
                y_unlabeled, keep = self._confident_assignments(g_unlabeled)
                streams.append((x_unlabeled, y_unlabeled))
                stats["cm_confident_frac"] = keep.to(torch.float32).mean().detach()

        self.update_centroids(streams)

        # The energy is applied to every stream that was forwarded, not
        # just the ones feeding the centroids: an unlabeled crop's voxels
        # are exactly the ones with no supervision, so pulling them toward
        # a cluster is the whole point. With cm_mode="l" there is no
        # unlabeled forward at all and this reduces to the labeled crops.
        energy_terms = [(x_labeled, g_labeled, valid_labeled)]

        if x_unlabeled is not None:
            energy_terms.append((x_unlabeled, g_unlabeled, torch.ones_like(g_unlabeled[:, 0], dtype=torch.bool)))

        total = None

        for x, g, valid in energy_terms:
            indices = self._subsample_indices(x.shape[2], x.device)

            if indices is not None:
                x = x.index_select(2, indices)
                g = g.index_select(2, indices)
                valid = valid.index_select(1, indices)

            term = self._energy(x, g, valid)
            total = term if total is None else total + term

        total = _clip_loss(total / float(len(energy_terms)))

        stats["cm_loss"] = total.detach()

        return total, stats
