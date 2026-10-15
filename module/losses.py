"""
Compound loss for the supervised trainer: Dice (MemoryEfficientSoftDiceLoss,
do_bg=False) + a foreground-weighted CE (_foreground_weighted_ce, not
nnU-Net's own DC_and_CE_loss/RobustCrossEntropyLoss, which averages CE per
voxel with no class weighting -- see _foreground_weighted_ce's docstring
for why that's worth overriding for a task like CT where background is
~98% of a crop).

The Tversky / Focal / Focal-Tversky region+pixel variants tried across
the phase 1-3 experiment branches didn't outperform plain Dice+CE, so
this file doesn't reimplement that family or expose a configurable
loss_type -- CompoundLoss always uses this Dice+CE combination.

A BoundaryLoss (Kervadec et al., 2019) third term lived here as an
optional convex-combination add-on, ramped over epochs. Removed: the
ssl_sweep_bnd_weakstrong grid put it at +0.5 Dice at best, inside seed
noise on a 30-case val split, while costing ~58% more wall-clock per epoch
(~38 vs ~24 min) because it ran scipy's distance transform on CPU per
sample per step.

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

import torch
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

    """

    def __init__(
        self,
        batch_dice: bool,
        ddp: bool,
        ignore_label=None,
        foreground_weight: float = 1.0,
    ):
        super().__init__()

        self.ignore_label = ignore_label
        self.foreground_weight = foreground_weight

        self.dc = MemoryEfficientSoftDiceLoss(
            apply_nonlin=softmax_helper_dim1, batch_dice=batch_dice, smooth=1e-5, do_bg=False, ddp=ddp
        )

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

        return _clip_loss(self._dice_ce(net_output, target))


# =============================================================================
# SuperCM-style clustering regularizer
# =============================================================================


class ClusteringCMLoss(nn.Module):
    """
    SuperCM clustering regularizer on the per-voxel embeddings.

    Each class gets a centroid mu_k, re-estimated in closed form every step
    as the assignment-weighted mean of the embeddings, EMA-blended, held in
    a buffer -- never learned by gradient. The energy scores each voxel
    against those centroids:

        attract      sum_k g_k ||x - mu_k||^2     g = softmax(Wx + b)
        contrastive  -sum_k y_k log g_hat_k       g_hat = softmax(-d^2/tau)

    cm_mode says where the assignments y come from: "l" ground truth on
    labeled crops, "u" confident pseudo-labels on unlabeled ones, "l+u"
    both. Only u/l+u need an unlabeled forward pass.

    No cross-entropy here -- the LightningModule owns
    DeepSupervisionWrapper(CompoundLoss) and forms
    total = sup_loss + weight_at(step) * energy.

    Shapes: embeddings (B, D, *spatial), logits (B, C, *spatial), target
    (B, 1, *spatial). Caller passes the highest-resolution stage only.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        cm_mode: str = "l",
        form: str = "contrastive",
        tau: float = 1.0,
        weight: float = 0.1,
        start_epoch: float = 5.0,
        ema_weight: float = 0.95,
        conf_thr: float = 0.9,
        topk_frac: float = 0.15,
        min_keep: int = 64,
        use_soft_pseudo: bool = False,
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

        if form not in ("contrastive", "attract"):
            raise ValueError(f"form must be 'contrastive' or 'attract', got {form!r}")

        self.form = form
        self.tau = float(tau)
        self.weight = float(weight)
        self.start_epoch = float(start_epoch)
        self.ema_weight = float(ema_weight)
        self.conf_thr = float(conf_thr)
        self.topk_frac = float(topk_frac)
        self.min_keep = int(min_keep)
        self.use_soft_pseudo = bool(use_soft_pseudo)
        self.normalize = bool(normalize)
        self.ignore_label = ignore_label
        self.eps = float(eps)

        # Buffer, not a parameter: closed-form estimated, never optimised.
        # Registered so it survives a checkpoint restore.
        self.register_buffer("mu", torch.zeros(self.num_classes, self.embedding_dim))

        # Per-class: a class missing from the first batch must not be
        # marked initialised at the zero vector and then EMA-crawl up.
        self.register_buffer("mu_initialized", torch.zeros(self.num_classes, dtype=torch.bool))

    # -------------------------------------------------------------------------
    # Scheduling
    # -------------------------------------------------------------------------
    @property
    def needs_unlabeled(self) -> bool:
        """True when cm_mode sources centroids from the unlabeled stream."""
        return self.cm_mode in ("u", "l+u")

    def weight_at(self, epoch: float) -> float:
        """
        Off until start_epoch, then constant. A step, not a ramp, so any
        difference against the cm_weight=0 arm is attributable to the term
        rather than to the schedule.

        epoch is fractional (global_step / batches per epoch). Centroids are
        still estimated during the delay, so mu is warm when it switches on.
        """

        return self.weight if float(epoch) >= self.start_epoch else 0.0

    # -------------------------------------------------------------------------
    # Flattening helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _flatten_spatial(tensor: Tensor) -> Tensor:
        """(B, C, *spatial) -> (B, C, P). A view when the input is contiguous."""
        return tensor.flatten(2)

    def _valid_mask(self, labels: Tensor) -> Tensor:
        """
        (B, P) bool of voxels that may contribute. Only a configured
        ignore_label excludes anything -- nnU-Net maps -1 padding to 0.
        """

        if self.ignore_label is None:
            return torch.ones_like(labels, dtype=torch.bool)

        return labels != int(self.ignore_label)

    def _gt_assignments(self, target: Tensor):
        """
        Ground-truth one-hot from the integer label map -> (y, valid).

        Expand against num_classes, never the target's own channel dim: an
        nnU-Net target is (B, 1, *spatial), so the latter would read class
        id 2 as a probability of 2.0.
        """

        labels = target[:, 0].reshape(target.shape[0], -1).long()
        valid = self._valid_mask(labels)

        # Clamp only so one_hot cannot fault; `valid` does the excluding.
        safe = labels.clamp(0, self.num_classes - 1)
        y = nn.functional.one_hot(safe, num_classes=self.num_classes)
        y = y.permute(0, 2, 1).to(torch.float32)

        return y * valid.unsqueeze(1), valid

    def _probabilities(self, logits: Tensor) -> Tensor:
        """
        (B, C, P) softmax over classes, float32 regardless of autocast --
        it feeds a sum over ~10^6 voxels, where fp16 drifts.
        """
        return torch.softmax(self._flatten_spatial(logits).float(), dim=1)

    # -------------------------------------------------------------------------
    # Confident pseudo-label selection (unlabeled centroid source)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _confident_assignments(self, probs: Tensor):
        """
        Pseudo-labels restricted to confident voxels -> (y, keep).

        Top-k is PER PREDICTED CLASS, not over the patch: the leaflets are a
        couple of percent of a crop, so a global top-k would be all
        background and the foreground centroids would come from nothing.

        use_soft_pseudo=True uses raw probabilities instead of argmax
        one-hot. Degenerate early on, where p ~ 1/C everywhere drives all
        centroids onto the global mean -- do not pair it with a loosened
        conf_thr.

        Zeroed outside the kept set, so excluded voxels reach neither the
        numerator nor the denominator of the centroid estimate.
        """

        confidence, predicted = probs.max(dim=1)

        if self.use_soft_pseudo:
            y = probs
        else:
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
        Re-estimate mu from (x, y) streams and EMA it in. Under no_grad, so
        the einsum keeps nothing for backward.

        Numerator and denominator are all-reduced BEFORE the division, so
        every rank gets the same mu. Dividing per rank would make the loss a
        different function on each one.
        """

        numerator = torch.zeros_like(self.mu)
        denominator = torch.zeros(self.num_classes, device=self.mu.device, dtype=self.mu.dtype)

        for x, y in streams:
            # Autocast OFF, fp32 forced. A plain .float() is NOT enough:
            # torch.einsum is on autocast's fp16 list, so autocast casts both
            # operands straight back and accumulates in fp16. This reduces
            # over 1.38M voxels against an fp16 ceiling of 65504, which is
            # how mu became inf and every gradient NaN.
            with torch.autocast(device_type=x.device.type, enabled=False):
                x32 = x.float()
                y32 = y.float()
                numerator += torch.einsum("bdp,bcp->cd", x32, y32)
                denominator += y32.sum(dim=(0, 2))

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(numerator)
            torch.distributed.all_reduce(denominator)

        present = denominator > self.eps

        if not bool(present.any()):
            return

        estimate = numerator[present] / denominator[present].unsqueeze(1)

        # Crash rather than poison the run: a non-finite mu clips the
        # forward to a merely-large number while the backward carries NaN
        # into every parameter, so training silently stops.
        if not torch.isfinite(estimate).all():
            raise RuntimeError(
                "ClusteringCMLoss: non-finite centroid estimate. The reduction over voxels "
                "overflowed -- check that update_centroids still runs with autocast disabled "
                "and fp32 operands (torch.einsum is on autocast's fp16 list and will silently "
                "override an explicit .float() cast)."
            )

        rows = present.nonzero(as_tuple=True)[0]
        fresh = ~self.mu_initialized[rows]

        # First sight of a class: take the estimate outright -- an EMA from
        # zero needs ~100 steps at 0.95 to approach the true mean.
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
    def _energy(self, x: Tensor, g: Tensor, y: Tensor, valid: Tensor) -> Tensor:
        """
        Squared distance to every centroid, turned into a loss two ways.

        contrastive (default)   -log softmax_k(-||x - mu_k||^2 / tau)[y]

            Soft-target cross-entropy over the negative distances: "would a
            nearest-centroid classifier get this voxel right?". Attraction
            and repulsion both fall out, since the true class's distance has
            to be smallest RELATIVE to the others. Bounded in [0, log C],
            and log C is exactly the collapse value -- parallel centroids
            make every distance equal and the posterior uniform. Collapse is
            the worst state available rather than the best.

        attract (old behaviour, kept so earlier sweeps reproduce)

            sum_k g_k ||x - mu_k||^2

            Attraction only, blind to between-class separation: measured at
            centroid separations of 120/90/60/20/5/1 deg it is flat to three
            decimals with its MINIMUM at 5 deg. Sweep 7yu5e5o5 duly
            collapsed 113 -> 12 deg, Dice 0.002-0.021 worse than control.

        Every voxel contributes -- 2.75M per step at batch 2. The distance
        matrix is (B, C, P), ~33 MB; x.pow(2) is transiently (B, D, P) but
        freed by the sum.

        autocast OFF and fp32 throughout: leaning on autocast's per-op policy
        is what broke the centroid einsum, and _clip_loss cannot clamp a Half
        tensor to +/-1e6 at all.
        """

        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            g = g.float()
            y = y.float()
            mu = self.mu.float()

            if self.normalize:
                x = nn.functional.normalize(x, dim=1)
                mu = nn.functional.normalize(mu, dim=1)

            x_squared = x.pow(2).sum(dim=1).unsqueeze(1)
            x_dot_mu = torch.einsum("bdp,cd->bcp", x, mu)
            mu_squared = mu.pow(2).sum(dim=1).view(1, -1, 1)

            # (B, C, P): squared distance from every voxel to every centroid
            dist_sq = (x_squared - 2.0 * x_dot_mu + mu_squared).clamp_min(0.0)

            if self.form == "contrastive":
                log_posterior = nn.functional.log_softmax(-dist_sq / self.tau, dim=1)
                per_voxel = -(y * log_posterior).sum(dim=1)
            else:
                per_voxel = (g * dist_sq).sum(dim=1)

                if not self.normalize:
                    per_voxel = per_voxel / float(self.embedding_dim)

            weights = valid.to(per_voxel.dtype)

            return (per_voxel * weights).sum() / weights.sum().clamp_min(1.0)

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(self, embeddings: Tensor, logits: Tensor, target: Tensor, u_embeddings=None, u_logits=None):
        """
        Returns (energy, stats). energy is UNWEIGHTED -- the caller scales
        it by weight_at(step).

        Centroids are updated here BEFORE the energy, so it is measured
        against mu that has seen this batch. Matters most at step 0, where
        an un-updated mu is still the zero vector.
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

        x_unlabeled = g_unlabeled = y_unlabeled = None

        if u_embeddings is not None and u_logits is not None:
            x_unlabeled = self._flatten_spatial(u_embeddings)
            g_unlabeled = self._probabilities(u_logits)

            if self.cm_mode in ("u", "l+u"):
                y_unlabeled, keep = self._confident_assignments(g_unlabeled)
                streams.append((x_unlabeled, y_unlabeled))
                stats["cm_confident_frac"] = keep.to(torch.float32).mean().detach()

        self.update_centroids(streams)

        # Energy covers every stream that was forwarded, not only those
        # feeding the centroids. y is the contrastive target and the same
        # assignment mu came from; the attract form ignores it, using g.
        energy_terms = [(x_labeled, g_labeled, y_labeled, valid_labeled)]

        if x_unlabeled is not None:
            # No pseudo-labels sourced -> fall back to the soft posterior
            # so the contrastive form still has a target.
            ones = torch.ones_like(g_unlabeled[:, 0], dtype=torch.bool)
            y_u = y_unlabeled if y_unlabeled is not None else g_unlabeled
            energy_terms.append((x_unlabeled, g_unlabeled, y_u, ones))

        total = None

        for x, g, y, valid in energy_terms:
            term = self._energy(x, g, y, valid)
            total = term if total is None else total + term

        total = _clip_loss(total / float(len(energy_terms)))

        stats["cm_loss"] = total.detach()

        return total, stats
