"""
Per-epoch t-SNE of the per-voxel embedding space.

This branch trains Dice+CE PLUS the SuperCM clustering term, and the plot is
here to show what that term actually does to the 128-d per-voxel embedding.

The companion branch fewshot_tee_inspect runs the same plot with the
clustering term removed. That measurement is already in: with Dice+CE alone
the centroids sit 81-146 degrees apart, within-class spread is 18-35
degrees, and a nearest-centroid rule matches the learned linear head to
within 1 percent -- i.e. supervision alone already produces everything the
clustering energy is designed to produce. So the question for THIS branch is
not whether the term creates structure, but whether it damages structure
that was already there. Every centroid/prototype method assumes it
does. Nothing in the segmentation loss asks for it: Dice+CE constrains
W*x + b, not x, so the embedding is free to be any full-rank
reparameterization of a 3-logit space (see EmbeddingAdapter's docstring).

So the plot is the experiment. If the classes already separate, prototypes
are reading real structure. If they do not, a clustering loss is not
"regularizing" an embedding -- it is manufacturing the thing it measures,
which is exactly how the attract-form energy collapsed every centroid onto
one direction.

The grid is 1 LABELED case and 8 HELD-OUT ones, because those answer two
different questions. Structure on the labeled case could be memorisation
of the one image the loss has seen. Structure on cases the network has
never received a gradient from is the thing a semi-supervised method would
actually exploit -- and it is the premise a global centroid rests on.

The held-out panels can be coloured by class at all because MVSeg2023
ships fully labeled. A case is "unlabeled" here only in the sense that the
datamodule routes it away from the loss; its segmentation is still on
disk. So the labels are visible to this plot and invisible to training,
which is exactly the condition that makes the comparison measurable.

Read the panels across epochs, not within one: the absolute t-SNE layout is
meaningless (the algorithm has no preferred orientation or scale, and is
re-fit from scratch each time), but whether the classes pull apart, stay
mixed, or merge is real.
"""

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Agg before pyplot: compute nodes have no display, and the default
# interactive backend would fail at import rather than at plot time.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
from sklearn.manifold import TSNE  # noqa: E402

from .adapter import embeddings_enabled  # noqa: E402


# Fixed per-class colours so a panel means the same thing at epoch 0 and
# epoch 199. Index = class id, matching dataset_json["labels"].
CLASS_COLOURS = ["#b0b0b0", "#0072b2", "#d55e00", "#009e73", "#cc79a7"]


def _foreground_patch(data, seg, patch_size):
    """
    One deterministic patch per case, centred on the segmentation.

    Deterministic on purpose: a random crop would change the voxels under
    comparison every epoch, so a shift in the plot could be the embedding
    moving or just a different piece of anatomy. Centred on the foreground
    centre of mass rather than the volume centre because the leaflets are
    thin and off-centre -- a volume-centred crop can miss them entirely,
    and a panel of pure background says nothing.

    `data` and `seg` are blosc2 arrays. The seg is single-channel and small,
    so it is read whole to locate the foreground; the data is sliced, so
    only the patch itself is decompressed.
    """

    seg_full = np.asarray(seg[0])
    shape = seg_full.shape

    foreground = np.argwhere(seg_full > 0)

    if len(foreground):
        centre = foreground.mean(axis=0).round().astype(int)
    else:
        centre = np.asarray(shape) // 2

    starts, stops, pads = [], [], []

    for axis, size in enumerate(patch_size):
        low = int(centre[axis]) - size // 2
        high = low + size

        pads.append((max(0, -low), max(0, high - shape[axis])))
        starts.append(max(0, low))
        stops.append(min(shape[axis], high))

    crop = tuple(slice(a, b) for a, b in zip(starts, stops))

    patch_data = np.asarray(data[(slice(None), *crop)], dtype=np.float32)
    patch_seg = seg_full[crop].astype(np.int16)

    # Zero-pad a volume smaller than the patch on any axis. Zero is the
    # right fill for both: the preprocessed intensities are already
    # z-scored (so 0 is the mean), and 0 is the background label.
    if any(pad != (0, 0) for pad in pads):
        patch_data = np.pad(patch_data, [(0, 0), *pads], constant_values=0.0)
        patch_seg = np.pad(patch_seg, pads, constant_values=0)

    return patch_data, patch_seg


def _stratified_sample(embeddings, labels, num_classes, max_per_class, rng):
    """
    Equal voxels per present class, instead of a uniform draw.

    A uniform sample of a TEE patch is ~99% background, so the leaflets
    would contribute a handful of points and the plot would show nothing
    about them. Equal-per-class distorts the DENSITY -- do not read cluster
    size off these panels -- but it is the only way the minority classes are
    visible at all.
    """

    keep_embeddings, keep_labels = [], []

    for class_id in range(num_classes):
        index = np.flatnonzero(labels == class_id)

        if index.size == 0:
            continue

        if index.size > max_per_class:
            index = rng.choice(index, size=max_per_class, replace=False)

        keep_embeddings.append(embeddings[index])
        keep_labels.append(labels[index])

    if not keep_embeddings:
        return None, None

    return np.concatenate(keep_embeddings), np.concatenate(keep_labels)


def _fit_tsne(points, perplexity, metric, seed, max_iter):
    """
    2-D t-SNE of one point cloud, or None when there is too little to fit.

    perplexity must stay below the sample count or sklearn raises; it is
    clamped rather than erroring so a case with one thin leaflet still
    produces a panel.

    n_jobs=1 on purpose. This runs inside on_train_epoch_end, while the
    augmentation workers are still alive, and the measured gain from
    parallel neighbour search at this point count was nil (2.92 s vs 2.80 s
    at n=1500) -- so letting t-SNE grab cores would cost the data pipeline
    real throughput to buy nothing.

    max_iter is floored at 300. sklearn spends its first 250 iterations in
    EARLY EXAGGERATION, so a fit that stops at exactly 250 never runs a
    single normal-phase step and returns a degenerate straight line -- it
    looks like a catastrophically collapsed embedding and is purely an
    artefact. Verified against real embeddings: 250 is a line, while 500,
    1000 and 2000 are visually identical (3.1 / 6.9 / 14.2 s per panel),
    which is why the default is 500.
    """

    if points is None or len(points) < 10:
        return None

    tsne = TSNE(
        n_components=2,
        perplexity=float(min(perplexity, max(5.0, (len(points) - 1) / 3.0))),
        metric=metric,
        init="pca",
        learning_rate="auto",
        random_state=int(seed),
        max_iter=max(300, int(max_iter)),
        n_jobs=1,
    )

    return tsne.fit_transform(points.astype(np.float32))


def _panel(axis, coords, labels, title, class_names, is_labeled, centroid_coords=None):
    """
    One case's scatter. The labeled panel is framed so it cannot be confused
    with the held-out ones -- the whole figure is a comparison between the
    two kinds, so which is which has to be unmissable at a glance.
    """

    axis.set_title(
        ("LABELED  " if is_labeled else "held-out  ") + title,
        fontsize=9,
        fontweight="bold" if is_labeled else "normal",
        color="#005aa0" if is_labeled else "0.25",
    )
    axis.set_xticks([])
    axis.set_yticks([])

    for spine in axis.spines.values():
        spine.set_linewidth(2.2 if is_labeled else 0.8)
        spine.set_color("#005aa0" if is_labeled else "0.7")

    for class_id in np.unique(labels):
        mask = labels == class_id
        axis.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=2.0,
            alpha=0.55,
            linewidths=0,
            color=CLASS_COLOURS[int(class_id) % len(CLASS_COLOURS)],
            label=class_names.get(int(class_id), str(class_id)),
        )

    if centroid_coords is None:
        return

    # The loss's live mu, embedded jointly with this panel's voxels so its
    # position is comparable to them. This is the whole point of running the
    # plot on this branch: mu is what the energy pulls every voxel toward, so
    # seeing it land OUTSIDE its own class -- or seeing all three land on top
    # of each other -- is the collapse, directly rather than inferred.
    # Decreasing marker size by class so that three COINCIDENT centroids are
    # still countable -- equal-sized stars would hide two of them behind the
    # last one drawn, which is precisely the state worth seeing.
    for class_id, (cx, cy) in enumerate(centroid_coords):
        axis.scatter(
            cx, cy, marker="*", s=420 - 90 * class_id,
            color=CLASS_COLOURS[class_id % len(CLASS_COLOURS)],
            edgecolors="black", linewidths=1.1,
            zorder=5 + class_id,
        )


def centroid_angle(mu) -> float:
    """
    Mean pairwise angle between class centroids, measured at their own mean.

    Centred, not origin-referenced: x -> x + t (with b -> b - W t) leaves
    every logit identical but swings an origin-referenced angle freely. The
    attract arm read 12 deg at the origin and 113-129 deg centred, having
    simply moved the cloud from ||mean|| 2.0 to 15.8. The origin is not a
    property of the data; embedding_offset tracks that drift instead.

    mu is identical on every rank (update_centroids all-reduces before
    dividing), so no gather is needed.
    """

    mu = mu.detach().float()
    mu = mu - mu.mean(dim=0, keepdim=True)

    unit = nn.functional.normalize(mu, dim=1)
    cosines = (unit @ unit.T).clamp(-1.0, 1.0)
    pairs = torch.triu_indices(unit.shape[0], unit.shape[0], offset=1)

    return float(torch.rad2deg(torch.acos(cosines[pairs[0], pairs[1]])).mean())


def embedding_offset(mu) -> float:
    """
    ||mean(mu)|| -- the translation gauge coordinate itself.

    Pure gauge: it can move without changing a single logit. Watching it rise
    is what identifies an arm minimising the energy by drifting away from the
    origin rather than by separating the classes.
    """

    return float(mu.detach().float().mean(dim=0).norm())


class EmbeddingTSNE:
    """
    Collects stage-0 embeddings for a fixed set of cases and plots them.

    Rank 0 only, under no_grad, on the UNWRAPPED network. Going around the
    DDP wrapper is normally a bug -- it skips the gradient-sync hooks -- but
    this pass takes no gradients at all, so there is nothing to sync, and
    calling the wrapper on one rank while the others move on is what would
    actually hang.
    """

    def __init__(
        self,
        output_folder,
        num_labeled_cases=1,
        num_unlabeled_cases=8,
        every_n_epochs=1,
        max_per_class=500,
        perplexity=30.0,
        max_iter=500,
        metric="cosine",
        seed=0,
    ):
        self.output_folder = Path(output_folder) / "embedding_tsne"
        self.num_labeled_cases = int(num_labeled_cases)
        self.num_unlabeled_cases = int(num_unlabeled_cases)
        self.every_n_epochs = int(every_n_epochs)
        self.max_per_class = int(max_per_class)
        self.perplexity = float(perplexity)
        self.max_iter = int(max_iter)
        self.metric = str(metric)
        self.seed = int(seed)

        self._cases = None

    def due(self, epoch, is_last: bool = False):
        """
        The final epoch always plots; every_n_epochs=0 means only then.

        t-SNE is the expensive part of this diagnostic (minutes, against
        ~1.5 s for the forwards), so by default it runs once on the finished
        model rather than every epoch.
        """

        if is_last:
            return True

        if self.every_n_epochs <= 0:
            return False

        return (int(epoch) + 1) % self.every_n_epochs == 0

    def _select_cases(self, datamodule):
        """
        A fixed panel list of (dataset, case_id, is_labeled), same every epoch.

        The point of the mix: the labeled panel shows what the embedding
        looks like where the loss HAS acted, and the unlabeled panels show
        what it looks like on cases the network has never received a
        gradient from. A prototype or pseudo-label method needs class
        structure on the second kind -- structure on the first could just be
        memorisation.

        The unlabeled panels can be coloured by true class because MVSeg2023
        ships fully labeled: a case is "unlabeled" here only because the
        datamodule routes it to a loader whose targets never reach a loss
        (and on this branch, a loader nothing reads at all). The seg files
        are still on disk, so the labels are available to the plot while
        being invisible to training. That is what makes this measurable.

        Both pools are RANK-LOCAL and already sorted by nnUNetBaseDataset,
        so this is deterministic without re-sorting.
        """

        if self._cases is not None:
            return self._cases

        labeled = list(datamodule.dataset_train_labeled.identifiers)
        unlabeled = list(datamodule.dataset_train_unlabeled.identifiers)

        # At num_labeled == len(train_all) the datamodule recycles a labeled
        # case into the empty unlabeled pool, so "unlabeled" panels would be
        # relabelled copies of the labeled one. Say so rather than drawing a
        # grid that silently means nothing.
        overlap = sorted(set(unlabeled) & set(labeled))

        if overlap:
            print(
                f"[embedding_tsne] WARNING: {len(overlap)} of {len(unlabeled)} 'unlabeled' "
                f"cases are also in the labeled pool ({overlap[:3]}...). At "
                "num_labeled == len(train_all) the unlabeled pool is a recycled labeled "
                "case, so the unlabeled panels do NOT show held-out data. Lower "
                "datamodule.num_labeled to make this grid meaningful.",
                flush=True,
            )

        cases = [
            (datamodule.dataset_train_labeled, case_id, True)
            for case_id in labeled[: self.num_labeled_cases]
        ]
        cases += [
            (datamodule.dataset_train_unlabeled, case_id, False)
            for case_id in unlabeled[: self.num_unlabeled_cases]
        ]

        self._cases = cases

        return cases

    @torch.no_grad()
    def _embeddings_for_case(self, network, dataset, case_id, patch_size, device):
        data, seg = _foreground_patch(*dataset.load_case(case_id)[:2], patch_size=patch_size)

        tensor = torch.from_numpy(data).unsqueeze(0).to(device=device, dtype=torch.float32)

        previous = network.decoder.deep_supervision

        # Deep supervision off: only stage 0 is plotted, so the four coarser
        # embedding maps would be pure compute. With it off the adapter
        # returns bare tensors rather than per-stage lists.
        #
        # return_embeddings goes through embeddings_enabled rather than being
        # assigned directly, per EmbeddingAdapter's docstring -- it restores
        # the previous value even if the forward raises, so a failure here
        # cannot leave the predictor receiving tuples.
        network.decoder.deep_supervision = False

        try:
            with embeddings_enabled(network):
                embeddings, _ = network(tensor)
        finally:
            network.decoder.deep_supervision = previous

        flat = embeddings[0].flatten(1).T.float().cpu().numpy()

        return flat, seg.reshape(-1)

    def plot(self, module, epoch):
        """
        Write one figure for this epoch, and log it if a logger is attached.

        Returns the path written, or None when nothing could be plotted.
        """

        datamodule = module.trainer.datamodule

        if datamodule.dataset_train_labeled is None or datamodule.dataset_train_unlabeled is None:
            return None

        network = module.network.module if hasattr(module.network, "module") else module.network
        was_training = network.training
        network.eval()

        patch_size = tuple(int(v) for v in datamodule.cm.patch_size)
        class_names = {int(v): k for k, v in module.dataset_json["labels"].items()}
        # nnunet.lm is the nnU-Net LabelManager; num_segmentation_heads is
        # what build_network() itself uses for the classifier width, so the
        # two cannot drift. NNUnetSetup has no `num_classes` attribute.
        num_classes = int(module.nnunet.lm.num_segmentation_heads)
        rng = np.random.default_rng(self.seed)

        # Only centroids the loss has actually estimated. An uninitialised row
        # is still the zero vector, which would plot at a meaningless place.
        centroids = module.cm_loss.mu.detach().float().cpu().numpy()
        initialized = module.cm_loss.mu_initialized.detach().cpu().numpy()

        if not initialized.all():
            centroids = None

        per_case = []

        try:
            for dataset, case_id, is_labeled in self._select_cases(datamodule):
                flat, labels = self._embeddings_for_case(
                    network, dataset, case_id, patch_size, module.device
                )

                points, point_labels = _stratified_sample(
                    flat, labels, num_classes, self.max_per_class, rng
                )

                per_case.append((case_id, points, point_labels, is_labeled))
        finally:
            if was_training:
                network.train()

        if not per_case:
            return None

        # Three columns, so the default 1 labeled + 8 held-out lands as a
        # 3x3 grid with the labeled case in the top-left corner.
        panels = len(per_case)
        columns = min(3, panels)
        rows = math.ceil(panels / columns)

        figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 4.0 * rows), squeeze=False)
        flat_axes = axes.ravel()

        for index, (case_id, points, point_labels, is_labeled) in enumerate(per_case):
            # Centroids are appended to the cloud so t-SNE places them in the
            # SAME embedding as the voxels. Fitting them separately, or
            # projecting them afterwards, would put them on an unrelated map --
            # t-SNE has no out-of-sample transform.
            stacked = points if centroids is None else np.vstack([points, centroids])
            coords = _fit_tsne(stacked, self.perplexity, self.metric, self.seed, self.max_iter)

            if coords is None:
                flat_axes[index].text(
                    0.5, 0.5, "too few voxels", ha="center", va="center", fontsize=8, color="0.5"
                )
                flat_axes[index].set_title(case_id, fontsize=9)
                flat_axes[index].set_xticks([])
                flat_axes[index].set_yticks([])
                continue

            centroid_coords = None if centroids is None else coords[len(points):]
            _panel(
                flat_axes[index], coords[: len(points)], point_labels, case_id,
                class_names, is_labeled, centroid_coords,
            )

        for axis in flat_axes[panels:]:
            axis.axis("off")

        handles, names = [], []

        for axis in flat_axes[:panels]:
            handles, names = axis.get_legend_handles_labels()

            if handles:
                break

        if handles:
            figure.legend(
                handles,
                names,
                loc="lower center",
                ncol=len(handles),
                markerscale=6.0,
                fontsize=9,
                frameon=False,
            )

        labeled_count = sum(1 for case in per_case if case[3])

        figure.suptitle(
            f"stage-0 embedding t-SNE ({self.metric}) | epoch {int(epoch)} | "
            f"{labeled_count} labeled + {panels - labeled_count} held-out cases | "
            f"{self.max_per_class} voxels/class | "
            f"centroid angle {centroid_angle(module.cm_loss.mu):.1f} deg, "
            f"offset {embedding_offset(module.cm_loss.mu):.2f}\n"
            "held-out panels coloured by ground truth the network never trained on; "
            "stars are the clustering loss's live centroids",
            fontsize=11,
        )
        figure.tight_layout(rect=(0, 0.06, 1, 0.97))

        self.output_folder.mkdir(parents=True, exist_ok=True)
        path = self.output_folder / f"epoch_{int(epoch):04d}.png"
        figure.savefig(path, dpi=110)

        logger = getattr(module, "logger", None)

        if logger is not None and not isinstance(logger, bool) and hasattr(logger, "log_image"):
            logger.log_image(key="embedding_tsne", images=[str(path)], step=module.global_step)

        plt.close(figure)

        return path
