#!/usr/bin/env python3
"""Task3 baseline semi-supervised training entrypoint."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import time
from pathlib import Path
from typing import Dict, Sequence

import torch
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import (
    LabeledDataset,
    UnlabeledPairDataset,
    build_fg_balanced_weights,
    discover_samples,
    discover_unlabeled_images,
    sample_has_foreground,
    split_train_val_by_video,
)
from model_factory import get_loss_fn, get_model
from utils import (
    MetricRefs,
    ensure_dir,
    get_device,
    metric_quality_weighted,
    save_json,
    seed_everything,
    setup_logger,
)

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
TASK3_ROOT = REPO_ROOT / "task3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task3 baseline semi-supervised training")

    parser.add_argument("--labeled-root", type=str, default=str(REPO_ROOT / "data" / "t3_vid" / "train"))
    parser.add_argument("--external-val-root", type=str, default=str(THIS_DIR / "data" / "labeled" / "val_external"))
    parser.add_argument("--use-external-val", action="store_true", default=False)
    parser.add_argument("--no-use-external-val", action="store_false", dest="use_external_val")
    parser.add_argument("--unlabeled-root", type=str, default=str(TASK3_ROOT / "未标记素材-图片" / "images"))
    parser.add_argument("--output-dir", type=str, default=str(THIS_DIR / "runs" / "semi_baseline_default"))

    parser.add_argument("--arch", type=str, default="unetplusplus", choices=["unet", "unetplusplus", "fpn", "deeplabv3plus"])
    parser.add_argument("--encoder-name", type=str, default="efficientnet-b4")
    parser.add_argument("--encoder-weights", type=str, default="none", choices=["none", "imagenet"])
    parser.add_argument("--target-label", type=int, default=10)
    parser.add_argument("--image-size", type=int, nargs=2, default=[448, 800], help="H W")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--unlabeled-batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-video-count", type=int, default=2)
    parser.add_argument("--val-only-fg", action="store_true", default=True)
    parser.add_argument("--no-val-only-fg", action="store_false", dest="val_only_fg")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Debug only; 0 means all")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Debug only; 0 means all")
    parser.add_argument("--max-unlabeled-samples", type=int, default=0, help="Debug only; 0 means all")

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    parser.add_argument("--loss-type", type=str, default="dice_focal", choices=["dice_bce", "dice_focal"])
    parser.add_argument("--dice-loss-weight", type=float, default=0.7)
    parser.add_argument("--bce-loss-weight", type=float, default=0.3)
    parser.add_argument("--focal-loss-weight", type=float, default=0.3)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--pos-weight", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--score-dsc-weight", type=float, default=0.6)
    parser.add_argument("--score-hd-weight", type=float, default=0.2)
    parser.add_argument("--score-asd-weight", type=float, default=0.2)
    parser.add_argument("--score-hd-ref", type=float, default=20.0)
    parser.add_argument("--score-asd-ref", type=float, default=3.0)

    parser.add_argument("--semi-warmup-epochs", type=int, default=20)
    parser.add_argument("--unsup-weight", type=float, default=0.6)
    parser.add_argument("--unsup-ramp-epochs", type=int, default=30)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--pseudo-pos-thr", type=float, default=0.70)
    parser.add_argument("--pseudo-neg-thr", type=float, default=0.10)
    parser.add_argument("--pseudo-min-area", type=float, default=80.0)
    parser.add_argument("--pseudo-min-pos-ratio", type=float, default=0.0005)

    parser.add_argument(
        "--threshold-candidates",
        type=float,
        nargs="+",
        default=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
    )
    parser.add_argument("--val-tta", action="store_true", default=True)
    parser.add_argument("--no-val-tta", action="store_false", dest="val_tta")

    parser.add_argument("--use-imagenet-norm", action="store_true", default=True)
    parser.add_argument("--no-imagenet-norm", action="store_false", dest="use_imagenet_norm")
    parser.add_argument("--cache-masks", action="store_true", default=True)
    parser.add_argument("--no-cache-masks", action="store_false", dest="cache_masks")

    parser.add_argument("--use-fg-balanced-sampling", action="store_true", default=True)
    parser.add_argument("--no-fg-balanced-sampling", action="store_false", dest="use_fg_balanced_sampling")
    parser.add_argument("--fg-sampling-power", type=float, default=0.5)
    parser.add_argument("--fg-sampling-min-weight", type=float, default=0.5)
    parser.add_argument("--fg-sampling-max-weight", type=float, default=4.0)

    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--early-stop-patience", type=int, default=40)
    return parser.parse_args()


def cycle_next(loader, iterator):
    try:
        batch = next(iterator)
        return batch, iterator
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
        return batch, iterator


def dice_from_preds(
    preds: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-6,
    ignore_empty_gt: bool = False,
) -> float:
    inter = (preds * labels).sum(dim=(1, 2, 3))
    pred_sum = preds.sum(dim=(1, 2, 3))
    gt_sum = labels.sum(dim=(1, 2, 3))
    denom = pred_sum + gt_sum
    dice = (2.0 * inter + eps) / (denom + eps)
    if ignore_empty_gt:
        valid = gt_sum > 0
        if bool(valid.any()):
            dice = dice[valid]
        else:
            return 0.0
    return float(dice.mean().item())


def dice_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-6,
    ignore_empty_gt: bool = False,
) -> float:
    preds = (torch.sigmoid(logits) > 0.5).float()
    return dice_from_preds(preds, labels, eps=eps, ignore_empty_gt=ignore_empty_gt)


def _nanmean_to_float(x: torch.Tensor) -> float:
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return float("nan")
        return float(torch.nanmean(x).item())
    return float(x)


@torch.no_grad()
def predict_probs(model, images: torch.Tensor, use_amp: bool, use_tta: bool) -> torch.Tensor:
    device_type = images.device.type
    with torch.amp.autocast(device_type=device_type, enabled=use_amp):
        logits = model(images)
    probs = torch.sigmoid(logits)

    if not use_tta:
        return probs

    probs_sum = probs
    for dims in [(3,), (2,), (2, 3)]:
        x = torch.flip(images, dims=dims)
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            logits_f = model(x)
        probs_f = torch.sigmoid(logits_f)
        probs_sum = probs_sum + torch.flip(probs_f, dims=dims)
    return probs_sum / 4.0


@torch.no_grad()
def evaluate(
    model,
    loader,
    loss_fn,
    device: torch.device,
    use_amp: bool,
    threshold_candidates: Sequence[float],
    use_tta: bool,
    fixed_threshold: float | None = None,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    steps = 0
    probs_list = []
    labels_list = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)

        probs = predict_probs(model, images, use_amp=use_amp, use_tta=use_tta)
        loss_sum += float(loss.item())
        steps += 1
        probs_list.append(probs.float().cpu())
        labels_list.append(labels.float().cpu())

    all_probs = torch.cat(probs_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    if fixed_threshold is None:
        best_thr = float(threshold_candidates[0])
        best_dice = -1.0
        for thr in threshold_candidates:
            preds = (all_probs > float(thr)).float()
            d = dice_from_preds(preds, all_labels, ignore_empty_gt=True)
            if d > best_dice:
                best_dice = d
                best_thr = float(thr)
    else:
        best_thr = float(fixed_threshold)

    final_preds = (all_probs > best_thr).float()
    pred_pos_ratio = float(final_preds.mean().item())
    gt_pos_ratio = float(all_labels.mean().item())

    pred_non_empty = final_preds.flatten(1).sum(dim=1) > 0
    gt_non_empty = all_labels.flatten(1).sum(dim=1) > 0
    valid_dist_mask = pred_non_empty & gt_non_empty
    valid_dist_cases = int(valid_dist_mask.sum().item())

    if valid_dist_cases > 0:
        hd_metric = HausdorffDistanceMetric(
            include_background=True,
            distance_metric="euclidean",
            percentile=None,
            reduction="mean_batch",
        )
        asd_metric = SurfaceDistanceMetric(
            include_background=True,
            symmetric=True,
            reduction="mean_batch",
        )
        eval_bs = 8
        dist_preds = final_preds[valid_dist_mask]
        dist_labels = all_labels[valid_dist_mask]
        for i in range(0, dist_preds.shape[0], eval_bs):
            p = dist_preds[i : i + eval_bs]
            y = dist_labels[i : i + eval_bs]
            hd_metric(y_pred=p, y=y)
            asd_metric(y_pred=p, y=y)
        hd = _nanmean_to_float(hd_metric.aggregate())
        asd = _nanmean_to_float(asd_metric.aggregate())
        hd_metric.reset()
        asd_metric.reset()
    else:
        hd = float("nan")
        asd = float("nan")

    return {
        "val_loss": float(loss_sum / max(1, steps)),
        "val_dice": float(dice_from_preds(final_preds, all_labels, ignore_empty_gt=True)),
        "val_hd": hd,
        "val_asd": asd,
        "val_threshold": best_thr,
        "val_pred_pos_ratio": pred_pos_ratio,
        "val_gt_pos_ratio": gt_pos_ratio,
        "val_valid_dist_cases": valid_dist_cases,
    }


def update_ema(teacher, student, decay: float) -> None:
    with torch.no_grad():
        for t_param, s_param in zip(teacher.parameters(), student.parameters()):
            t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)
        for t_buf, s_buf in zip(teacher.buffers(), student.buffers()):
            t_buf.copy_(s_buf)


def compute_unsup_weight(epoch: int, semi_warmup_epochs: int, unsup_weight: float, ramp_epochs: int) -> float:
    if epoch <= semi_warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(unsup_weight)
    ratio = min(1.0, float(epoch - semi_warmup_epochs) / float(ramp_epochs))
    return float(unsup_weight) * ratio


def main() -> int:
    args = parse_args()
    seed_everything(int(args.seed))

    image_size = (int(args.image_size[0]), int(args.image_size[1]))
    if image_size[0] % 32 != 0 or image_size[1] % 32 != 0:
        raise ValueError(f"image_size must be divisible by 32, got {image_size}")

    out_dir = ensure_dir(args.output_dir)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    logger = setup_logger(out_dir, log_name="train.log")
    logger.info("Start baseline semi training")
    logger.info("Args: %s", vars(args))

    labeled_root = Path(args.labeled_root)
    external_val_root = Path(args.external_val_root)
    unlabeled_root = Path(args.unlabeled_root)
    if not labeled_root.exists():
        raise FileNotFoundError(f"labeled_root not found: {labeled_root}")

    device = get_device()
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    logger.info("Device=%s AMP=%s", device, use_amp)

    all_labeled = discover_samples(labeled_root)
    train_samples, val_samples, train_video_ids, val_video_ids = split_train_val_by_video(
        all_labeled,
        val_video_count=int(args.val_video_count),
        seed=int(args.seed),
    )
    val_samples_all = list(val_samples)
    if int(args.max_train_samples) > 0:
        train_samples = train_samples[: int(args.max_train_samples)]
    if int(args.max_val_samples) > 0:
        val_samples = val_samples[: int(args.max_val_samples)]
        val_samples_all = val_samples_all[: int(args.max_val_samples)]

    val_fg_before = len(val_samples)
    if bool(args.val_only_fg):
        val_samples = [
            s for s in val_samples if sample_has_foreground(s, target_label=int(args.target_label))
        ]
    val_fg_after = len(val_samples)
    if len(val_samples) == 0:
        raise RuntimeError(
            "Validation set is empty after foreground filtering. "
            "Try changing --seed / --val-video-count, or disable --val-only-fg."
        )

    external_val_samples = []
    external_val_samples_all = []
    if args.use_external_val and external_val_root.exists():
        try:
            external_val_samples = discover_samples(external_val_root)
            external_val_samples_all = list(external_val_samples)
        except Exception:
            external_val_samples = []
            external_val_samples_all = []
    ext_fg_before = len(external_val_samples)
    if bool(args.val_only_fg) and external_val_samples:
        external_val_samples = [
            s for s in external_val_samples if sample_has_foreground(s, target_label=int(args.target_label))
        ]
    ext_fg_after = len(external_val_samples)

    unlabeled_paths = discover_unlabeled_images(unlabeled_root)
    if int(args.max_unlabeled_samples) > 0:
        unlabeled_paths = unlabeled_paths[: int(args.max_unlabeled_samples)]

    train_ds = LabeledDataset(
        samples=train_samples,
        image_size=image_size,
        target_label=int(args.target_label),
        train=True,
        cache_masks=bool(args.cache_masks),
        use_imagenet_norm=bool(args.use_imagenet_norm),
        seed=int(args.seed),
    )
    val_ds = LabeledDataset(
        samples=val_samples,
        image_size=image_size,
        target_label=int(args.target_label),
        train=False,
        cache_masks=bool(args.cache_masks),
        use_imagenet_norm=bool(args.use_imagenet_norm),
        seed=int(args.seed),
    )
    ext_val_ds = None
    if external_val_samples:
        ext_val_ds = LabeledDataset(
            samples=external_val_samples,
            image_size=image_size,
            target_label=int(args.target_label),
            train=False,
            cache_masks=bool(args.cache_masks),
            use_imagenet_norm=bool(args.use_imagenet_norm),
            seed=int(args.seed),
        )

    train_sampler = None
    if args.use_fg_balanced_sampling and len(train_ds) > 0:
        weights = build_fg_balanced_weights(
            train_ds.sample_fg_ratio,
            power=float(args.fg_sampling_power),
            min_weight=float(args.fg_sampling_min_weight),
            max_weight=float(args.fg_sampling_max_weight),
        )
        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(int(args.seed)),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=int(args.num_workers) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=min(2, int(args.num_workers)),
        pin_memory=True,
        persistent_workers=int(args.num_workers) > 0,
    )

    ext_val_loader = None
    if ext_val_ds is not None and len(ext_val_ds) > 0:
        ext_val_loader = DataLoader(
            ext_val_ds,
            batch_size=max(1, int(args.batch_size)),
            shuffle=False,
            num_workers=min(2, int(args.num_workers)),
            pin_memory=True,
            persistent_workers=int(args.num_workers) > 0,
        )

    unl_loader = None
    if unlabeled_paths:
        unl_ds = UnlabeledPairDataset(
            image_paths=unlabeled_paths,
            image_size=image_size,
            use_imagenet_norm=bool(args.use_imagenet_norm),
            seed=int(args.seed),
        )
        unl_loader = DataLoader(
            unl_ds,
            batch_size=max(1, int(args.unlabeled_batch_size)),
            shuffle=True,
            num_workers=min(2, int(args.num_workers)),
            pin_memory=True,
            persistent_workers=int(args.num_workers) > 0,
            drop_last=False,
        )

    logger.info(
        "Data: labeled all=%d train=%d val_internal=%d (all=%d, fg_only=%s, kept=%d/%d) "
        "external_val=%d (all=%d, kept=%d/%d) unlabeled=%d | train_videos=%s | val_videos=%s",
        len(all_labeled),
        len(train_samples),
        len(val_samples),
        len(val_samples_all),
        bool(args.val_only_fg),
        val_fg_after,
        val_fg_before,
        len(external_val_samples),
        len(external_val_samples_all),
        ext_fg_after,
        ext_fg_before,
        len(unlabeled_paths),
        train_video_ids,
        val_video_ids,
    )

    encoder_weights = None if args.encoder_weights == "none" else args.encoder_weights
    model = get_model(
        arch=args.arch,
        encoder_name=args.encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    ).to(device)

    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    sup_loss_fn = get_loss_fn(
        loss_type=args.loss_type,
        dice_weight=float(args.dice_loss_weight),
        bce_weight=float(args.bce_loss_weight),
        focal_weight=float(args.focal_loss_weight),
        focal_gamma=float(args.focal_gamma),
        focal_alpha=float(args.focal_alpha),
        pos_weight=float(args.pos_weight),
    )
    if isinstance(sup_loss_fn, torch.nn.Module):
        sup_loss_fn = sup_loss_fn.to(device)

    unsup_bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    warmup_epochs = max(0, min(int(args.warmup_epochs), max(0, int(args.epochs) - 1)))
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(args.epochs) - warmup_epochs),
            eta_min=float(args.min_lr),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(args.epochs)),
            eta_min=float(args.min_lr),
        )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    save_json(
        out_dir / "split.json",
        {
            "labeled_root": str(labeled_root),
            "external_val_root": str(external_val_root),
            "unlabeled_root": str(unlabeled_root),
            "val_only_fg": bool(args.val_only_fg),
            "all_labeled_samples": [s.image_path.as_posix() for s in all_labeled],
            "train_samples": [s.image_path.as_posix() for s in train_samples],
            "val_internal_all_samples": [s.image_path.as_posix() for s in val_samples_all],
            "val_internal_samples": [s.image_path.as_posix() for s in val_samples],
            "external_val_all_samples": [s.image_path.as_posix() for s in external_val_samples_all],
            "external_val_samples": [s.image_path.as_posix() for s in external_val_samples],
            "train_videos": train_video_ids,
            "val_videos": val_video_ids,
        },
    )
    save_json(out_dir / "config.json", vars(args))

    history_path = out_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "epoch",
                "train_loss",
                "train_dice",
                "val_loss",
                "val_dice",
                "val_hd",
                "val_asd",
                "val_threshold",
                "val_pred_pos_ratio",
                "val_gt_pos_ratio",
                "val_valid_dist_cases",
                "score",
                "best_score",
                "lr",
                "epoch_sec",
            ]
        )

    refs = MetricRefs(hd_ref=float(args.score_hd_ref), asd_ref=float(args.score_asd_ref))
    best_score = -1.0
    best_epoch = 0
    best_thr = 0.5
    best_val_dice = float("nan")
    best_val_hd = float("nan")
    best_val_asd = float("nan")
    no_improve_epochs = 0

    for epoch in range(1, int(args.epochs) + 1):
        epoch_start = time.time()
        model.train()
        teacher.eval()

        lambda_u = compute_unsup_weight(
            epoch=epoch,
            semi_warmup_epochs=int(args.semi_warmup_epochs),
            unsup_weight=float(args.unsup_weight),
            ramp_epochs=int(args.unsup_ramp_epochs),
        )

        sup_loss_sum = 0.0
        unsup_loss_sum = 0.0
        total_loss_sum = 0.0
        train_dice_sum = 0.0
        pseudo_pos_ratio_sum = 0.0
        pseudo_conf_ratio_sum = 0.0
        steps = 0

        train_iter = iter(train_loader)
        unl_iter = iter(unl_loader) if unl_loader is not None else None
        num_steps = len(train_loader)

        for step in range(1, num_steps + 1):
            sup_batch, train_iter = cycle_next(train_loader, train_iter)
            images = sup_batch["image"].to(device, non_blocking=True)
            labels = sup_batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits_sup = model(images)
                sup_loss = sup_loss_fn(logits_sup, labels)
            with torch.no_grad():
                batch_train_dice = dice_from_logits(logits_sup, labels, ignore_empty_gt=True)

            unsup_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            pseudo_pos_ratio = 0.0
            pseudo_conf_ratio = 0.0

            if lambda_u > 0.0 and unl_loader is not None and unl_iter is not None:
                unl_batch, unl_iter = cycle_next(unl_loader, unl_iter)
                weak = unl_batch["weak"].to(device, non_blocking=True)
                strong = unl_batch["strong"].to(device, non_blocking=True)

                with torch.no_grad():
                    t_probs = predict_probs(teacher, weak, use_amp=use_amp, use_tta=False)

                pseudo = (t_probs >= float(args.pseudo_pos_thr)).float()
                conf_mask = ((t_probs >= float(args.pseudo_pos_thr)) | (t_probs <= float(args.pseudo_neg_thr))).float()
                pseudo_pos_ratio = float(pseudo.mean().item())

                if float(args.pseudo_min_area) > 0:
                    area = pseudo.flatten(1).sum(dim=1)
                    small = area < float(args.pseudo_min_area)
                    if small.any():
                        pseudo[small] = 0.0
                        conf_mask[small] = (t_probs[small] <= float(args.pseudo_neg_thr)).float()
                    pseudo_pos_ratio = float(pseudo.mean().item())

                # Guard against collapse: if pseudo labels are almost all background, skip unsupervised loss.
                if pseudo_pos_ratio < float(args.pseudo_min_pos_ratio):
                    conf_mask.zero_()

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits_u = model(strong)
                    loss_map = unsup_bce(logits_u, pseudo)
                    valid_pixels = conf_mask.sum()
                    if float(valid_pixels.item()) > 0.0:
                        unsup_loss = (loss_map * conf_mask).sum() / valid_pixels

                pseudo_conf_ratio = float(conf_mask.mean().item())

            total_loss = sup_loss + float(lambda_u) * unsup_loss

            if scaler is not None:
                scaler.scale(total_loss).backward()
                if float(args.grad_clip_norm) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if float(args.grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
                optimizer.step()

            update_ema(teacher, model, decay=float(args.ema_decay))

            sup_loss_sum += float(sup_loss.item())
            unsup_loss_sum += float(unsup_loss.item())
            total_loss_sum += float(total_loss.item())
            train_dice_sum += float(batch_train_dice)
            pseudo_pos_ratio_sum += pseudo_pos_ratio
            pseudo_conf_ratio_sum += pseudo_conf_ratio
            steps += 1

            if int(args.print_freq) > 0 and (step % int(args.print_freq) == 0 or step == num_steps):
                logger.info(
                    "Epoch %d Step %d/%d | lambda_u=%.3f | sup=%.4f unsup=%.4f total=%.4f | pseudo_pos=%.4f conf=%.4f",
                    epoch,
                    step,
                    num_steps,
                    lambda_u,
                    sup_loss_sum / max(1, steps),
                    unsup_loss_sum / max(1, steps),
                    total_loss_sum / max(1, steps),
                    pseudo_pos_ratio_sum / max(1, steps),
                    pseudo_conf_ratio_sum / max(1, steps),
                )

        # Use student for validation/model selection; teacher is only used for pseudo labeling.
        eval_model = model
        val_metrics = evaluate(
            model=eval_model,
            loader=val_loader,
            loss_fn=sup_loss_fn,
            device=device,
            use_amp=use_amp,
            threshold_candidates=args.threshold_candidates,
            use_tta=bool(args.val_tta),
            fixed_threshold=None,
        )

        ext_val_dice = float("nan")
        if ext_val_loader is not None:
            ext = evaluate(
                model=eval_model,
                loader=ext_val_loader,
                loss_fn=sup_loss_fn,
                device=device,
                use_amp=use_amp,
                threshold_candidates=args.threshold_candidates,
                use_tta=bool(args.val_tta),
                fixed_threshold=float(val_metrics["val_threshold"]),
            )
            ext_val_dice = float(ext["val_dice"])

        lr_now = float(optimizer.param_groups[0]["lr"])
        epoch_sec = time.time() - epoch_start

        quality = metric_quality_weighted(
            dsc=val_metrics["val_dice"],
            hd=val_metrics["val_hd"],
            asd=val_metrics["val_asd"],
            refs=refs,
            dsc_weight=float(args.score_dsc_weight),
            hd_weight=float(args.score_hd_weight),
            asd_weight=float(args.score_asd_weight),
        )
        score = float(quality["score"])

        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_thr = float(val_metrics["val_threshold"])
            best_val_dice = float(val_metrics["val_dice"])
            best_val_hd = float(val_metrics["val_hd"])
            best_val_asd = float(val_metrics["val_asd"])
            no_improve_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "teacher_state": teacher.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "args": vars(args),
                    "val_metrics": val_metrics,
                    "ext_val_dice": ext_val_dice,
                    "score": score,
                },
                ckpt_dir / "best.pt",
            )
        else:
            no_improve_epochs += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "teacher_state": teacher.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "args": vars(args),
                "val_metrics": val_metrics,
                "ext_val_dice": ext_val_dice,
                "score": score,
            },
            ckpt_dir / "last.pt",
        )

        if int(args.save_every) > 0 and epoch % int(args.save_every) == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "teacher_state": teacher.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "args": vars(args),
                    "val_metrics": val_metrics,
                    "ext_val_dice": ext_val_dice,
                    "score": score,
                },
                ckpt_dir / f"epoch_{epoch:03d}.pt",
            )

        with history_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    epoch,
                    f"{total_loss_sum / max(1, steps):.6f}",
                    f"{train_dice_sum / max(1, steps):.6f}",
                    f"{val_metrics['val_loss']:.6f}",
                    f"{val_metrics['val_dice']:.6f}",
                    f"{val_metrics['val_hd']:.6f}",
                    f"{val_metrics['val_asd']:.6f}",
                    f"{val_metrics['val_threshold']:.4f}",
                    f"{val_metrics['val_pred_pos_ratio']:.6f}",
                    f"{val_metrics['val_gt_pos_ratio']:.6f}",
                    f"{int(val_metrics['val_valid_dist_cases'])}",
                    f"{score:.6f}",
                    f"{best_score:.6f}",
                    f"{lr_now:.8f}",
                    f"{epoch_sec:.2f}",
                ]
            )

        logger.info(
            "Epoch %d/%d | lambda_u=%.3f | train(sup/unsup/total)=%.4f/%.4f/%.4f train_dice=%.4f | val_loss=%.4f val_dice=%.4f val_hd=%s val_asd=%s thr=%.2f pred_pos=%.4f gt_pos=%.4f dist_n=%d | score=%.4f best=%.4f(epoch=%d) | ext_dice=%s | lr=%.6g | %.1fs",
            epoch,
            int(args.epochs),
            lambda_u,
            sup_loss_sum / max(1, steps),
            unsup_loss_sum / max(1, steps),
            total_loss_sum / max(1, steps),
            train_dice_sum / max(1, steps),
            val_metrics["val_loss"],
            val_metrics["val_dice"],
            ("nan" if math.isnan(val_metrics["val_hd"]) else f"{val_metrics['val_hd']:.4f}"),
            ("nan" if math.isnan(val_metrics["val_asd"]) else f"{val_metrics['val_asd']:.4f}"),
            val_metrics["val_threshold"],
            val_metrics["val_pred_pos_ratio"],
            val_metrics["val_gt_pos_ratio"],
            int(val_metrics["val_valid_dist_cases"]),
            score,
            best_score,
            best_epoch,
            "nan" if math.isnan(ext_val_dice) else f"{ext_val_dice:.4f}",
            lr_now,
            epoch_sec,
        )

        scheduler.step()

        if int(args.early_stop_patience) > 0 and no_improve_epochs >= int(args.early_stop_patience):
            logger.info("Early stop at epoch %d (no improvement for %d epochs)", epoch, no_improve_epochs)
            break

    save_json(
        out_dir / "final_metrics.json",
        {
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_val_dice": best_val_dice,
            "best_val_hd": best_val_hd,
            "best_val_asd": best_val_asd,
            "best_threshold": best_thr,
            "train_samples": len(train_samples),
            "val_internal_all_samples": len(val_samples_all),
            "val_internal_samples": len(val_samples),
            "val_only_fg": bool(args.val_only_fg),
            "val_internal_fg_kept": val_fg_after,
            "val_internal_fg_before": val_fg_before,
            "external_val_all_samples": len(external_val_samples_all),
            "external_val_samples": len(external_val_samples),
            "external_val_fg_kept": ext_fg_after,
            "external_val_fg_before": ext_fg_before,
            "unlabeled_samples": len(unlabeled_paths),
            "train_videos": train_video_ids,
            "val_videos": val_video_ids,
        },
    )
    logger.info("Training finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
