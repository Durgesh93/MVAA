#!/usr/bin/env python3
"""Semi-supervised training entry for task1 cardiac segmentation."""

from __future__ import annotations

import argparse
import csv
import logging
import shlex
import sys
import time
from pathlib import Path

import torch
from monai.data import decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.transforms import AsDiscrete

from dataset import get_task1_dataloaders
from model_factory import get_model, get_supervised_loss_fn, get_unsupervised_loss_fn
from utils import (
    MetricRefs,
    apply_strong_perturbation,
    ensure_dir,
    get_current_unsup_weight,
    get_device,
    metric_quality,
    save_json,
    seed_everything,
    update_ema_variables,
    update_metric_refs,
)


def setup_logger(output_dir: Path, log_name: str = "train.log") -> logging.Logger:
    logger = logging.getLogger("task1_train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(output_dir / log_name, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.handlers.clear()
    warnings_logger.setLevel(logging.WARNING)
    warnings_logger.propagate = False
    warnings_logger.addHandler(file_handler)
    warnings_logger.addHandler(stream_handler)

    def _handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Unhandled exception occurred.", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = _handle_exception
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MONAI semi-supervised segmentation model for task1.")
    parser.add_argument("--data-root", type=str, default="data", help="Task1 data root")
    parser.add_argument("--output-dir", type=str, default="runs/semi_supervised", help="Output directory")

    parser.add_argument("--model", type=str, default="unet3d")
    parser.add_argument("--model-size", type=str, default="base", choices=["small", "base", "large"])
    parser.add_argument("--num-classes", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="Optimization steps per epoch; <=0 means auto")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--roi-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--train-crops", type=int, default=1, help="Random crops per volume")

    parser.add_argument("--enable-spacing-resample", action="store_true", help="Enable spacing resampling")
    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.5],
        help="Target spacing (x y z) when spacing resample is enabled",
    )

    parser.add_argument("--unsup-weight", type=float, default=1.0, help="Maximum consistency loss weight")
    parser.add_argument(
        "--unsup-start-weight",
        type=float,
        default=0.0,
        help="Initial consistency weight before ramping to --unsup-weight",
    )
    parser.add_argument("--unsup-rampup-epochs", type=int, default=30, help="Rampup epochs for unsupervised weight")
    parser.add_argument("--unsup-warmup-epochs", type=int, default=40, help="Warmup epochs with zero unsupervised weight")
    parser.add_argument(
        "--unsup-ratio",
        type=int,
        default=2,
        help="Number of unlabeled mini-batches used per labeled mini-batch step",
    )
    parser.add_argument("--pseudo-threshold", type=float, default=0.6, help="Teacher pseudo-label confidence threshold")
    parser.add_argument("--ema-decay", type=float, default=0.99, help="EMA decay for teacher model")
    parser.add_argument("--strong-noise-std", type=float, default=0.08, help="Std of Gaussian noise for strong perturbation")
    parser.add_argument("--strong-drop-prob", type=float, default=0.05, help="Voxel dropout prob for strong perturbation")

    parser.add_argument("--sw-batch-size", type=int, default=8)
    parser.add_argument("--val-interval", type=int, default=2)
    
    parser.add_argument("--max-labeled-cases", type=int, default=0, help="Debug only, 0 means all")
    parser.add_argument("--max-unlabeled-cases", type=int, default=0, help="Debug only, 0 means all")
    parser.add_argument("--max-val-cases", type=int, default=0, help="Debug only, 0 means all")

    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio from labeled train set")
    parser.add_argument("--val-count", type=int, default=0, help="Override val size from labeled train set")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--log-name", type=str, default="train.log", help="Training log file name")

    return parser.parse_args()


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _supervised_step(model, batch, loss_fn, device):
    images = batch["image"].to(device)
    labels = batch["label"].long().to(device)
    logits = model(images)
    loss = loss_fn(logits, labels)
    return loss


def _unsupervised_step(
    student,
    teacher,
    batch,
    loss_fn,
    device,
    pseudo_threshold: float,
    strong_noise_std: float,
    strong_drop_prob: float,
):
    images_u = batch["image"].to(device)
    images_u_strong = apply_strong_perturbation(
        images_u,
        noise_std=strong_noise_std,
        drop_prob=strong_drop_prob,
    )

    with torch.no_grad():
        teacher_logits = teacher(images_u)
        teacher_probs = torch.softmax(teacher_logits, dim=1)
        pseudo_conf, pseudo_label = torch.max(teacher_probs, dim=1)
        mask = (pseudo_conf >= pseudo_threshold).float()

    student_logits = student(images_u_strong)
    per_voxel_ce = loss_fn(student_logits, pseudo_label)

    denom = mask.sum()
    if denom.item() < 1:
        unsup_loss = torch.zeros((), device=device)
        valid_ratio = 0.0
    else:
        unsup_loss = (per_voxel_ce * mask).sum() / denom
        valid_ratio = float(mask.mean().item())

    return unsup_loss, valid_ratio


def train_one_epoch(
    student,
    teacher,
    labeled_loader,
    unlabeled_loader,
    optimizer,
    sup_loss_fn,
    unsup_loss_fn,
    device,
    scaler,
    grad_clip_norm,
    unsup_weight,
    pseudo_threshold,
    ema_decay,
    strong_noise_std,
    strong_drop_prob,
    steps_per_epoch,
    unsup_ratio,
):
    student.train()
    teacher.eval()

    sup_loss_sum = 0.0
    unsup_loss_sum = 0.0
    total_loss_sum = 0.0
    valid_ratio_sum = 0.0

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader) if unsup_ratio > 0 else None
    for _ in range(steps_per_epoch):
        labeled_batch, labeled_iter = _next_batch(labeled_iter, labeled_loader)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                sup_loss = _supervised_step(student, labeled_batch, sup_loss_fn, device)
                unsup_loss = torch.zeros((), device=device)
                valid_ratio = 0.0
                for _ in range(max(0, unsup_ratio)):
                    unlabeled_batch, unlabeled_iter = _next_batch(unlabeled_iter, unlabeled_loader)
                    unsup_item, valid_item = _unsupervised_step(
                        student,
                        teacher,
                        unlabeled_batch,
                        unsup_loss_fn,
                        device,
                        pseudo_threshold,
                        strong_noise_std,
                        strong_drop_prob,
                    )
                    unsup_loss = unsup_loss + unsup_item
                    valid_ratio += valid_item
                if unsup_ratio > 0:
                    unsup_loss = unsup_loss / float(unsup_ratio)
                    valid_ratio = valid_ratio / float(unsup_ratio)
                total_loss = sup_loss + unsup_weight * unsup_loss

            scaler.scale(total_loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            sup_loss = _supervised_step(student, labeled_batch, sup_loss_fn, device)
            unsup_loss = torch.zeros((), device=device)
            valid_ratio = 0.0
            for _ in range(max(0, unsup_ratio)):
                unlabeled_batch, unlabeled_iter = _next_batch(unlabeled_iter, unlabeled_loader)
                unsup_item, valid_item = _unsupervised_step(
                    student,
                    teacher,
                    unlabeled_batch,
                    unsup_loss_fn,
                    device,
                    pseudo_threshold,
                    strong_noise_std,
                    strong_drop_prob,
                )
                unsup_loss = unsup_loss + unsup_item
                valid_ratio += valid_item
            if unsup_ratio > 0:
                unsup_loss = unsup_loss / float(unsup_ratio)
                valid_ratio = valid_ratio / float(unsup_ratio)
            total_loss = sup_loss + unsup_weight * unsup_loss
            total_loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip_norm)
            optimizer.step()

        update_ema_variables(student, teacher, ema_decay)

        sup_loss_sum += float(sup_loss.item())
        unsup_loss_sum += float(unsup_loss.item())
        total_loss_sum += float(total_loss.item())
        valid_ratio_sum += valid_ratio

    denom = max(1, steps_per_epoch)
    return {
        "sup_loss": sup_loss_sum / denom,
        "unsup_loss": unsup_loss_sum / denom,
        "train_loss": total_loss_sum / denom,
        "pseudo_valid_ratio": valid_ratio_sum / denom,
    }


@torch.no_grad()
def evaluate(model, loader, inferer, num_classes, device, logger: logging.Logger | None = None, phase: str = "val"):
    model.eval()

    if len(loader) == 0:
        if logger is not None:
            logger.warning("%s loader is empty; skipping metrics and returning NaN.", phase.capitalize())
        return float("nan"), float("nan"), float("nan")

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd_metric = HausdorffDistanceMetric(
        include_background=False,
        distance_metric="euclidean",
        percentile=None,
        reduction="mean",
    )
    asd_metric = SurfaceDistanceMetric(
        include_background=False,
        symmetric=True,
        reduction="mean",
    )

    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)

    total_batches = len(loader)
    for idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = inferer(images, model)
        pred_list = [post_pred(x) for x in decollate_batch(logits)]
        label_list = [post_label(x) for x in decollate_batch(labels)]

        dice_metric(y_pred=pred_list, y=label_list)
        hd_metric(y_pred=pred_list, y=label_list)
        asd_metric(y_pred=pred_list, y=label_list)

        if logger is not None:
            logger.info("%s progress: %d/%d", phase.capitalize(), idx, total_batches)

    dice = float(dice_metric.aggregate().item())
    hd = float(hd_metric.aggregate().item())
    asd = float(asd_metric.aggregate().item())

    dice_metric.reset()
    hd_metric.reset()
    asd_metric.reset()

    return dice, hd, asd


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)

    device = get_device()
    args.num_workers = args.num_workers if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_dir = ensure_dir(args.output_dir)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    logger = setup_logger(out_dir, log_name=args.log_name)

    logger.info("Starting task1 semi-supervised training")
    logger.info("Command: %s", shlex.join(["python", *sys.argv]))
    logger.info("Args: %s", vars(args))
    if args.num_workers > 0:
        logger.warning(
            "num_workers=%d may trigger DataLoader worker OOM on large 3D volumes. "
            "If workers exit unexpectedly, retry with --num-workers 0.",
            args.num_workers,
        )

    labeled_loader, unlabeled_loader, val_loader, split_info = get_task1_dataloaders(
        root_dir=args.data_root,
        roi_size=args.roi_size,
        batch_size=args.batch_size,
        num_samples=args.train_crops,
        num_workers=args.num_workers,
        enable_spacing_resample=args.enable_spacing_resample,
        target_spacing=args.target_spacing,
        max_labeled_cases=args.max_labeled_cases,
        max_unlabeled_cases=args.max_unlabeled_cases,
        max_val_cases=args.max_val_cases,
        val_ratio=args.val_ratio,
        val_count=args.val_count,
        split_seed=args.split_seed,
    )

    student = get_model(
        name=args.model,
        model_size=args.model_size,
        in_channels=1,
        out_channels=args.num_classes,
    ).to(device)
    teacher = get_model(
        name=args.model,
        model_size=args.model_size,
        in_channels=1,
        out_channels=args.num_classes,
    ).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)

    sup_loss_fn = get_supervised_loss_fn()
    unsup_loss_fn = get_unsupervised_loss_fn()

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    inferer = SlidingWindowInferer(
        roi_size=tuple(args.roi_size),
        sw_batch_size=args.sw_batch_size,
        overlap=0.25,
        mode="gaussian",
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    refs = MetricRefs()
    best_score = -1.0
    best_epoch = -1

    save_json(out_dir / "split_info.json", split_info)

    if args.steps_per_epoch <= 0:
        args.steps_per_epoch = max(1, len(labeled_loader))

    log_csv = out_dir / "history.csv"
    with log_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "sup_loss",
                "unsup_loss",
                "unsup_weight",
                "pseudo_valid_ratio",
                "val_dsc",
                "val_hd",
                "val_asd",
                "q_dsc",
                "q_hd",
                "q_asd",
                "score",
                "best_score",
                "lr",
                "elapsed_sec",
            ]
        )

    logger.info("Device: %s", device)
    logger.info(
        "Train labeled/unlabeled/val: %d/%d/%d",
        split_info["train_labeled_count"],
        split_info["train_unlabeled_count"],
        split_info["val_count"],
    )
    logger.info(
        "Train schedule: steps_per_epoch=%d, warmup=%d (supervised-only), then unsup_ratio=%d, unsup_start_weight=%.3f -> unsup_weight=%.3f",
        args.steps_per_epoch,
        args.unsup_warmup_epochs,
        args.unsup_ratio,
        args.unsup_start_weight,
        args.unsup_weight,
    )

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        if epoch <= args.unsup_warmup_epochs:
            unsup_weight = 0.0
        else:
            ramp_epoch = epoch - args.unsup_warmup_epochs
            ramp = get_current_unsup_weight(ramp_epoch, 1.0, args.unsup_rampup_epochs)
            unsup_weight = args.unsup_start_weight + (args.unsup_weight - args.unsup_start_weight) * ramp
            unsup_weight = max(0.0, float(unsup_weight))

        unsup_ratio_this_epoch = 0 if epoch <= args.unsup_warmup_epochs else max(1, args.unsup_ratio)

        train_stats = train_one_epoch(
            student=student,
            teacher=teacher,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            optimizer=optimizer,
            sup_loss_fn=sup_loss_fn,
            unsup_loss_fn=unsup_loss_fn,
            device=device,
            scaler=scaler,
            grad_clip_norm=1.0,
            unsup_weight=unsup_weight,
            pseudo_threshold=args.pseudo_threshold,
            ema_decay=args.ema_decay,
            strong_noise_std=args.strong_noise_std,
            strong_drop_prob=args.strong_drop_prob,
            steps_per_epoch=args.steps_per_epoch,
            unsup_ratio=unsup_ratio_this_epoch,
        )

        val_dsc = float("nan")
        val_hd = float("nan")
        val_asd = float("nan")
        quality = {"q_dsc": 0.0, "q_hd": 0.0, "q_asd": 0.0, "score": 0.0}

        if epoch % args.val_interval == 0:
            if len(val_loader) == 0:
                logger.warning("Validation skipped because val loader is empty.")
            else:
                val_dsc, val_hd, val_asd = evaluate(
                    model=student,
                    loader=val_loader,
                    inferer=inferer,
                    num_classes=args.num_classes,
                    device=device,
                    logger=logger,
                    phase="val",
                )
                refs = update_metric_refs(refs, hd=val_hd, asd=val_asd)
                quality = metric_quality(val_dsc, val_hd, val_asd, refs)

                score = quality["score"]
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    torch.save(
                    {
                        "epoch": epoch,
                        "student_state_dict": student.state_dict(),
                        "teacher_state_dict": teacher.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),
                        "metrics": {
                            "dsc": val_dsc,
                            "hd": val_hd,
                            "asd": val_asd,
                            **quality,
                        },
                        "refs": {"hd_ref": refs.hd_ref, "asd_ref": refs.asd_ref},
                    },
                        ckpt_dir / "best_model.pt",
                    )
                    save_json(
                    out_dir / "best_metrics.json",
                    {
                        "best_epoch": best_epoch,
                        "best_score": best_score,
                        "dsc": val_dsc,
                        "hd": val_hd,
                        "asd": val_asd,
                        **quality,
                        "hd_ref": refs.hd_ref,
                        "asd_ref": refs.asd_ref,
                        },
                    )

        torch.save(
            {
                "epoch": epoch,
                "student_state_dict": student.state_dict(),
                "teacher_state_dict": teacher.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            },
            ckpt_dir / "latest_model.pt",
        )

        scheduler.step()

        elapsed = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        with log_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    train_stats["train_loss"],
                    train_stats["sup_loss"],
                    train_stats["unsup_loss"],
                    unsup_weight,
                    train_stats["pseudo_valid_ratio"],
                    val_dsc,
                    val_hd,
                    val_asd,
                    quality["q_dsc"],
                    quality["q_hd"],
                    quality["q_asd"],
                    quality["score"],
                    best_score,
                    lr,
                    elapsed,
                ]
            )

        logger.info(
            "Epoch [%03d/%d] loss=%.4f (sup=%.4f unsup=%.4f w=%.3f valid=%.3f) "
            "DSC=%.4f HD=%.4f ASD=%.4f score=%.4f best=%.4f",
            epoch,
            args.epochs,
            train_stats["train_loss"],
            train_stats["sup_loss"],
            train_stats["unsup_loss"],
            unsup_weight,
            train_stats["pseudo_valid_ratio"],
            val_dsc,
            val_hd,
            val_asd,
            quality["score"],
            best_score,
        )

    total = time.time() - start
    logger.info(
        "Training done. best_epoch=%d, best_score=%.4f, total=%.1fs",
        best_epoch,
        best_score,
        total,
    )



    return 0


if __name__ == "__main__":
    raise SystemExit(main())
