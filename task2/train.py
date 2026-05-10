#!/usr/bin/env python3
"""Full-supervised training entry for task2 cardiac segmentation."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import torch
from monai.data import decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.transforms import AsDiscrete

from dataset import get_dataloaders
from model_factory import get_loss_fn, get_model
from utils import (
    MetricRefs,
    ensure_dir,
    get_device,
    metric_quality_weighted,
    save_json,
    seed_everything,
    update_metric_refs,
)


def setup_logger(output_dir: Path, log_name: str = "train.log") -> logging.Logger:
    logger = logging.getLogger("task2_train")
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
    parser = argparse.ArgumentParser(description="Train MONAI full-supervised segmentation model.")
    parser.add_argument("--data-dir", type=str, default="train", help="NIfTI directory")
    parser.add_argument("--output-dir", type=str, default="runs/full_supervised2", help="Output directory")
    parser.add_argument("--model", type=str, default="unet3d")
    parser.add_argument("--model-size", type=str, default="large", choices=["small", "base", "large"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-count", type=int, default=20, help="Use the last N cases as val")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--roi-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--enable-spacing-resample", action="store_true", help="Enable spacing resampling")
    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.5],
        help="Target voxel spacing (x y z) when spacing resample is enabled",
    )
    parser.add_argument("--train-crops", type=int, default=1, help="Random crops per volume each epoch")
    parser.add_argument("--sw-batch-size", type=int, default=4)
    parser.add_argument("--val-interval", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-cases", type=int, default=0, help="Debug only, 0 means all")
    parser.add_argument("--max-val-cases", type=int, default=0, help="Debug only, 0 means all")
    parser.add_argument("--log-name", type=str, default="train.log", help="Training log file name")
    parser.add_argument("--score-dsc-weight", type=float, default=0.6, help="Best-model score weight for DSC")
    parser.add_argument("--score-hd-weight", type=float, default=0.2, help="Best-model score weight for HD quality")
    parser.add_argument("--score-asd-weight", type=float, default=0.2, help="Best-model score weight for ASD quality")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, grad_clip_norm):
    model.train()
    loss_sum = 0.0
    step = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].long().to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        loss_sum += float(loss.item())
        step += 1

    return loss_sum / max(1, step)


@torch.no_grad()
def validate(model, loader, inferer, num_classes, device, logger: logging.Logger | None = None):
    model.eval()

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
            logger.info("Validation progress: %d/%d", idx, total_batches)

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

    out_dir = ensure_dir(args.output_dir)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    logger = setup_logger(out_dir, log_name=args.log_name)
    logger.info("Starting training")
    logger.info("Args: %s", vars(args))
    if args.num_workers > 0:
        logger.warning(
            "num_workers=%d may increase system RAM usage significantly for 3D spacing-resample. "
            "If training gets killed, retry with --num-workers 0.",
            args.num_workers,
        )

    train_loader, val_loader, train_files, val_files = get_dataloaders(
        data_dir=args.data_dir,
        val_count=args.val_count,
        roi_size=args.roi_size,
        batch_size=args.batch_size,
        num_samples=args.train_crops,
        num_workers=args.num_workers,
        max_train_cases=args.max_train_cases,
        max_val_cases=args.max_val_cases,
        num_classes=args.num_classes,
        enable_spacing_resample=args.enable_spacing_resample,
        target_spacing=args.target_spacing,
    )

    model = get_model(
        name=args.model,
        model_size=args.model_size,
        in_channels=1,
        out_channels=args.num_classes,
    ).to(device)
    loss_fn = get_loss_fn()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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

    log_csv = out_dir / "history.csv"
    with log_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
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
        ])

    split_info = {
        "train_count": len(train_files),
        "val_count": len(val_files),
        "train_cases": [x["case_id"] for x in train_files],
        "val_cases": [x["case_id"] for x in val_files],
    }
    save_json(out_dir / "split_info.json", split_info)

    logger.info("Device: %s", device)
    logger.info("Train/Val: %d/%d", len(train_files), len(val_files))
    logger.info("Validation cases (last %d): %s", args.val_count, split_info["val_cases"])
    logger.info(
        "Score weights (rank-normalized): DSC=%.3f, HD=%.3f, ASD=%.3f",
        args.score_dsc_weight,
        args.score_hd_weight,
        args.score_asd_weight,
    )

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            scaler,
            grad_clip_norm=1.0,
        )

        val_dsc = float("nan")
        val_hd = float("nan")
        val_asd = float("nan")
        quality = {"q_dsc": 0.0, "q_hd": 0.0, "q_asd": 0.0, "score": 0.0}

        if epoch % args.val_interval == 0:
            val_dsc, val_hd, val_asd = validate(
                model=model,
                loader=val_loader,
                inferer=inferer,
                num_classes=args.num_classes,
                device=device,
                logger=logger,
            )
            refs = update_metric_refs(refs, hd=val_hd, asd=val_asd)
            quality = metric_quality_weighted(
                dsc=val_dsc,
                hd=val_hd,
                asd=val_asd,
                refs=refs,
                dsc_weight=args.score_dsc_weight,
                hd_weight=args.score_hd_weight,
                asd_weight=args.score_asd_weight,
            )

            score = quality["score"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
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
                        "score_weights": {
                            "dsc": args.score_dsc_weight,
                            "hd": args.score_hd_weight,
                            "asd": args.score_asd_weight,
                        },
                    },
                )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
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
                    train_loss,
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

        msg = (
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"loss={train_loss:.4f} "
            f"DSC={val_dsc:.4f} HD={val_hd:.4f} ASD={val_asd:.4f} "
            f"score={quality['score']:.4f} best={best_score:.4f}"
        )
        logger.info(msg)

    total = time.time() - start
    logger.info(
        "Training done. best_epoch=%d, best_score=%.4f, total=%.1fs",
        best_epoch,
        best_score,
        total,
    )
    logger.info("Outputs: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
