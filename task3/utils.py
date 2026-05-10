#!/usr/bin/env python3
"""Utility helpers for task3 2D training."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch


@dataclass
class MetricRefs:
    hd_ref: float = 20.0
    asd_ref: float = 3.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data: Dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_float(x) -> float:
    if x is None:
        return float("nan")
    v = float(x)
    if math.isnan(v) or math.isinf(v):
        return float("nan")
    return v


def metric_quality_weighted(
    dsc: float,
    hd: float,
    asd: float,
    refs: MetricRefs,
    dsc_weight: float = 1.0,
    hd_weight: float = 1.0,
    asd_weight: float = 1.0,
) -> Dict[str, float]:
    dsc = safe_float(dsc)
    hd = safe_float(hd)
    asd = safe_float(asd)

    q_dsc = 0.0 if math.isnan(dsc) else max(0.0, min(1.0, dsc))
    q_hd = 0.0 if math.isnan(hd) else 1.0 / (1.0 + hd / max(1e-6, refs.hd_ref))
    q_asd = 0.0 if math.isnan(asd) else 1.0 / (1.0 + asd / max(1e-6, refs.asd_ref))

    w_dsc = max(0.0, float(dsc_weight))
    w_hd = max(0.0, float(hd_weight))
    w_asd = max(0.0, float(asd_weight))
    w_sum = max(1e-8, w_dsc + w_hd + w_asd)
    score = (w_dsc * q_dsc + w_hd * q_hd + w_asd * q_asd) / w_sum
    return {
        "q_dsc": q_dsc,
        "q_hd": q_hd,
        "q_asd": q_asd,
        "score": score,
    }


def update_metric_refs(refs: MetricRefs, hd: float, asd: float, momentum: float = 0.9) -> MetricRefs:
    hd = safe_float(hd)
    asd = safe_float(asd)
    if not math.isnan(hd):
        refs.hd_ref = momentum * refs.hd_ref + (1.0 - momentum) * max(hd, 1e-3)
    if not math.isnan(asd):
        refs.asd_ref = momentum * refs.asd_ref + (1.0 - momentum) * max(asd, 1e-3)
    return refs


def setup_logger(output_dir: Path, log_name: str = "train.log") -> logging.Logger:
    logger = logging.getLogger("task3_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

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
