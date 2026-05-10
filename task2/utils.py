#!/usr/bin/env python3
"""Training utility helpers."""

from __future__ import annotations

import json
import math
import os
import random
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
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_float(x) -> float:
    if x is None:
        return float("nan")
    v = float(x)
    if math.isnan(v) or math.isinf(v):
        return float("nan")
    return v


def metric_quality(dsc: float, hd: float, asd: float, refs: MetricRefs) -> Dict[str, float]:
    """Rank-style normalized score with configurable metric weights."""
    return metric_quality_weighted(
        dsc=dsc,
        hd=hd,
        asd=asd,
        refs=refs,
        dsc_weight=1.0,
        hd_weight=1.0,
        asd_weight=1.0,
    )


def metric_quality_weighted(
    dsc: float,
    hd: float,
    asd: float,
    refs: MetricRefs,
    dsc_weight: float = 1.0,
    hd_weight: float = 1.0,
    asd_weight: float = 1.0,
) -> Dict[str, float]:
    """Normalize metrics to [0,1]-like qualities, then aggregate by weights."""
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
        "w_dsc": w_dsc,
        "w_hd": w_hd,
        "w_asd": w_asd,
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


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
