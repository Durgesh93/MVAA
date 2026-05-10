#!/usr/bin/env python3
"""Utility helpers for task1 semi-supervised training."""

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
    """Normalize metric scales so DSC/HD/ASD have similar contribution (1:1:1)."""
    dsc = safe_float(dsc)
    hd = safe_float(hd)
    asd = safe_float(asd)

    q_dsc = 0.0 if math.isnan(dsc) else max(0.0, min(1.0, dsc))
    q_hd = 0.0 if math.isnan(hd) else 1.0 / (1.0 + hd / max(1e-6, refs.hd_ref))
    q_asd = 0.0 if math.isnan(asd) else 1.0 / (1.0 + asd / max(1e-6, refs.asd_ref))

    score = (q_dsc + q_hd + q_asd) / 3.0
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


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sigmoid_rampup(current: int, rampup_length: int) -> float:
    if rampup_length <= 0:
        return 1.0
    current = np.clip(current, 0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def get_current_unsup_weight(epoch: int, max_weight: float, rampup_epochs: int) -> float:
    return max_weight * sigmoid_rampup(epoch, rampup_epochs)


def update_ema_variables(student: torch.nn.Module, teacher: torch.nn.Module, ema_decay: float) -> None:
    with torch.no_grad():
        student_state = dict(student.named_parameters())
        for name, teacher_param in teacher.named_parameters():
            teacher_param.data.mul_(ema_decay).add_(student_state[name].data, alpha=1.0 - ema_decay)

        student_buffers = dict(student.named_buffers())
        for name, teacher_buffer in teacher.named_buffers():
            teacher_buffer.data.copy_(student_buffers[name].data)


def apply_strong_perturbation(images: torch.Tensor, noise_std: float = 0.08, drop_prob: float = 0.05) -> torch.Tensor:
    """Simple perturbation for consistency branch: Gaussian noise + dropout-like masking."""
    if noise_std > 0:
        images = images + torch.randn_like(images) * noise_std

    if drop_prob > 0:
        keep_mask = (torch.rand_like(images) > drop_prob).float()
        images = images * keep_mask

    return images.clamp(0.0, 1.0)
