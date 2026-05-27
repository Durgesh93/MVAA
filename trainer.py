"""
Lightning trainer builder for SSL nnU-Net experiments.
"""

import lightning as L

from omegaconf import OmegaConf
from lightning.pytorch.strategies import DDPStrategy
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar

from utils import ActualValidationTQDMCallback


def build_trainer(cfg):
    """
    Build Lightning Trainer from cfg.trainer.

    Expected input:
        build_trainer(cfg.trainer)
    """

    trainer_cfg = OmegaConf.to_container(
        cfg,
        resolve=True,
    )
    
    trainer_cfg["callbacks"] = [
        TQDMProgressBar(refresh_rate=1),
        ActualValidationTQDMCallback(every=1),
    ]

    if trainer_cfg.get("strategy", None) == "ddp":
        trainer_cfg["strategy"] = DDPStrategy(
            cluster_environment=LightningEnvironment(),
            process_group_backend="nccl",
            find_unused_parameters=False,
        )

    return L.Trainer(**trainer_cfg)