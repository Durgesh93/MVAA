"""
Training engine for SSL nnU-Net experiments.

Usage:
    python engine.py

Or from Python:
    from engine import run_experiment
    run_experiment()
"""

import lightning as L
from omegaconf import OmegaConf
from configs import build_config
from datamodule import SSLnnUNetDataModule
from module import SSLnnUNetLightningModule
from trainer import build_trainer


def run_experiment(overrides=None):
    """
    Build config, datamodule, LightningModule, trainer, then run training.

    Args:
        overrides:
            Optional list of Hydra-style overrides, for example:
            [
                "dataset_id=Dataset002_MVAA_TEE_SSL",
                "configuration=2d",
                "fold=0",
                "trainer.max_epochs=25",
            ]
    """

    cfg = build_config(overrides=overrides)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    L.seed_everything(cfg.seed, workers=True)

    datamodule = SSLnnUNetDataModule(cfg.datamodule)
    model = SSLnnUNetLightningModule(cfg.litmodule)

    trainer = build_trainer(cfg)

    trainer.fit(
        model,
        datamodule=datamodule,
    )

    return trainer, model, datamodule, cfg


def main():
    run_experiment()


if __name__ == "__main__":
    main()