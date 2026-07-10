from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_NAME = "experiment"


def build_config(
    config_name: str = DEFAULT_CONFIG_NAME, overrides: list[str] | None = None, resolve: bool = True
) -> DictConfig:
    """
    Build MVAA experiment config using Hydra + OmegaConf.

    Example:
        from configs import build_config

        cfg = build_config()

        cfg = build_config(overrides=[
            "dataset_id=Dataset002_MVAA_TEE_SSL",
            "configuration=2d",
            "fold=0",
            "trainer.max_epochs=25",
        ])
    """

    if overrides is None:
        overrides = []

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name, overrides=overrides)

    if resolve:
        OmegaConf.resolve(cfg)

    return cfg


def print_config(cfg: DictConfig) -> None:
    """
    Pretty-print config.
    """
    print(OmegaConf.to_yaml(cfg, resolve=True))


__all__ = ["build_config", "print_config"]
