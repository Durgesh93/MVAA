from pathlib import Path

from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parent

TASKS = ("ct", "tee", "video")


def build_config(task: str):
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}', must be one of {TASKS}")

    return OmegaConf.load(CONFIG_DIR / f"{task}.yaml")
