"""
Training / evaluation engine for the SSL nnU-Net TEE experiment.

Usage:
    python engine.py train

    python engine.py test
    python engine.py test --ckpt best
    python engine.py test --ckpt last
    python engine.py test --ckpt /path/to/checkpoint.ckpt

    All commands use the experiment_name set in config/experiment_TEE.yaml.
    Results are written to / read from nnUNet_results/<experiment_name>/.

    train always clears the experiment's whole output folder first
    (checkpoints, validation, test, _rank_outputs).

    test scores the 40 official MVSeg2023 test cases, which ship
    labeled. It clears validation/test/_rank_outputs before running (but
    never checkpoints/, which it needs to load from) -- so a test run
    always starts from a clean slate instead of leaving stale zips from a
    previous train/test run sitting alongside freshly written ones.
"""

import multiprocessing
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import typer
import lightning as L

from hydra.utils import instantiate
from omegaconf import OmegaConf

from lightning.pytorch.strategies import DDPStrategy
from lightning.fabric.plugins.environments import LightningEnvironment


from config import build_config

from utils import (
    set_nnunet_env,
    resolve_runtime_config,
    resolve_prediction_ckpt,
)

# The augmenter's background workers are launched with 'spawn', not the
# platform default 'fork'. On this cluster, Lightning/the trainer already
# initializes a CUDA context in this process before the augmenter's worker
# pool starts (moving the model to the GPU happens ahead of the first
# train_dataloader() call) -- a fork() after that point marks every child as
# a "bad fork" in PyTorch's CUDA runtime, so the FIRST time a forked worker
# ever touches any torch.cuda API (observed happening right at the
# train-to-validation boundary, not during training itself) it dies with
# "Cannot re-initialize CUDA in forked subprocess". spawn gives each worker
# a fresh interpreter with no inherited CUDA/fork state, so it's never
# poisoned regardless of when it starts relative to the parent's CUDA init.
# Must be set before any multiprocessing.Process is created anywhere
# (including inside batchgenerators), so this runs at import time.
multiprocessing.set_start_method("spawn", force=True)

# Also still pin this process to 1 thread: even with spawn, a parallelized
# CPU op in this process could otherwise contend with the augmentation
# workers for the same handful of CPUs (see the min-worker-count fix in datamodule/data_module.py) -- keeping
# this process single-threaded leaves more of a starved CPU budget for them.
torch.set_num_threads(1)

app = typer.Typer()


CKPT = "best"

CONFIG_NAME = "experiment_TEE"


def _is_rank_zero():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _select_cluster_environment(trainer_cfg):
    """
    This cluster does not launch training via srun, so Lightning's automatic
    cluster-environment detection is unsafe: SLURMEnvironment's constructor
    validates srun variables eagerly and raises whenever SLURM_NTASKS > 1
    without SLURM_NTASKS_PER_NODE (this cluster's job scripts set --ntasks),
    regardless of accelerator/strategy. Force Lightning's unmanaged
    environment and pin topology explicitly so that detection never runs,
    for both single-device and DDP training.
    """

    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1

    trainer_cfg["num_nodes"] = 1
    trainer_cfg["devices"] = device_count

    strategy = trainer_cfg.get("strategy", "auto")

    if strategy == "ddp":
        trainer_cfg["strategy"] = DDPStrategy(cluster_environment=LightningEnvironment())
    else:
        trainer_cfg["plugins"] = [LightningEnvironment()]

    return trainer_cfg


def _build_trainer(cfg, prediction=False):
    """
    Build Lightning Trainer.

    Keep ModelCheckpoint callback also during prediction.
    For prediction, ckpt='best'/'last' is resolved manually before trainer.test().
    """

    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)

    trainer_cfg = _select_cluster_environment(trainer_cfg)

    callbacks = []

    if trainer_cfg.get("enable_progress_bar", True):
        if "progress_bar" in cfg:
            callbacks.append(instantiate(cfg.progress_bar))

    if trainer_cfg.get("enable_checkpointing", True):
        callbacks.append(instantiate(cfg.checkpoint))
        callbacks.append(instantiate(cfg.checkpoint_last))

    trainer = L.Trainer(**trainer_cfg, callbacks=callbacks)

    return trainer


def _build_objects(config_name, prediction=False):
    """
    Both commands clear the output folder first; they differ only in
    clear_results' clear_checkpoints flag. train clears checkpoints too
    (fresh run), test does not -- it needs the existing checkpoint to
    still be there when resolve_prediction_ckpt runs right after this.
    """

    cfg = build_config(config_name=config_name)

    set_nnunet_env(cfg)

    L.seed_everything(cfg.seed, workers=True)

    from datamodule import SSLnnUNetDataModule
    from module import SSLnnUNetLightningModule

    cfg = resolve_runtime_config(cfg, prediction=prediction)

    clear_results(cfg, clear_checkpoints=not prediction)

    datamodule = SSLnnUNetDataModule(cfg.datamodule)

    model = SSLnnUNetLightningModule(cfg.litmodule)

    trainer = _build_trainer(cfg, prediction=prediction)

    return cfg, datamodule, model, trainer


def _run_training(config_name):
    cfg, datamodule, model, trainer = _build_objects(config_name=config_name, prediction=False)

    trainer.fit(model=model, datamodule=datamodule)


def _run_test(config_name, ckpt=CKPT):
    cfg, datamodule, model, trainer = _build_objects(config_name=config_name, prediction=True)

    resolved_ckpt = resolve_prediction_ckpt(cfg=cfg, ckpt=ckpt)

    if _is_rank_zero():
        print()
        print("[test] Using checkpoint:")
        print(f"  requested: {ckpt}")
        print(f"  resolved : {resolved_ckpt}")
        print()

    trainer.test(model=model, datamodule=datamodule, ckpt_path=resolved_ckpt)


def clear_results(cfg, clear_checkpoints=True):
    """
    Clear this experiment's output folder: validation, test,
    and any leftover _rank_outputs -- plus checkpoints/ when
    clear_checkpoints=True (the default, passed explicitly True for a
    fresh train run).

    test passes clear_checkpoints=False: it needs the existing
    checkpoint to still be there when resolve_prediction_ckpt runs right
    after this call. Without clearing at all, a test run would reuse
    whatever zips are already sitting in validation/test from a
    prior train/test run -- write_prediction_case_zip replaces each
    {case_id}.zip in place, but any case_id not produced this run (or
    written by older code before a naming change) lingers untouched.
    """

    is_rank_zero = _is_rank_zero()

    output_folder = (
        Path(cfg.paths.nnunet_results)
        / cfg.dataset_id
        / f"{cfg.plans_identifier}__{cfg.configuration}"
    )

    if is_rank_zero:
        subfolders = ["validation", "test", "_rank_outputs"]

        if clear_checkpoints:
            subfolders.append("checkpoints")

        for subfolder in subfolders:
            target = output_folder / subfolder

            if target.exists():
                shutil.rmtree(target)

        output_folder.mkdir(parents=True, exist_ok=True)

        print()
        print(
            "[clear-results] Cleared experiment output folder"
            + ("" if clear_checkpoints else " (checkpoints preserved)")
            + ":"
        )
        print(f"  {output_folder}")
        print()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


@app.command()
def train():
    if _is_rank_zero():
        print()
        print(f"Config: {CONFIG_NAME}")
        print()

    _run_training(config_name=CONFIG_NAME)


@app.command()
def test(ckpt: str = typer.Option(CKPT, "--ckpt", help="Checkpoint to use: best, last, or full .ckpt path.")):
    """Score the 40 official MVSeg2023 test cases with a trained checkpoint."""
    if _is_rank_zero():
        print()
        print(f"Config: {CONFIG_NAME}")
        print(f"Checkpoint: {ckpt}")
        print()

    _run_test(config_name=CONFIG_NAME, ckpt=ckpt)


if __name__ == "__main__":
    app()
