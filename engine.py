"""
Training / evaluation engine for the SSL nnU-Net TEE experiment.

Usage:
    python engine.py train

    python engine.py retrain

    python engine.py test
    python engine.py test --ckpt best
    python engine.py test --ckpt last
    python engine.py test --ckpt /path/to/checkpoint.ckpt

    All commands use the experiment_name set in config/experiment_TEE.yaml.
    Results are written to / read from nnUNet_results/<experiment_name>/.

    train always clears the experiment's whole output folder first
    (checkpoints, validation, test, _rank_outputs).

    retrain resumes from checkpoints/last.ckpt (model weights, optimizer,
    scheduler, and epoch count) and continues up to whatever num_epochs is
    currently set in the yaml -- bump it there first if you want more
    epochs than the original run. Unlike train, retrain does NOT clear
    the output folder, since it needs the existing checkpoint.

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
from lightning.pytorch.callbacks import Callback


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
# CPU op in this process before torch.load-ing a checkpoint on retrain could
# otherwise contend with the augmentation workers for the same handful of
# CPUs (see the min-worker-count fix in datamodule/data_module.py) -- keeping
# this process single-threaded leaves more of a starved CPU budget for them.
torch.set_num_threads(1)

app = typer.Typer()


CKPT = "best"

CONFIG_NAME = "experiment_TEE"


def _is_rank_zero():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _swa_checkpoint_path(cfg):
    return (
        Path(cfg.paths.nnunet_results)
        / cfg.dataset_id
        / f"{cfg.plans_identifier}__{cfg.configuration}"
        / "checkpoints"
        / "swa.ckpt"
    )


class _SWACheckpointCallback(Callback):
    """
    Saves the SWA running average's current weights to a fixed swa.ckpt
    path every epoch once the SWA phase starts, not just once at the very
    end. StochasticWeightAveraging itself only transfers/exposes the
    average into pl_module at on_train_end -- if the job crashes partway
    through the SWA phase (has happened at least once this session, for
    reasons unrelated to SWA itself), all that averaging work would
    otherwise be lost with nothing saved to show for it.

    Reads swa_callback's own running average directly (_average_model,
    n_averaged) rather than re-implementing averaging -- both are lightning
    private attributes, but this is a widely used pattern for exactly this
    gap (Lightning's built-in callback has no periodic-checkpoint hook of
    its own). Safe to read at on_train_epoch_end: _average_model is only
    ever updated once per epoch, at on_train_epoch_start, so its value is
    stable for the rest of that epoch.
    """

    def __init__(self, swa_callback, ckpt_path):
        self.swa_callback = swa_callback
        self.ckpt_path = Path(ckpt_path)

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        average_model = self.swa_callback._average_model
        n_averaged = self.swa_callback.n_averaged

        if average_model is None or n_averaged is None or int(n_averaged) == 0:
            return

        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": average_model.state_dict()}, self.ckpt_path)


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

    if "swa" in cfg:
        swa_callback = instantiate(cfg.swa)
        callbacks.append(swa_callback)
        callbacks.append(_SWACheckpointCallback(swa_callback, _swa_checkpoint_path(cfg)))

    trainer = L.Trainer(**trainer_cfg, callbacks=callbacks)

    return trainer


def _build_objects(config_name, prediction=False, clear=None):
    """
    clear defaults to True for both train and predict, but clear_results'
    clear_checkpoints flag differs: train clears checkpoints too (fresh
    run), predict doesn't (it needs the existing checkpoint to still be
    there when resolve_prediction_ckpt runs right after this). retrain
    passes clear=False explicitly -- it's not a prediction run, but it
    must not wipe the checkpoint it's about to resume from either.
    """

    cfg = build_config(config_name=config_name)

    set_nnunet_env(cfg)

    L.seed_everything(cfg.seed, workers=True)

    from datamodule import SSLnnUNetDataModule
    from module import SSLnnUNetLightningModule

    cfg = resolve_runtime_config(cfg, prediction=prediction)

    if clear is None:
        clear = True

    if clear:
        clear_results(cfg, clear_checkpoints=not prediction)

    datamodule = SSLnnUNetDataModule(cfg.datamodule)

    model = SSLnnUNetLightningModule(cfg.litmodule)

    trainer = _build_trainer(cfg, prediction=prediction)

    return cfg, datamodule, model, trainer


def _run_training(config_name):
    cfg, datamodule, model, trainer = _build_objects(config_name=config_name, prediction=False)

    trainer.fit(model=model, datamodule=datamodule)


def _run_retraining(config_name):
    cfg, datamodule, model, trainer = _build_objects(config_name=config_name, prediction=False, clear=False)

    resolved_ckpt = resolve_prediction_ckpt(cfg=cfg, ckpt="last")

    if not Path(resolved_ckpt).is_file():
        raise FileNotFoundError(
            f"No checkpoint to resume from: {resolved_ckpt}\n"
            "retrain needs an existing checkpoints/last.ckpt from a prior "
            "'train' run -- use 'train' for a fresh run instead."
        )

    if _is_rank_zero():
        print()
        print("[retrain] Resuming from checkpoint:")
        print(f"  {resolved_ckpt}")
        print(f"  target num_epochs (from current yaml): {trainer.max_epochs}")
        print()

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resolved_ckpt)


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
def retrain():
    """
    Resume training from checkpoints/last.ckpt (model, optimizer,
    scheduler, and epoch count), continuing up to whatever num_epochs is
    currently set in the yaml. Bump num_epochs in the yaml before running
    this if you want more epochs than the original run trained for.

    Unlike train, this does not clear the output folder first.
    """

    if _is_rank_zero():
        print()
        print(f"Config: {CONFIG_NAME}")
        print()

    _run_retraining(config_name=CONFIG_NAME)


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
