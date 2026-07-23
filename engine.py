"""
Training / prediction / submission engine for SSL nnU-Net experiments.

Usage:
    python engine.py train video
    python engine.py train tee
    python engine.py train ct

    python engine.py retrain video
    python engine.py retrain tee
    python engine.py retrain ct

    python engine.py predict video
    python engine.py predict tee
    python engine.py predict ct
    python engine.py predict all

    python engine.py predict video --ckpt best
    python engine.py predict video --ckpt last
    python engine.py predict video --ckpt /path/to/checkpoint.ckpt

    python engine.py submit

    All commands use the experiment_name set in the yaml configs. Results
    are written to / read from nnUNet_results/<experiment_name>/, and
    submit names its output submission_<experiment_name>.zip accordingly.
    To switch experiments, edit experiment_name in the yaml configs.

    train always clears the fold's whole output folder first (checkpoints,
    validation, prediction, submission, _rank_outputs).

    retrain resumes from checkpoints/last.ckpt (model weights, optimizer,
    scheduler, and epoch count) and continues up to whatever num_epochs is
    currently set in the yaml -- bump it there first if you want more
    epochs than the original run. Unlike train, retrain does NOT clear
    the output folder, since it needs the existing checkpoint.

    predict clears validation/prediction/submission/_rank_outputs before
    running (but never checkpoints/, which it needs to load from) -- so a
    predict run always starts from a clean slate instead of leaving stale
    zips from a previous train/predict run sitting alongside freshly
    written ones.
"""

import json
import re
import shutil
from pathlib import Path
import zipfile

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
    validate_experiment_name,
    collect_submission_files,
    resolve_prediction_ckpt,
)

# The augmenter's background workers are forked, not spawned. If this process
# ever runs a parallelized CPU op (e.g. torch.load-ing a checkpoint on
# retrain) before that fork, PyTorch's native intra-op thread pool gets
# initialized here first -- the fork then hands each worker a thread-pool
# descriptor pointing at threads that don't exist post-fork, and the
# worker's first CPU conv/op deadlocks forever waiting on them. Pinning this
# process to 1 thread means there's no pool to poison in the first place.
torch.set_num_threads(1)

app = typer.Typer()


FOLD_NUM = "all"
CKPT = "best"

VIDEO_SUBMISSION_SUFFIX = "_label_bin.png"
VIDEO_CASE_ID_PATTERN = re.compile(r"^(?P<video_id>.+)_(?P<frame>\d{6})$")
NIFTI_SUBMISSION_SUFFIX = "-pred.nii.gz"


CONFIG_MAP = {"ct": "experiment_CT", "tee": "experiment_TEE", "video": "experiment_video"}


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


def _build_objects(config_name, prediction=False, clear=None):
    """
    clear defaults to True for both train and predict, but clear_results'
    clear_checkpoints flag differs: train clears checkpoints too (fresh
    run), predict doesn't (it needs the existing checkpoint to still be
    there when resolve_prediction_ckpt runs right after this). retrain
    passes clear=False explicitly -- it's not a prediction run, but it
    must not wipe the checkpoint it's about to resume from either.
    """

    cfg = build_config(config_name=config_name, overrides=[f"fold={FOLD_NUM}"])

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

    print()
    print("[retrain] Resuming from checkpoint:")
    print(f"  {resolved_ckpt}")
    print(f"  target num_epochs (from current yaml): {trainer.max_epochs}")
    print()

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resolved_ckpt)


def _run_prediction(config_name, ckpt=CKPT):
    cfg, datamodule, model, trainer = _build_objects(config_name=config_name, prediction=True)

    resolved_ckpt = resolve_prediction_ckpt(cfg=cfg, ckpt=ckpt)

    print()
    print("[predict] Using checkpoint:")
    print(f"  requested: {ckpt}")
    print(f"  resolved : {resolved_ckpt}")
    print()

    trainer.test(model=model, datamodule=datamodule, ckpt_path=resolved_ckpt)


def _run_prediction_all(ckpt=CKPT):
    """
    Run prediction for ct, tee, and video.
    """

    if ckpt not in ["best", "last"]:
        raise typer.BadParameter(
            "When using 'predict all', use --ckpt best or --ckpt last. "
            "A single explicit .ckpt path cannot safely be shared across "
            "ct, tee, and video."
        )

    print()
    print("Predicting all experiments")
    print(f"Experiments: {', '.join(CONFIG_MAP.keys())}")
    print(f"Fold: {FOLD_NUM}")
    print(f"Checkpoint: {ckpt}")
    print()

    for experiment, config_name in CONFIG_MAP.items():
        print()
        print("=" * 80)
        print(f"[predict all] Experiment: {experiment}")
        print(f"[predict all] Config: {config_name}")
        print("=" * 80)
        print()

        _run_prediction(config_name=config_name, ckpt=ckpt)


def clear_results(cfg, clear_checkpoints=True):
    """
    Clear this experiment's fold output folder: validation, prediction,
    submission, and any leftover _rank_outputs -- plus checkpoints/ when
    clear_checkpoints=True (the default, passed explicitly True for a
    fresh train run).

    predict passes clear_checkpoints=False: it needs the existing
    checkpoint to still be there when resolve_prediction_ckpt runs right
    after this call. Without clearing at all, a predict run would reuse
    whatever zips are already sitting in validation/prediction/submission
    from a prior train/predict run -- write_prediction_case_zip replaces
    each {case_id}.zip in place, but any case_id not produced this run
    (or written by older code before a naming change) lingers untouched.

    `predict all` re-enters this function once per experiment (ct, tee,
    video) inside the same DDP-launched processes, with no barrier
    between experiments once the first .test() call has joined the
    process group -- every rank would otherwise call shutil.rmtree() on
    the same shared folder concurrently, racing each other's unlink calls
    (FileNotFoundError). Only rank 0 clears; every rank (rank 0 included)
    waits on a barrier afterwards so no rank touches the folder before
    the clear is done. Before the first .test() call the process group
    isn't initialized yet, so every rank is trivially "rank 0" -- safe
    here only because process spawn ordering guarantees the parent's
    clear for that first experiment completes before the child process
    that would redundantly repeat it even exists.
    """

    is_rank_zero = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

    fold_output_folder = (
        Path(cfg.paths.nnunet_results)
        / cfg.dataset_id
        / f"{cfg.plans_identifier}__{cfg.configuration}"
        / f"fold_{cfg.fold}"
    )

    if is_rank_zero:
        subfolders = ["validation", "prediction", "submission", "_rank_outputs"]

        if clear_checkpoints:
            subfolders.append("checkpoints")

        for subfolder in subfolders:
            target = fold_output_folder / subfolder

            if target.exists():
                shutil.rmtree(target)

        fold_output_folder.mkdir(parents=True, exist_ok=True)

        print()
        print(
            "[clear-results] Cleared fold output folder"
            + ("" if clear_checkpoints else " (checkpoints preserved)")
            + ":"
        )
        print(f"  {fold_output_folder}")
        print()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


@app.command()
def train(experiment: str = typer.Argument(..., help="Which experiment to train: ct, tee, or video.")):
    try:
        experiment, config_name = validate_experiment_name(experiment, CONFIG_MAP)
    except ValueError as e:
        raise typer.BadParameter(str(e))

    print()
    print(f"Training experiment: {experiment}")
    print(f"Config: {config_name}")
    print(f"Fold: {FOLD_NUM}")
    print()

    _run_training(config_name=config_name)


@app.command()
def retrain(experiment: str = typer.Argument(..., help="Which experiment to resume training: ct, tee, or video.")):
    """
    Resume training from checkpoints/last.ckpt (model, optimizer,
    scheduler, and epoch count), continuing up to whatever num_epochs is
    currently set in the yaml. Bump num_epochs in the yaml before running
    this if you want more epochs than the original run trained for.

    Unlike train, this does not clear the output folder first.
    """

    try:
        experiment, config_name = validate_experiment_name(experiment, CONFIG_MAP)
    except ValueError as e:
        raise typer.BadParameter(str(e))

    print()
    print(f"Resuming training for experiment: {experiment}")
    print(f"Config: {config_name}")
    print(f"Fold: {FOLD_NUM}")
    print()

    _run_retraining(config_name=config_name)


@app.command()
def predict(
    experiment: str = typer.Argument(..., help="Which experiment to predict: ct, tee, video, or all."),
    ckpt: str = typer.Option(CKPT, "--ckpt", help="Checkpoint to use: best, last, or full .ckpt path."),
):
    experiment = str(experiment).lower().strip()

    if experiment == "all":
        _run_prediction_all(ckpt=ckpt)

        return

    try:
        experiment, config_name = validate_experiment_name(experiment, CONFIG_MAP)
    except ValueError as e:
        raise typer.BadParameter(str(e))

    print()
    print(f"Predicting experiment: {experiment}")
    print(f"Config: {config_name}")
    print(f"Fold: {FOLD_NUM}")
    print(f"Checkpoint: {ckpt}")
    print()

    _run_prediction(config_name=config_name, ckpt=ckpt)


@app.command()
def submit():
    """
    Create one submission_<experiment_name>.zip for all experiments.

    task*_predictions.json is built here from the actual zip contents
    (case_id + path relative to the task's zip folder), not copied from
    disk, so it always matches the file layout inside the zip.

    Structure:
        submission_<experiment_name>.zip
        ├── t1_ct/
        │   ├── task1_predictions.json
        │   └── *-pred.nii.gz
        ├── t2_tee/
        │   ├── task2_predictions.json
        │   └── *-pred.nii.gz
        └── t3_vid/
            ├── task3_predictions.json
            └── <video_id>/
                └── *_label_bin.png
    """

    all_items = []
    resolved_experiment_name = None

    for experiment, config_name in CONFIG_MAP.items():
        cfg = build_config(config_name=config_name, overrides=[f"fold={FOLD_NUM}"])

        set_nnunet_env(cfg)

        resolved_experiment_name = str(cfg.experiment_name)

        item = collect_submission_files(cfg=cfg, fold_num=FOLD_NUM)

        item["experiment"] = experiment

        all_items.append(item)

    print()
    print("Creating one submission zip for all experiments")
    print(f"Fold: {FOLD_NUM}")
    print(f"Experiment name: {resolved_experiment_name}")
    print()

    output_zip = Path(__file__).resolve().parent / f"submission_{resolved_experiment_name}.zip"

    if output_zip.exists():
        output_zip.unlink()

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in all_items:
            prefix = item["prefix"]
            task_id = item["task_id"]
            prediction_files = item["prediction_files"]

            cases = []

            for prediction_file in prediction_files:
                relative_path = prediction_file.name

                if prediction_file.name.endswith(VIDEO_SUBMISSION_SUFFIX):
                    case_id = prediction_file.name[: -len(VIDEO_SUBMISSION_SUFFIX)]

                    match = VIDEO_CASE_ID_PATTERN.match(case_id)

                    if match is None:
                        raise ValueError(
                            "Video case_id does not match expected " f"'<video_id>_<6-digit frame>' pattern: {case_id}"
                        )

                    video_id = match.group("video_id")

                    relative_path = f"{video_id}/{prediction_file.name}"

                elif prediction_file.name.endswith(NIFTI_SUBMISSION_SUFFIX):
                    case_id = prediction_file.name[: -len(NIFTI_SUBMISSION_SUFFIX)]

                else:
                    raise ValueError(f"Unrecognized submission file name: {prediction_file.name}")

                zf.write(prediction_file, arcname=f"{prefix}/{relative_path}")

                cases.append({"case_id": case_id, "segmentation": relative_path})

            item["json_name"] = f"{task_id}_predictions.json"

            zf.writestr(
                f"{prefix}/{item['json_name']}",
                json.dumps({"cases": sorted(cases, key=lambda x: x["case_id"])}, indent=2),
            )

    print()
    print("[submission] Created single submission zip:")
    print(f"  {output_zip}")
    print()

    for item in all_items:
        print(f"[submission] {item['experiment']}")
        print(f"  folder inside zip: {item['prefix']}/")
        print(f"  json: {item['json_name']}")
        print(f"  prediction files: {len(item['prediction_files'])}")
        print()


if __name__ == "__main__":
    app()
