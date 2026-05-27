"""
Training / prediction / submission engine for SSL nnU-Net experiments.

Usage:
    python engine.py train video
    python engine.py train tee
    python engine.py train ct

    python engine.py predict video
    python engine.py predict tee
    python engine.py predict ct

    python engine.py submit

Examples:
    python engine.py train video --fold-num all
    python engine.py train video --fold-num all --keep-checkpoints

    python engine.py predict video --fold-num all --ckpt best
    python engine.py predict video --fold-num all --ckpt last
    python engine.py predict video --fold-num all --ckpt /path/to/checkpoint.ckpt

    python engine.py submit --fold-num all

With py wrapper:
    py engine.py train video --fold-num all
    py engine.py predict video --fold-num all --ckpt last
    py engine.py submit
"""

from pathlib import Path
import shutil
import zipfile

import typer
import torch
import lightning as L

from hydra.utils import instantiate
from omegaconf import OmegaConf

from config import build_config
from utils import (
    set_nnunet_env,
    ActualValidationTQDMCallback,
)


app = typer.Typer()


CONFIG_MAP = {
    "ct": "experiment_CT",
    "tee": "experiment_TEE",
    "video": "experiment_video",
}


def _get_num_devices(cfg):
    strategy = cfg.trainer.get("strategy", None)

    if strategy == "ddp":
        return max(1, torch.cuda.device_count())

    return 1


def _get_checkpoint_dir(cfg):
    return (
        Path(cfg.paths.nnunet_results)
        / cfg.dataset_id
        / f"{cfg.plans_identifier}__{cfg.configuration}"
        / f"fold_{cfg.fold}"
        / "checkpoints"
    )


def _clear_checkpoint_dir(cfg):
    checkpoint_dir = _get_checkpoint_dir(cfg)

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("[checkpoint] Cleared checkpoint directory:")
    print(f"  {checkpoint_dir}")
    print()


def _build_trainer(cfg, prediction=False):
    """
    Build Lightning Trainer.

    Training:
        - normal Lightning TQDMProgressBar from cfg.progress_bar
        - custom ActualValidationTQDMCallback for actual validation/prediction stages
        - checkpoint callback from cfg.checkpoint

    Prediction:
        - normal Lightning TQDMProgressBar from cfg.progress_bar
        - custom ActualValidationTQDMCallback
        - no checkpoint callback
    """

    trainer_cfg = OmegaConf.to_container(
        cfg.trainer,
        resolve=True,
    )

    callbacks = []

    # ------------------------------------------------------------
    # Normal Lightning progress bar
    # This requires cfg.progress_bar to be outside cfg.trainer.
    # ------------------------------------------------------------
    if cfg.trainer.get("enable_progress_bar", True):
        if "progress_bar" in cfg:
            callbacks.append(
                instantiate(cfg.progress_bar)
            )

    # ------------------------------------------------------------
    # Custom tqdm bar for actual validation / prediction / zip / metrics
    # ------------------------------------------------------------
    callbacks.append(
        ActualValidationTQDMCallback(every=1)
    )

    # ------------------------------------------------------------
    # Checkpointing only during training
    # ------------------------------------------------------------
    if not prediction and cfg.trainer.get("enable_checkpointing", True):
        checkpoint_callback = instantiate(cfg.checkpoint)
        callbacks.append(checkpoint_callback)

    trainer = L.Trainer(
        **trainer_cfg,
        callbacks=callbacks,
    )

    return trainer


def _build_objects(
    config_name,
    fold_num="all",
    prediction=False,
    clear_checkpoints=False,
):
    cfg = build_config(
        config_name=config_name,
        overrides=[f"fold={fold_num}"],
    )

    set_nnunet_env(cfg)
    L.seed_everything(cfg.seed, workers=True)

    from datamodule import SSLnnUNetDataModule
    from module import SSLnnUNetLightningModule

    if prediction:
        # Prediction/export should run on one process only.
        cfg.datamodule.num_devices = 1
        cfg.trainer.devices = 1
        cfg.trainer.strategy = "auto"
        cfg.trainer.num_nodes = 1

        # We load from ckpt_path in trainer.predict(...), so no callback needed.
        cfg.trainer.enable_checkpointing = False

    else:
        cfg.datamodule.num_devices = _get_num_devices(cfg)
        cfg.trainer.devices = cfg.datamodule.num_devices

        # Clear old checkpoints only when training.
        if clear_checkpoints:
            _clear_checkpoint_dir(cfg)

    datamodule = SSLnnUNetDataModule(cfg.datamodule)

    cfg.trainer.limit_train_batches = datamodule.limit_train_batches
    cfg.trainer.limit_val_batches = datamodule.limit_val_batches

    model = SSLnnUNetLightningModule(cfg.litmodule)

    trainer = _build_trainer(
        cfg,
        prediction=prediction,
    )

    return cfg, datamodule, model, trainer


def _run_training(
    config_name,
    fold_num="all",
    clear_checkpoints=True,
):
    cfg, datamodule, model, trainer = _build_objects(
        config_name=config_name,
        fold_num=fold_num,
        prediction=False,
        clear_checkpoints=clear_checkpoints,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


def _run_prediction(
    config_name,
    fold_num="all",
    ckpt="best",
):
    cfg, datamodule, model, trainer = _build_objects(
        config_name=config_name,
        fold_num=fold_num,
        prediction=True,
        clear_checkpoints=False,
    )

    trainer.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt,
    )


def _get_prediction_folder(cfg):
    return (
        Path(cfg.paths.nnunet_results)
        / cfg.dataset_id
        / f"{cfg.plans_identifier}__{cfg.configuration}"
        / f"fold_{cfg.fold}"
        / cfg.prefix
    )


def _collect_submission_files(config_name, fold_num="all"):
    cfg = build_config(
        config_name=config_name,
        overrides=[f"fold={fold_num}"],
    )

    set_nnunet_env(cfg)

    prediction_folder = _get_prediction_folder(cfg)

    if not prediction_folder.exists():
        raise FileNotFoundError(
            f"Prediction folder not found: {prediction_folder}"
        )

    json_file = prediction_folder / f"{cfg.task_id}_predictions.json"

    if not json_file.exists():
        raise FileNotFoundError(
            f"Prediction JSON not found: {json_file}"
        )

    nii_files = sorted(prediction_folder.glob("*.nii.gz"))

    if not nii_files:
        raise FileNotFoundError(
            f"No .nii.gz prediction files found in: {prediction_folder}"
        )

    return {
        "prefix": cfg.prefix,
        "task_id": cfg.task_id,
        "prediction_folder": prediction_folder,
        "json_file": json_file,
        "nii_files": nii_files,
    }


def _make_submission_zip_all(fold_num="all"):
    """
    Create one submission.zip in the engine.py folder.

    Structure:
        submission.zip
        ├── t1_ct/
        │   ├── task1_predictions.json
        │   └── *.nii.gz
        ├── t2_tee/
        │   ├── task2_predictions.json
        │   └── *.nii.gz
        └── t3_vid/
            ├── task3_predictions.json
            └── *.nii.gz
    """

    output_zip = Path(__file__).resolve().parent / "submission.zip"

    if output_zip.exists():
        output_zip.unlink()

    all_items = []

    for experiment, config_name in CONFIG_MAP.items():
        item = _collect_submission_files(
            config_name=config_name,
            fold_num=fold_num,
        )

        item["experiment"] = experiment
        all_items.append(item)

    with zipfile.ZipFile(
        output_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        for item in all_items:
            prefix = item["prefix"]
            json_file = item["json_file"]
            nii_files = item["nii_files"]

            zf.write(
                json_file,
                arcname=f"{prefix}/{json_file.name}",
            )

            for nii_file in nii_files:
                zf.write(
                    nii_file,
                    arcname=f"{prefix}/{nii_file.name}",
                )

    print()
    print("[submission] Created single submission zip:")
    print(f"  {output_zip}")
    print()

    for item in all_items:
        print(f"[submission] {item['experiment']}")
        print(f"  folder inside zip: {item['prefix']}/")
        print(f"  json: {item['json_file'].name}")
        print(f"  nii.gz files: {len(item['nii_files'])}")
        print()

    return output_zip


@app.command()
def train(
    experiment: str = typer.Argument(
        ...,
        help="Which experiment to train: ct, tee, or video.",
    ),
    fold_num: str = typer.Option(
        "all",
        "--fold-num",
        "-f",
        help="Fold number to train, or 'all'. Default is 'all'.",
    ),
    clear_checkpoints: bool = typer.Option(
        True,
        "--clear-checkpoints/--keep-checkpoints",
        help="Clear checkpoint directory before training. Default is true.",
    ),
):
    experiment = experiment.lower()

    if experiment not in CONFIG_MAP:
        valid = ", ".join(CONFIG_MAP.keys())
        raise typer.BadParameter(
            f"Unknown experiment '{experiment}'. Choose one of: {valid}"
        )

    config_name = CONFIG_MAP[experiment]

    print()
    print(f"Training experiment: {experiment}")
    print(f"Config: {config_name}")
    print(f"Fold: {fold_num}")
    print(f"Clear checkpoints: {clear_checkpoints}")
    print()

    _run_training(
        config_name=config_name,
        fold_num=fold_num,
        clear_checkpoints=clear_checkpoints,
    )


@app.command()
def predict(
    experiment: str = typer.Argument(
        ...,
        help="Which experiment to predict: ct, tee, or video.",
    ),
    fold_num: str = typer.Option(
        "all",
        "--fold-num",
        "-f",
        help="Fold number to predict, or 'all'. Default is 'all'.",
    ),
    ckpt: str = typer.Option(
        "best",
        "--ckpt",
        "-c",
        help="Checkpoint to use: 'best', 'last', or full checkpoint path.",
    ),
):
    experiment = experiment.lower()

    if experiment not in CONFIG_MAP:
        valid = ", ".join(CONFIG_MAP.keys())
        raise typer.BadParameter(
            f"Unknown experiment '{experiment}'. Choose one of: {valid}"
        )

    config_name = CONFIG_MAP[experiment]

    print()
    print(f"Predicting experiment: {experiment}")
    print(f"Config: {config_name}")
    print(f"Fold: {fold_num}")
    print(f"Checkpoint: {ckpt}")
    print()

    _run_prediction(
        config_name=config_name,
        fold_num=fold_num,
        ckpt=ckpt,
    )


@app.command()
def submit(
    fold_num: str = typer.Option(
        "all",
        "--fold-num",
        "-f",
        help="Fold number to package, or 'all'. Default is 'all'.",
    ),
):
    print()
    print("Creating one submission.zip for all experiments")
    print(f"Fold: {fold_num}")
    print()

    _make_submission_zip_all(
        fold_num=fold_num,
    )


if __name__ == "__main__":
    app()