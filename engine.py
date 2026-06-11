"""
Training / prediction / submission engine for SSL nnU-Net experiments.

Usage:
    python engine.py train video
    python engine.py train tee
    python engine.py train ct

    python engine.py predict video
    python engine.py predict tee
    python engine.py predict ct
    python engine.py predict all

    python engine.py predict video --ckpt best
    python engine.py predict video --ckpt last
    python engine.py predict video --ckpt /path/to/checkpoint.ckpt

    python engine.py predict all --ckpt best
    python engine.py predict all --ckpt last

    python engine.py submit
"""

from pathlib import Path
import zipfile

import typer
import lightning as L

from hydra.utils import instantiate
from omegaconf import OmegaConf

from lightning.pytorch.strategies import DDPStrategy

try:
    from lightning.fabric.plugins.environments import LightningEnvironment
except ImportError:
    from lightning.pytorch.plugins.environments import LightningEnvironment

from config import build_config

from utils import (
    set_nnunet_env,
    resolve_runtime_config,
    clear_checkpoint_dir,
    validate_experiment_name,
    collect_submission_files,
    resolve_prediction_ckpt,
)


app = typer.Typer()


FOLD_NUM = "all"
CKPT = "best"


CONFIG_MAP = {
    "ct": "experiment_CT",
    "tee": "experiment_TEE",
    "video": "experiment_video",
}


def _use_plain_lightning_ddp(
    trainer_cfg,
):
    """
    Force plain Lightning DDP for non-managed clusters.
    """

    strategy = trainer_cfg.get(
        "strategy",
        "auto",
    )

    if strategy != "ddp":
        return trainer_cfg

    trainer_cfg["num_nodes"] = 1

    trainer_cfg["strategy"] = DDPStrategy(
        cluster_environment=LightningEnvironment(),
    )

    return trainer_cfg


def _build_trainer(
    cfg,
    prediction=False,
):
    """
    Build Lightning Trainer.

    Keep ModelCheckpoint callback also during prediction.
    For prediction, ckpt='best'/'last' is resolved manually before trainer.test().
    """

    trainer_cfg = OmegaConf.to_container(
        cfg.trainer,
        resolve=True,
    )

    trainer_cfg = _use_plain_lightning_ddp(
        trainer_cfg,
    )

    callbacks = []

    if trainer_cfg.get("enable_progress_bar", True):
        if "progress_bar" in cfg:
            callbacks.append(
                instantiate(cfg.progress_bar)
            )

    if trainer_cfg.get("enable_checkpointing", True):
        callbacks.append(
            instantiate(cfg.checkpoint)
        )

    trainer = L.Trainer(
        **trainer_cfg,
        callbacks=callbacks,
    )

    return trainer


def _build_objects(
    config_name,
    prediction=False,
    clear_checkpoints=False,
):
    cfg = build_config(
        config_name=config_name,
        overrides=[
            f"fold={FOLD_NUM}",
        ],
    )

    set_nnunet_env(
        cfg,
    )

    L.seed_everything(
        cfg.seed,
        workers=True,
    )

    from datamodule import SSLnnUNetDataModule
    from module import SSLnnUNetLightningModule

    cfg = resolve_runtime_config(
        cfg,
        prediction=prediction,
    )

    if not prediction and clear_checkpoints:
        clear_checkpoint_dir(
            cfg,
        )

    datamodule = SSLnnUNetDataModule(
        cfg.datamodule,
    )

    model = SSLnnUNetLightningModule(
        cfg.litmodule,
    )

    trainer = _build_trainer(
        cfg,
        prediction=prediction,
    )

    return cfg, datamodule, model, trainer


def _run_training(
    config_name,
    clear_checkpoints=True,
):
    cfg, datamodule, model, trainer = _build_objects(
        config_name=config_name,
        prediction=False,
        clear_checkpoints=clear_checkpoints,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


def _run_prediction(
    config_name,
    ckpt=CKPT,
):
    cfg, datamodule, model, trainer = _build_objects(
        config_name=config_name,
        prediction=True,
        clear_checkpoints=False,
    )

    resolved_ckpt = resolve_prediction_ckpt(
        cfg=cfg,
        ckpt=ckpt,
    )

    print()
    print("[predict] Using checkpoint:")
    print(f"  requested: {ckpt}")
    print(f"  resolved : {resolved_ckpt}")
    print()

    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=resolved_ckpt,
    )


def _run_prediction_all(
    ckpt=CKPT,
):
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

        _run_prediction(
            config_name=config_name,
            ckpt=ckpt,
        )


@app.command()
def train(
    experiment: str = typer.Argument(
        ...,
        help="Which experiment to train: ct, tee, or video.",
    ),
    clear_checkpoints: bool = typer.Option(
        True,
        "--clear-checkpoints/--keep-checkpoints",
        help="Clear checkpoint directory before training. Default is true.",
    ),
):
    try:
        experiment, config_name = validate_experiment_name(
            experiment,
            CONFIG_MAP,
        )
    except ValueError as e:
        raise typer.BadParameter(
            str(e)
        )

    print()
    print(f"Training experiment: {experiment}")
    print(f"Config: {config_name}")
    print(f"Fold: {FOLD_NUM}")
    print(f"Clear checkpoints: {clear_checkpoints}")
    print()

    _run_training(
        config_name=config_name,
        clear_checkpoints=clear_checkpoints,
    )


@app.command()
def predict(
    experiment: str = typer.Argument(
        ...,
        help="Which experiment to predict: ct, tee, video, or all.",
    ),
    ckpt: str = typer.Option(
        CKPT,
        "--ckpt",
        help="Checkpoint to use: best, last, or full .ckpt path.",
    ),
):
    experiment = str(
        experiment
    ).lower().strip()

    if experiment == "all":
        _run_prediction_all(
            ckpt=ckpt,
        )

        return

    try:
        experiment, config_name = validate_experiment_name(
            experiment,
            CONFIG_MAP,
        )
    except ValueError as e:
        raise typer.BadParameter(
            str(e)
        )

    print()
    print(f"Predicting experiment: {experiment}")
    print(f"Config: {config_name}")
    print(f"Fold: {FOLD_NUM}")
    print(f"Checkpoint: {ckpt}")
    print()

    _run_prediction(
        config_name=config_name,
        ckpt=ckpt,
    )


@app.command()
def submit():
    """
    Create one submission.zip for all experiments.

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
            └── *.png
    """

    print()
    print("Creating one submission.zip for all experiments")
    print(f"Fold: {FOLD_NUM}")
    print()

    output_zip = Path(__file__).resolve().parent / "submission.zip"

    if output_zip.exists():
        output_zip.unlink()

    all_items = []

    for experiment, config_name in CONFIG_MAP.items():
        cfg = build_config(
            config_name=config_name,
            overrides=[
                f"fold={FOLD_NUM}",
            ],
        )

        set_nnunet_env(
            cfg,
        )

        item = collect_submission_files(
            cfg=cfg,
            fold_num=FOLD_NUM,
        )

        item["experiment"] = experiment

        all_items.append(
            item
        )

    with zipfile.ZipFile(
        output_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        for item in all_items:
            prefix = item["prefix"]
            json_file = item["json_file"]
            prediction_files = item["prediction_files"]

            zf.write(
                json_file,
                arcname=f"{prefix}/{json_file.name}",
            )

            for prediction_file in prediction_files:
                zf.write(
                    prediction_file,
                    arcname=f"{prefix}/{prediction_file.name}",
                )

    print()
    print("[submission] Created single submission zip:")
    print(f"  {output_zip}")
    print()

    for item in all_items:
        print(f"[submission] {item['experiment']}")
        print(f"  folder inside zip: {item['prefix']}/")
        print(f"  json: {item['json_file'].name}")
        print(f"  prediction files: {len(item['prediction_files'])}")
        print()


if __name__ == "__main__":
    app()
