"""
Docker inference entrypoint.

Called directly by the image's ENTRYPOINT -- no CLI flags at runtime. Per
the submission contract, the organizer's runner only sets
MVAA_INPUT_DIR=/input, MVAA_OUTPUT_DIR=/output (both mounted volumes) and
-w /work (a third mounted, writable scratch volume, set as the container's
cwd -- there is no separate env var for it). Runs all 3 tasks, loading
each one's checkpoint into a Lightning module and running trainer.predict()
over /input, writing masks + task*_predictions.json into /output/<prefix>/.

Training used Lightning, so inference reuses the same LightningModule/
LightningDataModule shape (InferenceLightningModule/InferenceDataModule)
instead of hand-rolling device placement and the predict loop -- Lightning
handles moving the module/batches to the right accelerator, no_grad/eval
mode, etc. Mask writing happens inside predict_step itself (see
module/inference_module.py), which just records the written path (not
accumulated in trainer.predict()'s return value, to avoid holding every
case's full-resolution volume in memory at once) -- turning that into a
task*_predictions.json entry is utils.write_predictions_json's job, kept
in utils.py alongside the other inference-entrypoint helpers so this file
stays pure orchestration.
"""

import os
import shutil
from pathlib import Path

import lightning as L
import torch
from lightning.fabric.plugins.environments import LightningEnvironment

BASE_DIR = Path(__file__).resolve().parent

from datamodule import InferenceDataModule  # noqa: E402
from module import InferenceLightningModule  # noqa: E402
from utils import load_checkpoint, load_task_cfg, write_predictions_json  # noqa: E402

TASKS = ["ct", "tee", "video"]


def run_task(task: str, device: torch.device) -> None:
    cfg = load_task_cfg(task)
    prefix = cfg.prefix
    is_video = prefix == "t3_vid"

    print(f"[{task}] loading plans/checkpoint for {prefix}")

    # Must be set before NNUnetSetup() is constructed. nnunetv2.paths.nnUNet_preprocessed
    # is a lazy os.PathLike wrapper around os.environ, re-read on every access -- so this
    # just needs to run before first use, not before nnunetv2 is imported.
    os.environ["nnUNet_preprocessed"] = str(BASE_DIR / cfg.nnunet_preprocessed)

    # cfg.input_dir/output_dir/work_dir are resolved by load_task_cfg --
    # see config/<task>.yaml.
    image_folder = Path(cfg.input_dir)

    task_output_dir = Path(cfg.output_dir)
    task_output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(cfg.work_dir) / ".temp"

    module = InferenceLightningModule(cfg, output_dir=task_output_dir)

    ckpt_path = BASE_DIR / "ckpts" / prefix / "model.ckpt"
    load_checkpoint(module, ckpt_path)

    dataset_json = module.nnunet.dataset_json
    file_ending = dataset_json["file_ending"]
    num_channels = len(dataset_json["channel_names"])

    datamodule = InferenceDataModule(
        is_video=is_video,
        image_folder=image_folder,
        file_ending=file_ending,
        num_channels=num_channels,
        plans_manager=module.nnunet.pm,
        configuration_manager=module.nnunet.cm,
        dataset_json=dataset_json,
        work_dir=(work_dir / prefix) if is_video else None,
    )

    # This cluster does not launch jobs via srun, so Lightning's automatic
    # cluster-environment detection is unsafe here -- SLURMEnvironment's
    # constructor validates srun variables eagerly and raises whenever
    # SLURM_NTASKS > 1 without SLURM_NTASKS_PER_NODE (see engine.py's own
    # _select_cluster_environment for the same workaround during training).
    trainer = L.Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        num_nodes=1,
        plugins=[LightningEnvironment()],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    trainer.predict(model=module, datamodule=datamodule, return_predictions=False)

    json_path = write_predictions_json(module.pred_files, task_output_dir, cfg.task_id)

    print(f"[{task}] wrote {len(module.pred_files)} cases to {json_path}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        for task in TASKS:
            run_task(task, device)
    finally:
        # Scratch (video frame staging only) lives under cfg.work_dir/.temp --
        # same fixed /work path for every task's yaml, safe to remove whether
        # or not every task succeeded.
        shutil.rmtree(Path("/work") / ".temp", ignore_errors=True)


if __name__ == "__main__":
    main()
