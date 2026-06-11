"""
LightningModule for SSL nnU-Net.

Folder structure:
    fold_all/
    ├── validation/          # Slicer MRML case zips, includes gt
    ├── Test/                # Slicer MRML case zips, no gt
    ├── submission/          # *_pred.nii.gz / *_pred.png + task_id_predictions.json
    ├── _rank_outputs/       # temporary DDP rank-local outputs
    ├── checkpoints/
    └── training_progress.png
"""

from typing import Any, Dict

import torch
import lightning as L

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
)

from nnunetv2.paths import (
    nnUNet_preprocessed,
    nnUNet_results,
)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnunetv2.utilities.dataset_name_id_conversion import (
    maybe_convert_to_dataset_name,
)

from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from prediction_module import PredictionMixin
from metrics_module import MetricsMixin

from utils import (
    to_tensor,
    write_prediction_case_zip,
    write_submission_prediction,
    get_rank_output_folder,
    merge_rank_folders,
    cleanup_rank_outputs,
)


class SSLnnUNetLightningModule(
    PredictionMixin,
    MetricsMixin,
    L.LightningModule,
):
    def __init__(self, litmodule_cfg):
        super().__init__()

        self.cfg = litmodule_cfg

        self.dataset_id = self.cfg.dataset_id
        self.configuration = self.cfg.configuration
        self.fold = self.cfg.fold
        self.seed = self.cfg.seed
        self.plans_identifier = self.cfg.plans_identifier

        self.initial_lr = self.cfg.initial_lr
        self.weight_decay = self.cfg.weight_decay
        self.num_epochs = self.cfg.num_epochs

        self.enable_deep_supervision = True

        self.lambda_pseudo = self.cfg.lambda_pseudo
        self.pseudo_threshold = self.cfg.pseudo_threshold

        self.task_id = self.cfg.task_id
        self.prefix = self.cfg.prefix

        # ------------------------------------------------------------
        # Task-specific submission format
        # ------------------------------------------------------------
        self.is_t3_vid = str(self.prefix) == "t3_vid"

        # Task 3 is binary, so PNG submission should be 0/255.
        # Other tasks keep normal label values.
        self.convert_to_255 = self.is_t3_vid

        # Task 3 Codabench submission uses PNG.
        # CT/TEE stay as NIfTI.
        self.submission_output_format = (
            "png" if self.is_t3_vid else "nii.gz"
        )

        self.dataset_name = maybe_convert_to_dataset_name(
            self.dataset_id
        )

        self.base = join(
            nnUNet_preprocessed,
            self.dataset_name,
        )

        self.plans = load_json(
            join(
                self.base,
                self.plans_identifier + ".json",
            )
        )

        self.dataset_json = load_json(
            join(
                self.base,
                "dataset.json",
            )
        )

        self.pm = PlansManager(self.plans)

        self.cm = self.pm.get_configuration(
            self.configuration
        )

        self.lm = self.pm.get_label_manager(
            self.dataset_json
        )

        self.num_input_channels = determine_num_input_channels(
            self.pm,
            self.cm,
            self.dataset_json,
        )

        self.actual_validation_output_base = join(
            nnUNet_results,
            self.dataset_name,
            self.plans_identifier + "__" + self.configuration,
        )

        self.fold_output_folder = join(
            self.actual_validation_output_base,
            f"fold_{self.fold}",
        )

        self.actual_validation_output_folder = join(
            self.fold_output_folder,
            "validation",
        )

        self.actual_prediction_output_folder = join(
            self.fold_output_folder,
            "prediction",
        )

        self.actual_submission_output_folder = join(
            self.fold_output_folder,
            "submission",
        )

        self.progress_png_file = join(
            self.fold_output_folder,
            "training_progress.png",
        )

        self._init_metric_tracking()

        self.network = None
        self.loss = None
        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def _ensure_built(self):
        if self._built:
            return

        self.network = self._build_network()
        self.loss = self._build_loss()

        self._built = True

    def setup(self, stage=None):
        self._ensure_built()

    # ------------------------------------------------------------------
    # DDP rank-local output
    # ------------------------------------------------------------------
    def _rank_info(self):
        return (
            int(self.trainer.global_rank),
            int(self.trainer.world_size),
            bool(self.trainer.is_global_zero),
        )

    def _rank_output_folder(self):
        global_rank, _, _ = self._rank_info()

        return get_rank_output_folder(
            fold_output_folder=self.fold_output_folder,
            global_rank=global_rank,
        )

    def _barrier(self, name=None):
        if int(self.trainer.world_size) <= 1:
            return

        if name is None:
            self.trainer.strategy.barrier()
        else:
            self.trainer.strategy.barrier(name)

    def _merge_rank_outputs(
        self,
        require_submission,
        cleanup=True,
    ):
        """
        Simple DDP-safe merge.

        Order:
            1. all ranks finish writing
            2. barrier
            3. rank 0 merges rank folders
            4. barrier
            5. rank 0 deletes _rank_outputs

        We do not need done marker files here.
        """

        trainer = self.trainer
        _, _, is_global_zero = self._rank_info()

        self._barrier(
            name="before_merge_rank_outputs",
        )

        if is_global_zero:
            merge_rank_folders(
                fold_output_folder=self.fold_output_folder,
                task_id=self.task_id,
                overwrite=True,
                require_submission=require_submission,
            )

        self._barrier(
            name="after_merge_rank_outputs",
        )

        if cleanup and is_global_zero:
            cleanup_rank_outputs(
                fold_output_folder=self.fold_output_folder,
            )

    # ------------------------------------------------------------------
    # nnU-Net shim
    # ------------------------------------------------------------------
    def _make_trainer_shim(self):
        shim = type("S", (), {})()

        shim.configuration_manager = self.cm
        shim.label_manager = self.lm
        shim.enable_deep_supervision = self.enable_deep_supervision
        shim.is_ddp = int(self.trainer.world_size) > 1

        shim._get_deep_supervision_scales = (
            lambda: nnUNetTrainer._get_deep_supervision_scales(shim)
        )

        shim._do_i_compile = lambda: False

        return shim

    def _build_network(self):
        return nnUNetTrainer.build_network_architecture(
            self.pm,
            self.cm,
            self.num_input_channels,
            self.lm.num_segmentation_heads,
            self.enable_deep_supervision,
        )

    def _build_loss(self):
        shim = self._make_trainer_shim()

        return nnUNetTrainer._build_loss(shim)

    def _unwrap_network(self):
        if hasattr(self.network, "module"):
            return self.network.module

        return self.network

    def forward(self, x):
        return self.network(x)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def _get_train_batch_data_target(
        self,
        batch: Dict[str, Any],
    ):
        data = to_tensor(
            batch["data"],
            device=self.device,
            dtype=torch.float32,
        )

        target = batch["target"]

        if isinstance(target, (list, tuple)):
            target = [
                to_tensor(
                    t,
                    device=self.device,
                )
                for t in target
            ]
        else:
            target = to_tensor(
                target,
                device=self.device,
            )

        return data, target

    def _supervised_loss(
        self,
        batch: Dict[str, Any],
    ):
        data, target = self._get_train_batch_data_target(
            batch
        )

        output = self.network(data)

        loss = self.loss(
            output,
            target,
        )

        return loss, output, target

    def _pseudo_loss(
        self,
        batch: Dict[str, Any],
    ):
        if self.lambda_pseudo <= 0 or batch is None:
            return torch.tensor(
                0.0,
                device=self.device,
            )

        data, _ = self._get_train_batch_data_target(
            batch
        )

        with torch.no_grad():
            pseudo_output = self.network(data)

            if isinstance(pseudo_output, (list, tuple)):
                pseudo_logits = pseudo_output[0]
            else:
                pseudo_logits = pseudo_output

            if self.lm.has_regions:
                probs = torch.sigmoid(
                    pseudo_logits
                )

                pseudo_target = (
                    probs > self.pseudo_threshold
                ).float()

                confident_voxels = (
                    probs > self.pseudo_threshold
                )

            else:
                probs = torch.softmax(
                    pseudo_logits,
                    dim=1,
                )

                conf, pseudo_target = probs.max(
                    dim=1,
                    keepdim=True,
                )

                pseudo_target = pseudo_target.long()

                confident_voxels = (
                    conf >= self.pseudo_threshold
                )

                if self.lm.ignore_label is not None:
                    pseudo_target[~confident_voxels] = (
                        self.lm.ignore_label
                    )

        student_output = self.network(data)

        if isinstance(student_output, (list, tuple)):
            student_logits = student_output[0]
        else:
            student_logits = student_output

        pseudo_loss = self.loss.loss(
            student_logits,
            pseudo_target,
        )

        if confident_voxels.sum() == 0:
            pseudo_loss = pseudo_loss * 0.0

        return pseudo_loss

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    def training_step(
        self,
        batch,
        batch_idx,
    ):
        labeled_batch = batch["labeled"]

        unlabeled_batch = batch.get(
            "unlabeled",
            None,
        )

        sup_loss, _, _ = self._supervised_loss(
            labeled_batch
        )

        pseudo_loss = self._pseudo_loss(
            unlabeled_batch
        )

        total_loss = (
            sup_loss
            + self.lambda_pseudo * pseudo_loss
        )

        self.update_step_training_metrics(
            train_loss=total_loss.detach(),
            train_sup_loss=sup_loss.detach(),
            train_pseudo_loss=pseudo_loss.detach(),
        )

        return total_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if self.trainer.sanity_checking:
            return None

        prediction = self.run_prediction(
            batch=batch,
            batch_idx=batch_idx,
        )

        rank_output_folder = self._rank_output_folder()

        # ------------------------------------------------------------
        # Dataloader 0:
        # validation cases with GT
        # ------------------------------------------------------------
        if dataloader_idx == 0:
            write_prediction_case_zip(
                prediction=prediction,
                zip_dir=rank_output_folder / "validation",
                image_reader_writer=self.pm.image_reader_writer_class,
                configuration_manager=self.cm,
                include_gt=True,
                convert_to_255=self.convert_to_255,
            )

            metrics = self.compute_metrics(
                prediction
            )

            self.update_step_val_metrics(
                metrics=metrics,
            )

            return prediction

        # ------------------------------------------------------------
        # Dataloader 1:
        # test cases without GT
        # ------------------------------------------------------------
        write_prediction_case_zip(
            prediction=prediction,
            zip_dir=rank_output_folder / "prediction",
            image_reader_writer=self.pm.image_reader_writer_class,
            configuration_manager=self.cm,
            include_gt=False,
            convert_to_255=self.convert_to_255,
        )

        write_submission_prediction(
            prediction=prediction,
            output_folder=rank_output_folder / "submission",
            image_reader_writer=self.pm.image_reader_writer_class,
            configuration_manager=self.cm,
            convert_to_255=self.convert_to_255,
            output_format=self.submission_output_format,
        )

        return prediction

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def test_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        return self.validation_step(
            batch,
            batch_idx,
            dataloader_idx,
        )

    # ------------------------------------------------------------------
    # Shared validation/test epoch-end
    # ------------------------------------------------------------------
    def _eval_epoch_end(
        self,
        stage,
        save_training_progress,
        print_val_metrics=False,
    ):
        if self.trainer.sanity_checking:
            self.step_metrics.reset()
            return

        synced_metrics = self.step_metrics.compute()

        self.log_validation_epoch_metrics(
            synced_metrics=synced_metrics,
        )

        if print_val_metrics:
            self.print_validation_epoch_metrics(
                synced_metrics
            )

        # Always update epoch history.
        self._update_epoch_metrics(
            synced_metrics=synced_metrics,
        )

        if self.trainer.is_global_zero:
            if save_training_progress:
                self._save_training_progress_plot()

        self.step_metrics.reset()

        # Always merge rank outputs after every validation/test epoch.
        self._merge_rank_outputs(
            require_submission=True,
            cleanup=True,
        )

    def on_validation_epoch_end(self):
        self._eval_epoch_end(
            stage="validation",
            save_training_progress=True,
        )

    def on_test_epoch_end(self):
        self._eval_epoch_end(
            stage="test",
            save_training_progress=False,
            print_val_metrics=True,
        )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        self._ensure_built()

        shim = type("S", (), {})()

        shim.network = self.network
        shim.initial_lr = self.initial_lr
        shim.weight_decay = self.weight_decay
        shim.num_epochs = self.num_epochs

        optimizer, scheduler = (
            nnUNetTrainer.configure_optimizers(shim)
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
