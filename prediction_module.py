"""
Prediction utilities for SSL nnU-Net LightningModule.

Contains:
- _make_predictor
- run_validation_prediction_with_metrics
- run_test_prediction
"""

from pathlib import Path

import numpy as np
import torch

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p,
    save_json,
)

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_raw

from utils import (
    update_actual_validation_progress,
    convert_predictions_to_255,
    write_predictions_json,
    reset_folder,
    remove_folder,
    zip_validation_cases,
    zip_test_cases,
    safe_binary_segmentation_metrics,
)


class PredictionMixin:
    # ------------------------------------------------------------------
    # Progress helpers
    # ------------------------------------------------------------------
    def _update_actual_val_progress(self, stage, current, total):
        """
        Update actual-validation tqdm callback.
        """

        update_actual_validation_progress(
            trainer=self.trainer,
            stage=stage,
            current=current,
            total=total,
        )

    # ------------------------------------------------------------------
    # Predictor
    # ------------------------------------------------------------------
    def _make_predictor(self, net):
        """
        Create nnU-Net predictor from current trained network.

        Used by:
        - run_validation_prediction_with_metrics()
        - run_test_prediction()
        """

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )

        predictor.plans_manager = self.pm
        predictor.configuration_manager = self.cm
        predictor.network = net
        predictor.dataset_json = self.dataset_json
        predictor.trainer_name = self.__class__.__name__

        predictor.allowed_mirroring_axes = tuple(
            range(len(self.cm.patch_size))
        )

        predictor.label_manager = self.lm
        predictor.list_of_parameters = [
            net.state_dict()
        ]

        return predictor

    # ------------------------------------------------------------------
    # Custom MedPy metrics
    # ------------------------------------------------------------------
    def _compute_custom_metrics_from_saved_predictions(self):
        dataset_val = self.trainer.datamodule.dataset_val
        val_keys = list(dataset_val.identifiers)

        labels = self.lm.foreground_labels
        file_ending = self.dataset_json["file_ending"]
        rw = self.pm.image_reader_writer_class()

        results = {
            "unit": {
                "Dice": "unitless",
                "ASD_mm": "mm",
                "HD_mm": "mm",
                "HD95_mm": "mm",
            },
            "cases": {},
            "mean": {},
        }

        total_cases = len(val_keys)

        for i, case_id in enumerate(val_keys, start=1):
            self._update_actual_val_progress(
                stage="metrics",
                current=i,
                total=total_cases,
            )

            pred_file = join(
                self.actual_validation_tmp_preds_folder,
                f"{case_id}_pred{file_ending}",
            )

            gt_file = join(
                self.gt_folder,
                f"{case_id}{file_ending}",
            )

            pred, pred_props = rw.read_seg(pred_file)
            gt, gt_props = rw.read_seg(gt_file)

            pred = pred[0]
            gt = gt[0]

            voxel_spacing = pred_props["spacing"]
            case_result = {}

            for label in labels:
                pred_mask = pred == label
                gt_mask = gt == label

                case_result[str(label)] = safe_binary_segmentation_metrics(
                    pred_mask,
                    gt_mask,
                    voxel_spacing,
                )

            results["cases"][case_id] = case_result

        metric_names = ["Dice", "ASD_mm", "HD_mm", "HD95_mm"]

        for label in labels:
            label_key = str(label)
            results["mean"][label_key] = {}

            for metric_name in metric_names:
                values = []

                for case_id in val_keys:
                    value = results["cases"][case_id][label_key][metric_name]

                    if value is not None and np.isfinite(value):
                        values.append(value)

                results["mean"][label_key][metric_name] = (
                    float(np.mean(values)) if len(values) > 0 else None
                )

        results["foreground_mean"] = {}

        for metric_name in metric_names:
            values = []

            for label in labels:
                value = results["mean"][str(label)][metric_name]

                if value is not None and np.isfinite(value):
                    values.append(value)

            results["foreground_mean"][metric_name] = (
                float(np.mean(values)) if len(values) > 0 else None
            )

        output_json = join(
            self.actual_validation_output_folder,
            "metrics.json",
        )

        save_json(
            results,
            output_json,
            sort_keys=False,
        )

        return results

    # ------------------------------------------------------------------
    # Validation prediction with metrics
    # ------------------------------------------------------------------
    def run_validation_prediction_with_metrics(self):
        """
        Full actual validation prediction using preprocessed nnU-Net validation data.

        This does:
        - prediction
        - export
        - zip image/pred/gt
        - custom metrics

        This is for labeled validation cases only.
        """

        maybe_mkdir_p(self.actual_validation_output_folder)
        reset_folder(self.actual_validation_cases_folder)
        reset_folder(self.actual_validation_tmp_preds_folder)

        save_json(
            self.dataset_json,
            join(self.actual_validation_output_folder, "dataset.json"),
            sort_keys=False,
        )

        save_json(
            self.pm.plans,
            join(self.actual_validation_output_folder, "plans.json"),
            sort_keys=False,
        )

        dataset_val = self.trainer.datamodule.dataset_val
        val_keys = list(dataset_val.identifiers)

        net = self.network

        if hasattr(net, "module"):
            net = net.module

        old_deep_supervision = net.decoder.deep_supervision
        net.decoder.deep_supervision = False

        predictor = self._make_predictor(net)
        custom_metrics = {}

        try:
            total_cases = len(val_keys)

            for i, case_id in enumerate(val_keys, start=1):
                self._update_actual_val_progress(
                    stage="prediction",
                    current=i,
                    total=total_cases,
                )

                data, _, seg_prev, properties = dataset_val.load_case(case_id)

                if self.cm.previous_stage_name is not None:
                    raise RuntimeError(
                        f"Configuration {self.configuration} is cascaded, "
                        "but this SSL LightningModule does not support "
                        "cascaded inference."
                    )

                data = torch.from_numpy(data[:])

                prediction = predictor.predict_sliding_window_return_logits(
                    data
                )
                prediction = prediction.cpu()

                output_filename_truncated = join(
                    self.actual_validation_tmp_preds_folder,
                    f"{case_id}_pred",
                )

                export_prediction_from_logits(
                    prediction,
                    properties,
                    self.cm,
                    self.pm,
                    self.dataset_json,
                    output_filename_truncated,
                    False,
                )

            zip_validation_cases(
                val_keys=val_keys,
                raw_dataset_dir=Path(nnUNet_raw) / self.dataset_name,
                prediction_folder=self.actual_validation_tmp_preds_folder,
                gt_folder=self.gt_folder,
                zip_dir=self.actual_validation_cases_folder,
                file_ending=self.dataset_json["file_ending"],
                progress_fn=self._update_actual_val_progress,
            )

            custom_metrics = (
                self._compute_custom_metrics_from_saved_predictions()
            )

            for metric_name in ["Dice", "ASD_mm", "HD_mm", "HD95_mm"]:
                value = custom_metrics["foreground_mean"][metric_name]

                self.log(
                    f"{metric_name.lower()}",
                    value,
                    prog_bar=metric_name in ["Dice", "ASD_mm", "HD95_mm"],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )

        finally:
            net.decoder.deep_supervision = old_deep_supervision
            compute_gaussian.cache_clear()
            remove_folder(self.actual_validation_tmp_preds_folder)

        return custom_metrics

    # ------------------------------------------------------------------
    # Test prediction only
    # ------------------------------------------------------------------
    def run_test_prediction(
        self,
        save_probabilities=False,
        overwrite=True,
    ):
        """
        Run prediction on raw nnU-Net imagesTs cases.

        No metrics.
        No GT.

        Uses raw files from:
            nnUNet_raw/<dataset_name>/imagesTs

        Saves:
            output_folder/*.nii.gz
                Predicted masks, converted to 0/255.

            output_folder/<task_id>_predictions.json
                Submission JSON.

            output_folder/cases/*.zip
                One zip per test case containing:
                    image_<raw image filename>
                    <prediction filename>
        """

        input_folder = Path(nnUNet_raw) / self.dataset_name / "imagesTs"
        output_folder = Path(self.actual_prediction_output_folder)

        if not input_folder.exists():
            raise FileNotFoundError(
                f"Could not find raw test folder: {input_folder}"
            )

        maybe_mkdir_p(str(output_folder))

        if overwrite:
            reset_folder(str(output_folder))

        maybe_mkdir_p(str(output_folder))

        net = self.network

        if hasattr(net, "module"):
            net = net.module

        old_deep_supervision = net.decoder.deep_supervision
        net.decoder.deep_supervision = False

        predictor = self._make_predictor(net)

        try:
            if self.cm.previous_stage_name is not None:
                raise RuntimeError(
                    f"Configuration {self.configuration} is cascaded, "
                    "but this SSL LightningModule does not support "
                    "cascaded inference."
                )

            predictor.predict_from_files_sequential(
                str(input_folder),
                str(output_folder),
                save_probabilities=save_probabilities,
                overwrite=overwrite,
                folder_with_segs_from_prev_stage=None,
            )

            convert_predictions_to_255(output_folder)

            write_predictions_json(
                output_folder=str(output_folder),
                task_id=self.task_id,
            )

            zip_test_cases(
                raw_dataset_dir=Path(nnUNet_raw) / self.dataset_name,
                prediction_folder=self.actual_prediction_output_folder,
                zip_dir=self.actual_prediction_cases_folder,
                file_ending=self.dataset_json["file_ending"],
                progress_fn=self._update_actual_val_progress,
            )

        finally:
            net.decoder.deep_supervision = old_deep_supervision
            compute_gaussian.cache_clear()
