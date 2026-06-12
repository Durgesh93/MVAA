"""
PredictionMixin for SSL nnU-Net.

Takes a prepared case from the datamodule, runs sliding-window inference,
restores prediction to original image shape, applies task-specific
post-processing, and returns the prediction dictionary.
"""

import numpy as np
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
)

from utils import to_numpy


class PredictionMixin:


    def _unwrap_network(self):
        net = self.network

        if hasattr(net, "module"):
            net = net.module

        return net

    def _make_predictor(self, net):
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
        predictor.allowed_mirroring_axes = tuple(range(len(self.cm.patch_size)))
        predictor.label_manager = self.lm
        predictor.list_of_parameters = [net.state_dict()]

        return predictor

    def _predict_logits(self, data):
        if self.cm.previous_stage_name is not None:
            raise RuntimeError(
                f"Configuration {self.configuration} is cascaded, "
                "but this module does not support cascaded inference."
            )

        net = self._unwrap_network()

        old_deep_supervision = net.decoder.deep_supervision
        net.decoder.deep_supervision = False

        predictor = self._make_predictor(net)

        try:
            logits = predictor.predict_sliding_window_return_logits(data)
        finally:
            net.decoder.deep_supervision = old_deep_supervision
            compute_gaussian.cache_clear()

        return logits

    def _restore_prediction_shape(self, logits, properties):
        predicted_segments, predicted_probs = (
            convert_predicted_logits_to_segmentation_with_correct_shape(
                predicted_logits=logits.cpu(),
                plans_manager=self.pm,
                configuration_manager=self.cm,
                label_manager=self.lm,
                properties_dict=properties,
                return_probabilities=True,
            )
        )

        return predicted_segments, predicted_probs

    # =========================================================================
    # Task 3: VIDEO post-processing
    # =========================================================================
    def _postprocess_t3_vid(
        self,
        predicted_segments,
        predicted_probs
    ):
        """
        Task 3 VIDEO post-processing.

        Binary assumption:
            probs[0] = background
            probs[1] = foreground

        Handles:
            predicted_probs:
                [2, H, W]
                [2, 1, H, W]

            predicted_segments:
                [H, W]
                [1, H, W]
        """

        import cv2
        import numpy as np

        probs = np.asarray(
            predicted_probs,
            dtype=np.float32,
        ).copy()

        seg_original_shape = predicted_segments.shape

        seg = np.asarray(
            predicted_segments,
            dtype=np.uint8,
        )

        # ------------------------------------------------------------
        # Force segmentation to 2D for cv2
        # ------------------------------------------------------------
        seg_2d = np.squeeze(seg)

        if seg_2d.ndim != 2:
            raise ValueError(
                f"Task 3 post-processing expects 2D segmentation after squeeze, "
                f"got shape {seg.shape} -> {seg_2d.shape}"
            )

        # ------------------------------------------------------------
        # Foreground probability channel
        # Supports [2, H, W] and [2, 1, H, W]
        # ------------------------------------------------------------
        fg_prob = probs[1]
        fg_prob_2d = np.squeeze(fg_prob)

        if fg_prob_2d.ndim != 2:
            raise ValueError(
                f"Task 3 post-processing expects 2D foreground probability "
                f"after squeeze, got shape {fg_prob.shape} -> {fg_prob_2d.shape}"
            )

        prob_threshold = 0.10

        # ------------------------------------------------------------
        # Low-threshold foreground evidence
        # ------------------------------------------------------------
        candidate = fg_prob_2d >= prob_threshold

        # Also keep existing hard prediction if any exists
        candidate = np.logical_or(
            candidate,
            seg_2d == 1,
        )

        if int(candidate.sum()) == 0:
            post_seg_2d = np.zeros_like(
                seg_2d,
                dtype=np.uint8,
            )

            probs[1] = 0.0
            probs[0] = 1.0

            post_seg = post_seg_2d.reshape(
                seg_original_shape,
            )

            return post_seg.astype(np.uint8), probs

        candidate_uint8 = np.ascontiguousarray(
            candidate.astype(np.uint8) * 255,
        )

        # ------------------------------------------------------------
        # Connect weak dots/blobs
        # ------------------------------------------------------------
        kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )

        candidate_uint8 = cv2.dilate(
            candidate_uint8,
            kernel_dilate,
            iterations=1,
        )

        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7),
        )

        candidate_uint8 = cv2.morphologyEx(
            candidate_uint8,
            cv2.MORPH_CLOSE,
            kernel_close,
            iterations=1,
        )

        # ------------------------------------------------------------
        # Convex hull around all foreground evidence
        # ------------------------------------------------------------
        ys, xs = np.where(
            candidate_uint8 > 0,
        )

        if len(xs) < 3:
            post_seg_2d = np.zeros_like(
                seg_2d,
                dtype=np.uint8,
            )

            probs[1] = 0.0
            probs[0] = 1.0

            post_seg = post_seg_2d.reshape(
                seg_original_shape,
            )

            return post_seg.astype(np.uint8), probs

        points = np.stack(
            [xs, ys],
            axis=1,
        ).astype(np.int32)

        hull = cv2.convexHull(
            points,
        )

        hull_mask = np.zeros_like(
            candidate_uint8,
            dtype=np.uint8,
        )

        cv2.fillConvexPoly(
            hull_mask,
            hull,
            1,
        )

        hull_mask = cv2.morphologyEx(
            hull_mask,
            cv2.MORPH_CLOSE,
            kernel_close,
            iterations=1,
        )

        hull_mask = np.ascontiguousarray(
            hull_mask.astype(np.uint8),
        )

        # ------------------------------------------------------------
        # Final segmentation
        # ------------------------------------------------------------
        post_seg_2d = np.zeros_like(
            seg_2d,
            dtype=np.uint8,
        )

        post_seg_2d[hull_mask > 0] = 1

        post_seg = post_seg_2d.reshape(
            seg_original_shape,
        )

        # ------------------------------------------------------------
        # Update probabilities.
        # Keep the original probability shape.
        # ------------------------------------------------------------
        hull_float_2d = hull_mask.astype(np.float32)

        if probs.ndim == 3:
            probs[1] = probs[1] * hull_float_2d

            probs[1][hull_mask > 0] = np.maximum(
                probs[1][hull_mask > 0],
                prob_threshold,
            )

            probs[0] = 1.0 - probs[1]
            probs[0] = np.clip(probs[0], 0.0, 1.0)

        elif probs.ndim == 4 and probs.shape[1] == 1:
            probs[1, 0] = probs[1, 0] * hull_float_2d

            probs[1, 0][hull_mask > 0] = np.maximum(
                probs[1, 0][hull_mask > 0],
                prob_threshold,
            )

            probs[0, 0] = 1.0 - probs[1, 0]
            probs[0, 0] = np.clip(probs[0, 0], 0.0, 1.0)

        else:
            raise ValueError(
                f"Task 3 post-processing expects probs shape [2,H,W] "
                f"or [2,1,H,W], got {probs.shape}"
            )

        return post_seg.astype(np.uint8), probs
        
    # =========================================================================
    # Task dispatcher
    # =========================================================================

    def _postprocess_prediction_by_task(
        self,
        predicted_probs,
        predicted_segments,
    ):
        prefix = str(
            getattr(self, "prefix", "")
        ).lower()

        if prefix.startswith("t3_vid") or prefix.startswith("vid"):
            return self._postprocess_t3_vid(
                predicted_probs=predicted_probs,
                predicted_segments=predicted_segments,
            )

        return predicted_segments, predicted_probs

    # =========================================================================
    # Main prediction entry point
    # =========================================================================
    def run_prediction(self, batch, batch_idx=None):
        item = batch[0]

        logits = self._predict_logits(
            data=item["data"],
        )

        predicted_segments, predicted_probs = self._restore_prediction_shape(
            logits=logits,
            properties=item["properties"],
        )

        predicted_segments_pp, predicted_probs_pp = self._postprocess_prediction_by_task(
            predicted_probs=predicted_probs,
            predicted_segments=predicted_segments,
        )

        item.update(
            {
                "logits": logits,

                # Original nnU-Net output for debugging
                "predicted_segments_raw": predicted_segments,
                "predicted_probs_raw": predicted_probs,

                # Post-processed output used by export/zip
                "predicted_segments": predicted_segments_pp,
                "predicted_probs": predicted_probs_pp,
            }
        )

        return item