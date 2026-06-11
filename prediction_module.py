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
    # Shared binary helpers
    # =========================================================================

    def _get_binary_foreground_prob(
        self,
        predicted_probs,
        predicted_segments,
    ):
        """
        Binary probability handling.

        Supported probability shapes:

            [H, W]
            [D, H, W]
            [1, H, W]
            [1, D, H, W]
            [2, H, W]
            [2, D, H, W]

        For [2, ...], channel 1 is treated as foreground.
        For [1, ...], channel 0 is treated as foreground.
        """

        probs = to_numpy(predicted_probs)
        seg = to_numpy(predicted_segments)

        probs = np.asarray(
            probs,
            dtype=np.float32,
        )

        seg = np.asarray(seg)

        if seg.ndim >= 3 and seg.shape[0] == 1:
            seg = seg[0]

        probs_had_channel = True

        if probs.ndim == seg.ndim:
            probs = probs[None]
            probs_had_channel = False

        if probs.ndim != seg.ndim + 1:
            return None, None, seg.astype(np.uint8), probs, probs_had_channel

        if probs.shape[0] == 1:
            fg_channel = 0
        else:
            fg_channel = 1

        fg_prob = probs[fg_channel]

        return (
            fg_prob,
            fg_channel,
            seg.astype(np.uint8),
            probs,
            probs_had_channel,
        )

    def _restore_probs_shape(
        self,
        probs,
        probs_had_channel,
    ):
        if probs_had_channel:
            return probs

        return probs[0]

    # =========================================================================
    # SimpleITK 3D morphology helpers
    # =========================================================================

    def _sitk_binary_closing(
        self,
        mask,
        radius=1,
        iterations=1,
    ):
        """
        3D/2D binary closing using SimpleITK.

        Closing smooths walls and closes small gaps.
        """

        import SimpleITK as sitk

        mask = np.asarray(mask).astype(np.uint8)

        if mask.sum() == 0:
            return mask.astype(bool)

        img = sitk.GetImageFromArray(mask)

        for _ in range(iterations):
            f = sitk.BinaryMorphologicalClosingImageFilter()
            f.SetForegroundValue(1)
            f.SetKernelType(sitk.sitkBall)
            f.SetKernelRadius([radius] * mask.ndim)

            img = f.Execute(img)

        out = sitk.GetArrayFromImage(img) > 0

        return out

    def _sitk_fill_small_holes_only(
        self,
        mask,
        max_hole_size=500,
        max_hole_fraction=0.002,
    ):
        """
        Fill only small accidental holes.

        Important:
            This does NOT fill the large anatomical mitral valve opening.
        """

        import SimpleITK as sitk

        mask = np.asarray(mask).astype(bool)

        if mask.sum() == 0:
            return mask

        mask_img = sitk.GetImageFromArray(
            mask.astype(np.uint8)
        )

        fill_filter = sitk.BinaryFillholeImageFilter()
        fill_filter.SetForegroundValue(1)

        filled_img = fill_filter.Execute(mask_img)

        filled = sitk.GetArrayFromImage(filled_img) > 0

        holes = filled & (~mask)

        if holes.sum() == 0:
            return mask

        holes_img = sitk.GetImageFromArray(
            holes.astype(np.uint8)
        )

        cc_img = sitk.ConnectedComponent(holes_img)

        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(cc_img)

        cc_arr = sitk.GetArrayFromImage(cc_img)

        fraction_limit = int(
            mask.size * max_hole_fraction
        )

        hole_limit = min(
            max_hole_size,
            max(1, fraction_limit),
        )

        fill_ids = []

        for label_id in stats.GetLabels():
            size = int(
                stats.GetNumberOfPixels(label_id)
            )

            if size <= hole_limit:
                fill_ids.append(label_id)

        if len(fill_ids) == 0:
            return mask

        small_holes = np.isin(
            cc_arr,
            fill_ids,
        )

        out = mask.copy()
        out[small_holes] = True

        return out

    def _sitk_keep_largest_component(
        self,
        mask,
        min_component_size=100,
    ):
        """
        Keep only largest connected foreground component.
        """

        import SimpleITK as sitk

        mask = np.asarray(mask).astype(bool)

        if mask.sum() == 0:
            return np.zeros_like(mask, dtype=bool)

        img = sitk.GetImageFromArray(
            mask.astype(np.uint8)
        )

        cc_img = sitk.ConnectedComponent(img)

        relabel_img = sitk.RelabelComponent(
            cc_img,
            sortByObjectSize=True,
        )

        relabel = sitk.GetArrayFromImage(relabel_img)

        largest = relabel == 1

        if int(largest.sum()) < min_component_size:
            return np.zeros_like(mask, dtype=bool)

        return largest

    # =========================================================================
    # Task 1: CT post-processing
    # =========================================================================

    def _postprocess_t1_ct(
        self,
        predicted_probs,
        predicted_segments,
    ):
        """
        Task 1 CT.

        Foreground is expected.

        Uses 3D medical-image morphology:
            - threshold foreground probability
            - fallback adaptive threshold if weak
            - 3D closing for smoother wall
            - fill only small holes
            - keep largest coherent component
        """

        fg_prob, fg_channel, seg, probs, probs_had_channel = (
            self._get_binary_foreground_prob(
                predicted_probs=predicted_probs,
                predicted_segments=predicted_segments,
            )
        )

        if fg_prob is None:
            return seg.astype(np.uint8), predicted_probs

        if fg_prob.ndim != 3:
            return seg.astype(np.uint8), self._restore_probs_shape(
                probs,
                probs_had_channel,
            )

        mask = fg_prob >= 0.50

        if int(mask.sum()) < 150:
            peak = float(
                np.max(fg_prob)
            )

            if peak > 0:
                adaptive_threshold = max(
                    0.25,
                    peak * 0.45,
                )

                mask = fg_prob >= adaptive_threshold

        mask = self._sitk_binary_closing(
            mask=mask,
            radius=1,
            iterations=2,
        )

        mask = self._sitk_fill_small_holes_only(
            mask=mask,
            max_hole_size=800,
            max_hole_fraction=0.002,
        )

        mask = self._sitk_keep_largest_component(
            mask=mask,
            min_component_size=150,
        )

        # Fallback to original prediction if probability-based result vanished.
        if int(mask.sum()) == 0:
            original_mask = seg == 1

            original_mask = self._sitk_binary_closing(
                mask=original_mask,
                radius=1,
                iterations=1,
            )

            original_mask = self._sitk_fill_small_holes_only(
                mask=original_mask,
                max_hole_size=800,
                max_hole_fraction=0.002,
            )

            mask = self._sitk_keep_largest_component(
                mask=original_mask,
                min_component_size=150,
            )

        post_seg = np.zeros_like(
            seg,
            dtype=np.uint8,
        )

        post_seg[mask] = 1

        return post_seg.astype(np.uint8), self._restore_probs_shape(
            probs,
            probs_had_channel,
        )

    # =========================================================================
    # Task 2: TEE post-processing
    # =========================================================================

    def _postprocess_t2_tee(
        self,
        predicted_probs,
        predicted_segments,
    ):
        """
        Task 2 TEE.

        TEE has two foreground classes.

        Expected probability shape:
            [C, D, H, W]

        Expected labels:
            0 = background
            1 = foreground class 1
            2 = foreground class 2

        This post-processes each foreground class separately:
            - threshold class probability
            - 3D closing
            - fill only small holes
            - keep largest component
            - combine labels back without collapsing class 2
        """

        seg = to_numpy(predicted_segments)
        probs = to_numpy(predicted_probs)

        seg = np.asarray(seg)
        probs = np.asarray(probs, dtype=np.float32)

        if seg.ndim >= 3 and seg.shape[0] == 1:
            seg = seg[0]

        # Need [C, D, H, W] for two foreground classes.
        if probs.ndim != seg.ndim + 1:
            return seg.astype(np.uint8), predicted_probs

        if seg.ndim != 3:
            return seg.astype(np.uint8), predicted_probs

        # Use label manager if available, otherwise use [1, 2].
        foreground_labels = []

        for label in list(getattr(self.lm, "foreground_labels", [1, 2])):
            try:
                label = int(label)
            except Exception:
                continue

            if label > 0 and label < probs.shape[0]:
                foreground_labels.append(label)

        if len(foreground_labels) == 0:
            foreground_labels = [
                label for label in [1, 2]
                if label < probs.shape[0]
            ]

        if len(foreground_labels) == 0:
            return seg.astype(np.uint8), predicted_probs

        masks_by_label = {}

        for label in foreground_labels:
            class_prob = probs[label]

            # ------------------------------------------------------------
            # Threshold class probability
            # ------------------------------------------------------------
            mask = class_prob >= 0.10

            if int(mask.sum()) < 80:
                peak = float(np.max(class_prob))

                if peak > 0:
                    adaptive_threshold = max(
                        0.20,
                        peak * 0.40,
                    )

                    mask = class_prob >= adaptive_threshold

            # ------------------------------------------------------------
            # Smooth class mask in 3D
            # ------------------------------------------------------------
            mask = self._sitk_binary_closing(
                mask=mask,
                radius=1,
                iterations=2,
            )

            # Fill only small accidental holes.
            # This should not fill the real mitral valve opening.
            mask = self._sitk_fill_small_holes_only(
                mask=mask,
                max_hole_size=500,
                max_hole_fraction=0.002,
            )

            # Keep largest coherent component for this class.
            mask = self._sitk_keep_largest_component(
                mask=mask,
                min_component_size=80,
            )

            # ------------------------------------------------------------
            # Fallback: use original segmentation for this label
            # if probability-based mask disappeared.
            # ------------------------------------------------------------
            if int(mask.sum()) == 0:
                original_mask = seg == label

                original_mask = self._sitk_binary_closing(
                    mask=original_mask,
                    radius=1,
                    iterations=1,
                )

                original_mask = self._sitk_fill_small_holes_only(
                    mask=original_mask,
                    max_hole_size=500,
                    max_hole_fraction=0.002,
                )

                mask = self._sitk_keep_largest_component(
                    mask=original_mask,
                    min_component_size=80,
                )

            masks_by_label[label] = mask.astype(bool)

        # ------------------------------------------------------------
        # Combine class masks back into one segmentation.
        # If two class masks overlap, choose the class with higher probability.
        # ------------------------------------------------------------
        post_seg = np.zeros_like(
            seg,
            dtype=np.uint8,
        )

        any_mask = np.zeros_like(
            seg,
            dtype=bool,
        )

        for label, mask in masks_by_label.items():
            any_mask = any_mask | mask

        if int(any_mask.sum()) == 0:
            return post_seg.astype(np.uint8), predicted_probs

        label_list = list(masks_by_label.keys())

        label_prob_stack = np.stack(
            [
                probs[label]
                for label in label_list
            ],
            axis=0,
        )

        winner_idx = np.argmax(
            label_prob_stack,
            axis=0,
        )

        for idx, label in enumerate(label_list):
            label_mask = (
                any_mask
                & masks_by_label[label]
                & (winner_idx == idx)
            )

            post_seg[label_mask] = label

        # Safety: if a class mask has no overlap conflict, keep it.
        for label in label_list:
            clean_mask = masks_by_label[label] & (post_seg == 0)

            post_seg[clean_mask] = label

        return post_seg.astype(np.uint8), predicted_probs
        
    # =========================================================================
    # Task 3: VIDEO post-processing
    # =========================================================================

    def _postprocess_t3_vid(
        self,
        predicted_probs,
        predicted_segments,
    ):
        """
        Task 3 VIDEO.

        2D mitral valve may genuinely be absent.

        Rule:
            If foreground probability is weak or scattered:
                remove foreground completely
                set foreground probability channel to zero

            Otherwise:
                smooth using cv2 morphology
                keep largest coherent component
        """

        import cv2

        fg_prob, fg_channel, seg, probs, probs_had_channel = (
            self._get_binary_foreground_prob(
                predicted_probs=predicted_probs,
                predicted_segments=predicted_segments,
            )
        )

        if fg_prob is None:
            return seg.astype(np.uint8), predicted_probs

        if fg_prob.ndim != 2:
            return seg.astype(np.uint8), self._restore_probs_shape(
                probs,
                probs_had_channel,
            )

        post_seg = seg.copy().astype(np.uint8)

        def remove_foreground():
            post_seg[post_seg == 1] = 0

            probs[fg_channel] = 0.0

            if probs.shape[0] >= 2:
                probs[0] = 1.0

            return post_seg.astype(np.uint8), self._restore_probs_shape(
                probs,
                probs_had_channel,
            )

        peak_prob = float(
            np.max(fg_prob)
        )

        # If no strong valve evidence exists, remove valve.
        if peak_prob < 0.45:
            return remove_foreground()

        candidate = fg_prob >= 0.30

        if int(candidate.sum()) == 0:
            return remove_foreground()

        total_candidate_size = int(
            candidate.sum()
        )

        candidate_uint8 = candidate.astype(np.uint8) * 255

        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )

        candidate_uint8 = cv2.morphologyEx(
            candidate_uint8,
            cv2.MORPH_CLOSE,
            kernel_close,
            iterations=1,
        )

        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

        candidate_uint8 = cv2.morphologyEx(
            candidate_uint8,
            cv2.MORPH_OPEN,
            kernel_open,
            iterations=1,
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate_uint8,
            connectivity=8,
        )

        if num_labels <= 1:
            return remove_foreground()

        component_sizes = stats[1:, cv2.CC_STAT_AREA]

        largest_idx = int(
            np.argmax(component_sizes)
        ) + 1

        largest_size = int(
            stats[largest_idx, cv2.CC_STAT_AREA]
        )

        image_size = float(
            candidate.shape[0] * candidate.shape[1]
        )

        largest_fraction = largest_size / image_size

        coherence_ratio = largest_size / max(
            1,
            total_candidate_size,
        )

        # Suppress if valve evidence is tiny or scattered.
        if largest_size < 40:
            return remove_foreground()

        if largest_fraction < 0.001:
            return remove_foreground()

        if coherence_ratio < 0.50:
            return remove_foreground()

        smooth_mask = labels == largest_idx

        post_seg = np.zeros_like(
            seg,
            dtype=np.uint8,
        )

        post_seg[smooth_mask] = 1

        return post_seg.astype(np.uint8), self._restore_probs_shape(
            probs,
            probs_had_channel,
        )

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

        if prefix.startswith("t1_ct") or prefix.startswith("ct"):
            return self._postprocess_t1_ct(
                predicted_probs=predicted_probs,
                predicted_segments=predicted_segments,
            )

        if prefix.startswith("t2_tee") or prefix.startswith("tee"):
            return self._postprocess_t2_tee(
                predicted_probs=predicted_probs,
                predicted_segments=predicted_segments,
            )

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