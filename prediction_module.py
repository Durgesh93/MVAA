"""
PredictionMixin for SSL nnU-Net.

Takes a prepared case from the datamodule, runs sliding-window inference,
restores prediction to original image shape, and returns the prediction
dictionary.
"""

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
)


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

        item.update(
            {
                "logits": logits,
                "predicted_segments": predicted_segments,
                "predicted_probs": predicted_probs,
            }
        )

        return item