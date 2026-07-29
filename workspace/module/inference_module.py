"""
Lightning wrapper for inference.

Builds the network + prediction ops via NNUnetSetup (module/nnunet.py --
static, shared as-is across the SSL family; see that file's own docstring
for why). predict_step writes each case's mask directly (mirrors how the
original training code's SSLnnUNetLightningModule.validation_step writes
as it goes, rather than accumulating trainer.predict()'s return values,
which would hold every case's full-resolution volume/logits in memory at
once), and records the written path -- turning that into a
task*_predictions.json entry is run_inference.py's job, not this class's.
"""

from pathlib import Path

import lightning as L

from .nnunet import NNUnetSetup
from utils import write_submission_prediction


class InferenceLightningModule(L.LightningModule):
    def __init__(self, litmodule_cfg, output_dir):
        super().__init__()

        self.cfg = litmodule_cfg
        self.output_dir = Path(output_dir)

        # enable_deep_supervision=True must match training (SSLnnUNetLightningModule
        # hardcodes True) -- the checkpoint's state_dict includes the deep-supervision
        # decoder heads' parameters, so the architecture must be built with them present
        # for load_state_dict to match. PredictionOps toggles deep_supervision off at
        # the *forward-pass* level per case; the parameters themselves are unaffected.
        self.nnunet = NNUnetSetup(litmodule_cfg, enable_deep_supervision=True, trainer_name=self.__class__.__name__)
        self.network = self.nnunet.build_network()

        self.is_video = str(litmodule_cfg.prefix) == "t3_vid"
        self.convert_to_255 = self.is_video
        self.keep_classes = [self.nnunet.dataset_json["labels"]["class_10"]] if self.is_video else None
        self.submission_output_format = "png" if self.is_video else "nii.gz"

        self.pred_files = []

    def forward(self, x):
        return self.network(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        prediction = self.nnunet.predictor.run_prediction(
            network=self.network, device=self.device, batch=batch, batch_idx=batch_idx
        )

        # Called directly (PredictionOps has no write wrapper -- see
        # nnunet.py) since we need the returned path to record it below.
        pred_file = write_submission_prediction(
            prediction=prediction,
            output_folder=self.output_dir,
            configuration_manager=self.nnunet.cm,
            convert_to_255=self.convert_to_255,
            keep_classes=self.keep_classes,
            output_format=self.submission_output_format,
        )

        self.pred_files.append(Path(pred_file))
