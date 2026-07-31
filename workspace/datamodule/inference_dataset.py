"""
Inference-only nnU-Net raw-case datasets for the Docker submission contract.

Mirrors datamodule/raw_case_dataset.py's nnUNetRawCaseDataset (has_gt=False
path: preprocesses one raw case via nnU-Net's own
preprocessing_iterator_fromfiles) but sourced from /input instead of
nnUNet_raw/imagesTs, with no TrL/TrU split, DDP rank splitting, or
augmenters.

CT/TEE reference data is plain nii.gz -- InferenceCaseDataset resolves its
case list from test_cases.json (see discover_cases). Video's reference
data is PNG frames -- the '_0000/_0001/_0002.nii.gz' triplet is only how
*our* training pipeline stores it internally, so video gets its own
dataset (VideoInferenceCaseDataset) that converts each frame on the fly
instead (see video_source.py), likewise driven by test_cases.json.
"""

import json
from pathlib import Path

from torch.utils.data import Dataset

from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles

from .video_source import discover_video_frames, write_frame_as_nnunet_channels


def discover_cases(image_folder, file_ending, num_channels):
    """
    Map each case_id to its single-channel image file, per the official
    submission contract's test_cases.json manifest:
        {"cases": [{"case_id": ..., "image": "relative/path"}, ...]}
    "image" is resolved relative to image_folder's parent (/input itself),
    not to image_folder (/input/<prefix>) -- confirmed against a real
    failed hidden-test run whose organizer-side error showed the path
    resolved as .../t1_ct/t1_ct/0001.nii.gz when this code joined "image"
    onto image_folder directly, meaning "image" already carries the
    <prefix>/ segment (e.g. "t1_ct/0001.nii.gz"). Only case_id/image are
    read -- any additional fields on an entry are ignored. Only holds for
    num_channels=1 (all CT/TEE ever use): the manifest gives one filename
    per case_id, with no way to say which files belong to the same
    multi-channel case.
    """
    if num_channels != 1:
        raise ValueError(f"discover_cases only supports single-channel inputs, got num_channels={num_channels}")

    image_folder = Path(image_folder)
    manifest_path = image_folder / "test_cases.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing test_cases.json manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())

    input_root = image_folder.parent

    return {entry["case_id"]: [str(input_root / entry["image"])] for entry in manifest["cases"]}


def preprocess_case_files(image_files, case_id, plans_manager, configuration_manager, dataset_json):
    iterator = preprocessing_iterator_fromfiles(
        list_of_lists=[image_files],
        list_of_segs_from_prev_stage_files=None,
        output_filenames_truncated=[case_id],
        plans_manager=plans_manager,
        dataset_json=dataset_json,
        configuration_manager=configuration_manager,
        num_processes=1,
        pin_memory=False,
        verbose=False,
    )

    item = next(iterator)

    return item["data"], item["data_properties"]


class InferenceCaseDataset(Dataset):
    """Raw, unlabeled nii.gz case dataset for prediction (CT/TEE): preprocesses one full case on the fly."""

    def __init__(
        self,
        image_folder,
        file_ending,
        num_channels,
        plans_manager,
        configuration_manager,
        dataset_json,
    ):
        self.image_folder = Path(image_folder)
        self.file_ending = file_ending
        self.num_channels = int(num_channels)

        self.pm = plans_manager
        self.cm = configuration_manager
        self.dataset_json = dataset_json

        self.cases = discover_cases(self.image_folder, file_ending, self.num_channels)
        self.case_ids = sorted(self.cases)

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        case_id = self.case_ids[index]
        image_files = self.cases[case_id]

        data, properties = preprocess_case_files(image_files, case_id, self.pm, self.cm, self.dataset_json)

        return {
            "case_id": case_id,
            "image_files": image_files,
            "data": data,
            "properties": properties,
            "gt_data": None,
            "gt_properties": None,
            "has_gt": False,
            "split": "test",
        }


class VideoInferenceCaseDataset(Dataset):
    """Raw PNG frame dataset for video: converts each frame to nnU-Net channels on the fly, then preprocesses."""

    def __init__(
        self,
        image_folder,
        work_dir,
        plans_manager,
        configuration_manager,
        dataset_json,
    ):
        self.image_folder = Path(image_folder)
        self.work_dir = Path(work_dir)

        self.pm = plans_manager
        self.cm = configuration_manager
        self.dataset_json = dataset_json

        self.frames = discover_video_frames(self.image_folder)
        self.case_ids = sorted(self.frames)

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        case_id = self.case_ids[index]
        frame = self.frames[case_id]

        image_files = write_frame_as_nnunet_channels(frame["path"], case_id, self.work_dir)

        data, properties = preprocess_case_files(image_files, case_id, self.pm, self.cm, self.dataset_json)

        return {
            "case_id": case_id,
            "image_files": image_files,
            "data": data,
            "properties": properties,
            "gt_data": None,
            "gt_properties": None,
            "has_gt": False,
            "split": "test",
            # Mirrors the manifest's own "image" subfolder (e.g. a
            # recording id) into the output, instead of guessing grouping
            # from case_id's shape -- case_id is an opaque, organizer-
            # supplied string that must be preserved as-is, not parsed.
            "output_subdir": frame["output_subdir"],
        }


def inference_case_collate(batch):
    return batch
