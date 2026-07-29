"""
Inference-only nnU-Net raw-case datasets for the Docker submission
contract.

Mirrors datamodule/raw_case_dataset.py's nnUNetRawCaseDataset (has_gt=False
prediction path: preprocesses one raw case via nnU-Net's own
preprocessing_iterator_fromfiles, returning the same dict shape consumed by
module/nnunet.py's PredictionOps.run_prediction) but sourced from /input
instead of nnUNet_raw/imagesTs, with no pre-known case-id list, no TrL/TrU
split, no DDP rank splitting, no augmenters.

CT/TEE raw reference data is plain nii.gz on disk (see
data_task2_TEE_nnunet.py's collect_tee_files) -- InferenceCaseDataset
treats every '*<file_ending>' file in image_folder as its own case,
whatever it's named, no naming convention assumed. Video's raw reference
data is PNG frames (see data_task3_VIDEO_nnunet.py's collect_video_files)
-- the '_0000/_0001/_0002.nii.gz' triplet is only how *our* training
pipeline stores it internally after conversion, so video needs its own
dataset (VideoInferenceCaseDataset, in this file) that converts each frame
on the fly instead (see video_source.py), likewise accepting any PNG.
"""

from pathlib import Path

from torch.utils.data import Dataset

from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles

from .video_source import discover_video_frames, write_frame_as_nnunet_channels


def discover_cases(image_folder, file_ending, num_channels):
    """
    Map each case_id under image_folder to its single-channel image file.

    Every '*<file_ending>' file is one case, using its own filename (minus
    file_ending) as case_id -- no assumption about naming, so this works
    regardless of what the files are actually called. Only holds for
    num_channels=1 (all CT/TEE ever use): filenames alone can't say which
    files belong to the same multi-channel case.
    """
    if num_channels != 1:
        raise ValueError(f"discover_cases only supports single-channel inputs, got num_channels={num_channels}")

    image_folder = Path(image_folder)

    return {f.name[: -len(file_ending)]: [str(f)] for f in sorted(image_folder.glob(f"*{file_ending}"))}


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
        png_path = self.frames[case_id]

        image_files = write_frame_as_nnunet_channels(png_path, case_id, self.work_dir)

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


def inference_case_collate(batch):
    return batch
