"""
Raw-case Dataset for validation/prediction: preprocesses one full case
(all image channels + optional GT) from imagesTr/imagesTs on the fly,
used by SSLnnUNetDataModule.val_dataloader()/test_dataloader().
"""

from pathlib import Path

from torch.utils.data import Dataset

from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles

from utils import to_tensor


class nnUNetRawCaseDataset(Dataset):
    """
    Dataset for raw validation/test cases.

    It prepares one full case:
        - finds raw image channel files
        - runs nnU-Net preprocessing
        - loads GT if available
    """

    def __init__(
        self,
        raw_dataset_folder,
        case_ids,
        split,
        file_ending,
        num_channels,
        has_gt,
        plans_manager,
        configuration_manager,
        dataset_json,
    ):
        self.raw_dataset_folder = Path(raw_dataset_folder)
        self.case_ids = list(sorted(case_ids))
        self.split = split
        self.file_ending = file_ending
        self.num_channels = int(num_channels)
        self.has_gt = bool(has_gt)

        self.pm = plans_manager
        self.cm = configuration_manager
        self.dataset_json = dataset_json

        if self.split == "val":
            self.image_folder = self.raw_dataset_folder / "imagesTr"
            self.label_folder = self.raw_dataset_folder / "labelsTr"

        elif self.split == "test":
            self.image_folder = self.raw_dataset_folder / "imagesTs"
            self.label_folder = None

        else:
            raise ValueError(f"Unknown split '{self.split}'. Use 'val' or 'test'.")

    def __len__(self):
        return len(self.case_ids)

    def _image_files_for_case(self, case_id):
        image_files = []

        for c in range(self.num_channels):
            image_file = self.image_folder / f"{case_id}_{c:04d}{self.file_ending}"

            if not image_file.exists():
                raise FileNotFoundError(f"Missing image file for case '{case_id}': {image_file}")

            image_files.append(str(image_file))

        return image_files

    def _preprocess_case(self, image_files, case_id):
        iterator = preprocessing_iterator_fromfiles(
            list_of_lists=[image_files],
            list_of_segs_from_prev_stage_files=None,
            output_filenames_truncated=[case_id],
            plans_manager=self.pm,
            dataset_json=self.dataset_json,
            configuration_manager=self.cm,
            num_processes=1,
            pin_memory=False,
            verbose=False,
        )

        item = next(iterator)

        # nnU-Net preprocessing_iterator_fromfiles already returns
        # item["data"] as torch.Tensor with dtype float32.
        # Keep it unchanged because nnUNetPredictor expects 4D:
        #   [C, X, Y, Z]
        data = item["data"]
        properties = item["data_properties"]

        return data, properties

    def _load_gt_for_case(self, case_id):
        if not self.has_gt:
            return None, None

        label_file = self.label_folder / f"{case_id}{self.file_ending}"

        if not label_file.exists():
            raise FileNotFoundError(f"Missing label file for case '{case_id}': {label_file}")

        rw = self.pm.image_reader_writer_class()

        gt_data, gt_properties = rw.read_seg(str(label_file))

        # gt_data is a label map (integer class ids), not multi-channel
        # logits/probabilities -- it was read straight from labelsTr.
        # rw.read_seg wraps it with a leading dim-1 axis that is just an
        # I/O convention, not a real class channel:
        #   3D: (1, D, H, W)
        #   2D: (1, 1, H, W)
        #
        # gt_data[0] removes only that reader-convention axis:
        #   3D -> [D, H, W]            (clean)
        #   2D -> [1, H, W]            (still has a leftover singleton,
        #                               because the 2D slice itself is
        #                               represented as (1, H, W))
        gt_data = to_tensor(gt_data[0])

        return gt_data, gt_properties

    def __getitem__(self, index):
        case_id = self.case_ids[index]

        image_files = self._image_files_for_case(case_id)

        data, properties = self._preprocess_case(image_files=image_files, case_id=case_id)

        gt_data, gt_properties = self._load_gt_for_case(case_id)

        return {
            "case_id": case_id,
            "image_files": image_files,
            "data": data,
            "properties": properties,
            "gt_data": gt_data,
            "gt_properties": gt_properties,
            "has_gt": self.has_gt,
            "split": self.split,
        }


def nnunet_raw_case_collate(batch):
    """
    Keep batch unstacked.

    Lightning receives:
        batch = [case_dict]

    Use batch_size=1.
    """

    return batch
