"""
Supervised-only LightningDataModule for nnU-Net.

Train:
    preprocessed nnUNetDataset + nnUNetDataLoader for TrL.

Validation:
    raw imagesTr + labelsTr.
    Dataset preprocesses raw case and returns data, properties, GT.

Prediction:
    raw imagesTs.
    Dataset preprocesses raw case and returns data, properties.

All split logic is inside setup(), because setup() has trainer.world_size
and trainer.global_rank.
"""

from pathlib import Path
import math
import os

import numpy as np
import torch
import lightning as L

from torch.utils.data import Dataset, DataLoader

from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter

from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json, save_json, maybe_mkdir_p

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles

from utils import split_by_rank, to_tensor

# =============================================================================
# Raw case dataset for validation and prediction
# =============================================================================


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


# =============================================================================
# SSL DataModule
# =============================================================================


class SSLnnUNetDataModule(L.LightningDataModule):
    def __init__(self, datamodule_cfg):
        super().__init__()

        self.cfg = datamodule_cfg

        self.dataset_id = self.cfg.dataset_id
        self.configuration = self.cfg.configuration
        self.fold = self.cfg.fold
        self.seed = self.cfg.seed
        self.plans_identifier = self.cfg.plans_identifier

        self.enable_deep_supervision = True
        self.oversample_fg = 0.33
        self.print_case_ids = True

        # These are resolved automatically in setup(),
        # because Lightning trainer.world_size is only reliable there.
        self.num_processes = None
        self.num_cached = None
        self.pin_memory = torch.cuda.is_available()

        self.dataset_name = maybe_convert_to_dataset_name(self.dataset_id)

        self.base = join(nnUNet_preprocessed, self.dataset_name)

        self.raw_base = join(nnUNet_raw, self.dataset_name)

        self.plans = load_json(join(self.base, self.plans_identifier + ".json"))

        self.dataset_json = load_json(join(self.base, "dataset.json"))

        self.pm = PlansManager(self.plans)
        self.cm = self.pm.get_configuration(self.configuration)
        self.lm = self.pm.get_label_manager(self.dataset_json)

        self.folder = join(self.base, self.cm.data_identifier)

        self.batch_size = self.cm.batch_size
        self.ds_scales = self._get_deep_supervision_scales_from_nnunet()

        self.file_ending = self.dataset_json["file_ending"]
        self.num_channels = len(self.dataset_json["channel_names"])

        if self.cm.previous_stage_name is not None:
            raise RuntimeError(
                f"Configuration {self.configuration} is cascaded, "
                "but this SSL DataModule does not support cascaded training."
            )

        self.is_cascaded = False
        self.prev_stage_folder = None

        ssl = self.dataset_json["ssl_case_ids"]

        self.trl_all = list(ssl["TrL"])
        self.ts_all = list(ssl["Ts"])

        self.splits_file = join(self.base, "splits_final_TrL.json")

        self.ds_class = None

        self.dataset_train_labeled = None

        self.raw_dataset_val = None
        self.raw_dataset_test = None

        self.limit_train_batches = None

    # ------------------------------------------------------------------
    # nnU-Net helpers
    # ------------------------------------------------------------------

    def _get_deep_supervision_scales_from_nnunet(self):
        shim = type("S", (), {})()

        shim.configuration_manager = self.cm
        shim.enable_deep_supervision = self.enable_deep_supervision

        return nnUNetTrainer._get_deep_supervision_scales(shim)

    def _get_da_params_from_nnunet(self):
        shim = type("S", (), {})()

        shim.configuration_manager = self.cm
        shim.print_to_log_file = lambda *a, **k: None

        return nnUNetTrainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size(shim)

    def _available_cpu_count(self):
        """
        Return CPUs visible to this job/process.

        On SLURM, os.sched_getaffinity(0) is usually better than
        os.cpu_count(), because it respects the CPU set assigned to the job.
        """

        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            return os.cpu_count() or 1

    def _resolve_augmentation_processes(self, world_size):
        """
        Automatically choose augmentation workers PER AUGMENTER PER RANK.

        This version enforces minimum 4 workers per augmenter.

        In this DataModule each rank creates one augmentation pipeline
        (labeled only -- no unlabeled/pseudo-label data here).

        Therefore total augmentation workers are approximately:
            world_size * num_processes

        Example:
            6 GPUs/ranks and num_processes=8

            total workers = 6 * 8 = 48
            plus 6 main rank processes.
        """

        cpu_count = max(1, int(self._available_cpu_count()))

        world_size = max(1, int(world_size))

        # One train augmenter per rank (labeled only).
        augmenters_per_rank = 1
        total_augmenters = world_size * augmenters_per_rank

        # Reserve CPUs for:
        # - main Lightning/DDP rank processes
        # - file IO
        # - validation/prediction preprocessing
        # - OS/runtime overhead
        reserved_cpus = max(world_size, math.ceil(0.20 * cpu_count))

        usable_cpus = max(1, cpu_count - reserved_cpus)

        workers_per_augmenter = usable_cpus // total_augmenters

        # Minimum 4 workers per augmenter.
        workers_per_augmenter = max(4, workers_per_augmenter)

        # Safety cap.
        workers_per_augmenter = min(workers_per_augmenter, 8)

        num_cached = max(8, min(16, workers_per_augmenter * 2))

        return workers_per_augmenter, num_cached

    def _make_augmenter(self, loader):
        """
        Wrap nnUNetDataLoader with either:
            - SingleThreadedAugmenter if num_processes <= 0
            - NonDetMultiThreadedAugmenter if num_processes > 0
        """

        if self.num_processes is None:
            raise RuntimeError("self.num_processes is None. setup() must run before train_dataloader().")

        if self.num_processes <= 0:
            return SingleThreadedAugmenter(loader, None)

        return NonDetMultiThreadedAugmenter(
            loader,
            None,
            num_processes=self.num_processes,
            num_cached=self.num_cached,
            seeds=None,
            pin_memory=self.pin_memory,
            wait_time=0.002,
        )

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def prepare_data(self):
        cls = infer_dataset_class(self.folder)

        cls.unpack_dataset(self.folder, overwrite_existing=False, num_processes=1, verify=True)

    def setup(self, stage=None):
        """
        Build rank-local datasets.

        All split logic is here because trainer.world_size and
        trainer.global_rank are available here.
        """

        self.ds_class = infer_dataset_class(self.folder)

        trainer = self.trainer

        if trainer is None:
            raise RuntimeError(
                "SSLnnUNetDataModule.setup() was called outside Lightning "
                "Trainer. Use trainer.fit(...), trainer.validate(...), or "
                "trainer.predict(...)."
            )

        world_size = max(1, int(trainer.world_size))

        global_rank = int(trainer.global_rank)

        self.num_processes, self.num_cached = self._resolve_augmentation_processes(world_size=world_size)

        # ------------------------------------------------------------
        # Full TrL train/val split
        # ------------------------------------------------------------
        if self.fold == "all":
            rng = np.random.default_rng(self.seed)

            trl = np.array(sorted(self.trl_all))
            rng.shuffle(trl)

            n_val = min(5, len(trl))

            full_val = sorted(trl[:n_val].tolist())
            full_tr = sorted(trl[n_val:].tolist())

        else:
            if not isfile(self.splits_file):
                maybe_mkdir_p(self.base)

                splits = generate_crossval_split(list(np.sort(self.trl_all)), seed=self.seed, n_splits=5)

                save_json(splits, self.splits_file)

            else:
                splits = load_json(self.splits_file)

            fold = int(self.fold)

            full_tr = list(sorted(splits[fold]["train"]))
            full_val = list(sorted(splits[fold]["val"]))

        full_test = list(sorted(self.ts_all))

        # ------------------------------------------------------------
        # Rank-local split
        # ------------------------------------------------------------
        tr_cases = split_by_rank(full_tr, global_rank=global_rank, world_size=world_size)

        val_cases = split_by_rank(full_val, global_rank=global_rank, world_size=world_size)

        test_cases = split_by_rank(full_test, global_rank=global_rank, world_size=world_size)

        # ------------------------------------------------------------
        # Train limit for infinite nnU-Net loaders
        # ------------------------------------------------------------
        max_rank_cases = 0

        for rank in range(world_size):
            rank_tr_cases = split_by_rank(full_tr, global_rank=rank, world_size=world_size)

            max_rank_cases = max(max_rank_cases, len(rank_tr_cases))

        self.limit_train_batches = max(1, math.ceil(max_rank_cases / self.batch_size))

        trainer.limit_train_batches = self.limit_train_batches

        print(
            f"\n[rank {global_rank}/{world_size}] "
            f"TrL={len(tr_cases)} | "
            f"ValRaw={len(val_cases)} | "
            f"TsRaw={len(test_cases)} | "
            f"limit_train_batches={self.limit_train_batches} | "
            f"visible_cpus={self._available_cpu_count()} | "
            f"aug_workers_per_augmenter={self.num_processes} | "
            f"num_cached={self.num_cached}",
            flush=True,
        )

        if self.print_case_ids and global_rank == 0:
            print("\n[case ids | rank 0]")

            print("TrL:")
            print(tr_cases)

            print("Val raw:")
            print(val_cases)

            print("Test raw:")
            print(test_cases)

            print()

        # ------------------------------------------------------------
        # Patch-based training datasets
        # ------------------------------------------------------------
        self.dataset_train_labeled = self.ds_class(
            self.folder, tr_cases, folder_with_segs_from_previous_stage=self.prev_stage_folder
        )

        # ------------------------------------------------------------
        # Raw validation/test datasets
        # ------------------------------------------------------------
        self.raw_dataset_val = nnUNetRawCaseDataset(
            raw_dataset_folder=self.raw_base,
            case_ids=val_cases,
            split="val",
            file_ending=self.file_ending,
            num_channels=self.num_channels,
            has_gt=True,
            plans_manager=self.pm,
            configuration_manager=self.cm,
            dataset_json=self.dataset_json,
        )

        self.raw_dataset_test = nnUNetRawCaseDataset(
            raw_dataset_folder=self.raw_base,
            case_ids=test_cases,
            split="test",
            file_ending=self.file_ending,
            num_channels=self.num_channels,
            has_gt=False,
            plans_manager=self.pm,
            configuration_manager=self.cm,
            dataset_json=self.dataset_json,
        )

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(self):
        if self.dataset_train_labeled is None:
            raise RuntimeError("dataset_train_labeled is None. setup() did not run correctly.")

        rot, dummy_2d, init_ps, mirror = self._get_da_params_from_nnunet()

        tfm = nnUNetTrainer.get_training_transforms(
            patch_size=self.cm.patch_size,
            rotation_for_DA=rot,
            deep_supervision_scales=self.ds_scales,
            mirror_axes=mirror,
            do_dummy_2d_data_aug=dummy_2d,
            use_mask_for_norm=self.cm.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.lm.foreground_labels,
            regions=(self.lm.foreground_regions if self.lm.has_regions else None),
            ignore_label=self.lm.ignore_label,
        )

        labeled_loader = nnUNetDataLoader(
            data=self.dataset_train_labeled,
            batch_size=self.batch_size,
            patch_size=init_ps,
            final_patch_size=self.cm.patch_size,
            label_manager=self.lm,
            oversample_foreground_percent=self.oversample_fg,
            sampling_probabilities=None,
            pad_sides=None,
            probabilistic_oversampling=False,
            transforms=tfm,
        )

        labeled_iter = self._make_augmenter(labeled_loader)

        return labeled_iter

    def val_dataloader(self):
        if self.raw_dataset_val is None:
            raise RuntimeError("raw_dataset_val is None. setup() did not run correctly.")

        if self.raw_dataset_test is None:
            raise RuntimeError("raw_dataset_test is None. setup() did not run correctly.")

        val_loader = DataLoader(
            self.raw_dataset_val,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=nnunet_raw_case_collate,
        )

        prediction_loader = DataLoader(
            self.raw_dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=nnunet_raw_case_collate,
        )

        return [val_loader, prediction_loader]

    def test_dataloader(self):
        return self.val_dataloader()
