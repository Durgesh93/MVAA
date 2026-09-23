"""
Semi-supervised LightningDataModule for SSL nnU-Net on MVSeg2023.

Splits come from dataset.json's `case_ids`, which records MVSeg2023's own
train (105) / val (30) / test (40) partition, which is the only split the
dataset defines and is used as-is.

Every MVSeg2023 case is labeled, so the labeled/unlabeled partition is
SYNTHESIZED rather than read: cfg.labeled_fraction of the train split
keeps its labels (TrL) and the remainder becomes the unlabeled pool
(TrU). The draw is seeded and nested across fractions. Nothing is
stripped on disk -- a case is unlabeled only because it is routed to the
unlabeled loader, whose targets never reach a loss. At
labeled_fraction=1.0 the pool is empty, no unlabeled loader is built, and
training is plain supervised.

Train:
    preprocessed nnUNetDataset + nnUNetDataLoader for TrL, plus a
    MultiViewUnlabeledDataLoader for TrU (K independently-augmented
    intensity views per sample, sharing one geometric draw -- see
    multiview_loader.MultiViewUnlabeledDataLoader's docstring).

Validation:
    raw imagesTr + labelsTr, the 30 official val cases.

Test:
    raw imagesTs + labelsTs, the 40 official test cases. MVSeg2023
    releases these labeled, so the held-out set is scorable.

Rank-local sharding lives in setup(), which is where trainer.world_size
and trainer.global_rank are available.
"""

import math
import os

import numpy as np
import torch
import lightning as L

from torch.utils.data import DataLoader
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json

from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

from utils import split_by_rank, override_patch_size

from .raw_case_dataset import nnUNetRawCaseDataset, nnunet_raw_case_collate
from .multiview_loader import MultiViewUnlabeledDataLoader
from .transform_builders import TransformBuilderMixin


class SSLnnUNetDataModule(TransformBuilderMixin, L.LightningDataModule):
    def __init__(self, datamodule_cfg):
        super().__init__()

        self.cfg = datamodule_cfg

        self.dataset_id = self.cfg.dataset_id
        self.configuration = self.cfg.configuration
        self.seed = self.cfg.seed
        self.plans_identifier = self.cfg.plans_identifier
        self.K = int(self.cfg.K)
        self.transform_geometric = bool(self.cfg.transform_geometric)

        self.enable_deep_supervision = True
        self.oversample_fg = float(self.cfg.oversample_fg)
        self.print_case_ids = True

        self.labeled_fraction = float(self.cfg.labeled_fraction)

        if not 0.0 < self.labeled_fraction <= 1.0:
            raise ValueError(
                f"labeled_fraction must be in (0, 1], got {self.labeled_fraction}. "
                "Use 1.0 for the fully supervised baseline (empty unlabeled pool)."
            )

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

        override_patch_size(self.cm, self.cfg.patch_size_override)

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

        # Official MVSeg2023 partition, written by
        # data_preparation/data_mvseg2023_nnunet.py. Every one of these
        # cases is labeled -- unlike the old ssl_case_ids TrL/TrU/Ts key,
        # this says nothing about supervision, only provenance. The
        # labeled/unlabeled split is synthesized in setup() from
        # cfg.labeled_fraction.
        case_ids = self.dataset_json["case_ids"]

        self.train_all = list(case_ids["train"])
        self.val_all = list(case_ids["val"])
        self.ts_all = list(case_ids["test"])

        self.ds_class = None

        self.dataset_train_labeled = None
        self.dataset_train_unlabeled = None

        self.raw_dataset_val = None
        self.raw_dataset_test = None

        self.limit_train_batches = None

        # Tracks every augmenter _make_augmenter() has handed out, so
        # teardown() can close them deterministically instead of relying on
        # __del__/GC -- see NonDetMultiThreadedAugmenter._finish()'s
        # docstring for why an ungraceful shutdown races the watcher thread
        # against the daemon worker processes dying on their own.
        self._augmenters = []

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

        In this DataModule each rank creates two augmentation pipelines:
            1. labeled (TrL)
            2. unlabeled (TrU)

        Therefore total augmentation workers are approximately:
            world_size * 2 * num_processes

        Example:
            6 GPUs/ranks and num_processes=4

            total workers = 6 * 2 * 4 = 48
            plus 6 main rank processes.
        """

        cpu_count = max(1, int(self._available_cpu_count()))

        world_size = max(1, int(world_size))

        # Two train augmenters per rank: labeled + unlabeled.
        augmenters_per_rank = 2
        total_augmenters = world_size * augmenters_per_rank

        # Reserve CPUs for:
        # - main Lightning/DDP rank processes
        # - file IO
        # - validation/prediction preprocessing
        # - OS/runtime overhead
        reserved_cpus = max(world_size, math.ceil(0.20 * cpu_count))

        usable_cpus = max(1, cpu_count - reserved_cpus)

        workers_per_augmenter = usable_cpus // total_augmenters

        # At least 1 worker per augmenter -- a hardcoded floor of 4 here
        # would oversubscribe CPUs whenever usable_cpus // total_augmenters
        # comes out below that (e.g. 2 visible CPUs/rank on a 6-GPU job),
        # which is exactly what caused the severe slowdowns/timeouts seen
        # in practice.
        workers_per_augmenter = max(1, workers_per_augmenter)

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
            augmenter = SingleThreadedAugmenter(loader, None)
        else:
            augmenter = NonDetMultiThreadedAugmenter(
                loader,
                None,
                num_processes=self.num_processes,
                num_cached=self.num_cached,
                seeds=None,
                pin_memory=self.pin_memory,
                wait_time=0.002,
            )

        self._augmenters.append(augmenter)
        return augmenter

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def prepare_data(self):
        cls = infer_dataset_class(self.folder)
        cls.unpack_dataset(self.folder, overwrite_existing=False, num_processes=1, verify=True)

    def teardown(self, stage=None):
        # Close background augmenter workers deterministically here, before
        # the process exits -- otherwise their shutdown races the daemon
        # worker processes dying on their own vs. __del__/GC eventually
        # calling _finish(), which is what produces the "background workers
        # are no longer alive" RuntimeError from the results_loop watcher
        # thread at interpreter shutdown.
        for augmenter in self._augmenters:
            finish = getattr(augmenter, "_finish", None)
            if finish is not None:
                finish()
        self._augmenters = []

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
        # Train/val/test split
        #
        # Taken verbatim from MVSeg2023's own partition. The dataset
        # defines exactly one train/val boundary, so cross-validation
        # would mean re-splitting a split that is already authoritative.
        #
        # The only split decided here is labeled vs unlabeled, below.
        # ------------------------------------------------------------
        full_tr = list(sorted(self.train_all))
        full_val = list(sorted(self.val_all))
        full_test = list(sorted(self.ts_all))

        # ------------------------------------------------------------
        # Synthesize the labeled/unlabeled partition
        #
        # MVSeg2023 is fully labeled, so the unlabeled pool is made by
        # withholding labels rather than read from the dataset. The
        # shuffle is seeded and drawn ONCE over the whole train split,
        # then truncated -- which makes the subsets nested (the cases
        # chosen at 0.1 are a prefix of those chosen at 0.2), so a
        # labeled_fraction sweep varies how much supervision there is,
        # not which cases supply it.
        #
        # Nothing strips the segmentations on disk: a case is "unlabeled"
        # purely because it is routed to the unlabeled loader, whose
        # targets never reach a loss (see the LightningModule's
        # _pseudo_loss, which reads only data_views).
        # ------------------------------------------------------------
        shuffled_tr = np.array(sorted(full_tr))
        np.random.default_rng(self.seed).shuffle(shuffled_tr)

        n_labeled = max(1, int(round(self.labeled_fraction * len(shuffled_tr))))

        full_labeled = list(sorted(shuffled_tr[:n_labeled].tolist()))
        full_tru = list(sorted(shuffled_tr[n_labeled:].tolist()))

        full_tr = full_labeled

        self.has_unlabeled = len(full_tru) > 0

        # ------------------------------------------------------------
        # Rank-local split
        # ------------------------------------------------------------
        tr_cases = split_by_rank(full_tr, global_rank=global_rank, world_size=world_size)

        val_cases = split_by_rank(full_val, global_rank=global_rank, world_size=world_size)

        test_cases = split_by_rank(full_test, global_rank=global_rank, world_size=world_size)

        tru_cases = split_by_rank(full_tru, global_rank=global_rank, world_size=world_size)

        # ------------------------------------------------------------
        # Train limit for infinite nnU-Net loaders
        #
        # CombinedLoader(mode="max_size_cycle") cycles the smaller of
        # TrL/TrU per rank to match the larger, so the epoch length must
        # account for whichever is bigger -- not just TrL.
        # ------------------------------------------------------------
        max_rank_cases = 0

        for rank in range(world_size):
            rank_tr_cases = split_by_rank(full_tr, global_rank=rank, world_size=world_size)
            rank_tru_cases = split_by_rank(full_tru, global_rank=rank, world_size=world_size)
            max_rank_cases = max(max_rank_cases, len(rank_tr_cases), len(rank_tru_cases))

        self.limit_train_batches = max(1, math.ceil(max_rank_cases / self.batch_size))

        trainer.limit_train_batches = self.limit_train_batches

        print(
            f"\n[rank {global_rank}/{world_size}] "
            f"labeled_fraction={self.labeled_fraction} "
            f"({len(full_tr)}/{len(full_tr) + len(full_tru)} train cases labeled) | "
            f"TrL={len(tr_cases)} | "
            f"TrU={len(tru_cases)} | "
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

            print("TrU:")
            print(tru_cases)

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

        # At labeled_fraction=1.0 there is no unlabeled pool, so no
        # unlabeled dataset/loader is built at all and training runs
        # purely supervised through this same code path.
        self.dataset_train_unlabeled = (
            self.ds_class(self.folder, tru_cases, folder_with_segs_from_previous_stage=self.prev_stage_folder)
            if self.has_unlabeled
            else None
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
            # MVSeg2023's test split ships real masks, which the prep
            # script keeps in labelsTs -- so the held-out set is scorable
            # rather than prediction-only.
            has_gt=True,
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

        if self.has_unlabeled and self.dataset_train_unlabeled is None:
            raise RuntimeError("dataset_train_unlabeled is None. setup() did not run correctly.")

        _, _, init_ps, _ = self._get_da_params_from_nnunet()

        # Same split pipeline (geometric + task-specific intensity) for all
        # three tasks, including CT -- no more bundled
        # nnUNetTrainer.get_training_transforms() special case.
        labeled_tfm = ComposeTransforms(
            [
                self._build_geometric_transforms(use_spatial_transform=self.transform_geometric),
                self._build_intensity_transforms(),
            ]
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
            transforms=labeled_tfm,
        )

        if not self.has_unlabeled:
            return CombinedLoader({"labeled": self._make_augmenter(labeled_loader)}, mode="max_size_cycle")

        unlabeled_loader = MultiViewUnlabeledDataLoader(
            data=self.dataset_train_unlabeled,
            batch_size=self.batch_size,
            patch_size=init_ps,
            final_patch_size=self.cm.patch_size,
            label_manager=self.lm,
            oversample_foreground_percent=0.0,
            sampling_probabilities=None,
            pad_sides=None,
            probabilistic_oversampling=False,
            geometric_transforms=self._build_geometric_transforms(use_spatial_transform=self.transform_geometric),
            intensity_transforms=self._build_intensity_transforms_strong(),
            num_views=self.K,
        )

        labeled_iter = self._make_augmenter(labeled_loader)
        unlabeled_iter = self._make_augmenter(unlabeled_loader)

        return CombinedLoader({"labeled": labeled_iter, "unlabeled": unlabeled_iter}, mode="max_size_cycle")

    def _raw_case_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=nnunet_raw_case_collate,
        )

    def val_dataloader(self):
        """
        The 30 official val cases only.

        This used to return [val_loader, test_loader], which meant every
        validation epoch also ran full sliding-window inference over the
        held-out test set -- 40 extra volumes per epoch, scored by
        nothing. The test set now has its own loader and is only touched
        by `engine.py test`.
        """

        if self.raw_dataset_val is None:
            raise RuntimeError("raw_dataset_val is None. setup() did not run correctly.")

        return self._raw_case_loader(self.raw_dataset_val)

    def test_dataloader(self):
        """The 40 official test cases, which MVSeg2023 ships labeled."""

        if self.raw_dataset_test is None:
            raise RuntimeError("raw_dataset_test is None. setup() did not run correctly.")

        return self._raw_case_loader(self.raw_dataset_test)
