"""
Stage-1 self-supervised pretraining LightningDataModule.

Combines TrL + TrU case ids into one patch-based pool -- labels are
irrelevant here, this stage never reads seg, only data -- holding out a
handful of cases for a validation reconstruction loss. Reuses nnU-Net's
own patch-based Dataset/DataLoader machinery (the same ds_class and
nnUNetDataLoader SSLnnUNetDataModule uses for TrL/TrU) so patch
sampling, geometric augmentation, and preprocessing stay identical to
stage 2 -- only the pretext task (masked-volume reconstruction, built in
PretrainLightningModule) differs.

No deep supervision here (self.ds_scales = None), so
TransformBuilderMixin._build_geometric_transforms skips
DownsampleSegForDSTransform -- irrelevant for a reconstruction target.
"""

import math
import os

import numpy as np
import torch
import lightning as L

from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

from utils import split_by_rank

from .transform_builders import TransformBuilderMixin


class PretrainDataModule(TransformBuilderMixin, L.LightningDataModule):
    def __init__(self, datamodule_cfg):
        super().__init__()

        self.cfg = datamodule_cfg

        self.dataset_id = self.cfg.dataset_id
        self.configuration = self.cfg.configuration
        self.fold = self.cfg.fold
        self.seed = self.cfg.seed
        self.plans_identifier = self.cfg.plans_identifier
        self.prefix = str(self.cfg.prefix)
        self.transform_geometric = bool(self.cfg.transform_geometric)
        self.num_val_cases = int(self.cfg.num_val_cases)

        self.ds_scales = None
        self.pin_memory = torch.cuda.is_available()

        self.num_processes = None
        self.num_cached = None

        self.dataset_name = maybe_convert_to_dataset_name(self.dataset_id)
        self.base = join(nnUNet_preprocessed, self.dataset_name)
        self.plans = load_json(join(self.base, self.plans_identifier + ".json"))
        self.dataset_json = load_json(join(self.base, "dataset.json"))

        self.pm = PlansManager(self.plans)
        self.cm = self.pm.get_configuration(self.configuration)
        self.lm = self.pm.get_label_manager(self.dataset_json)

        self.folder = join(self.base, self.cm.data_identifier)
        self.batch_size = self.cm.batch_size

        ssl = self.dataset_json["ssl_case_ids"]
        self.all_cases = sorted(set(ssl["TrL"]) | set(ssl["TrU"]))

        self.ds_class = None
        self.dataset_train = None
        self.dataset_val = None
        self.limit_train_batches = None

        # Tracks every augmenter _make_augmenter() has handed out, so
        # teardown() can close them deterministically instead of relying on
        # __del__/GC -- see NonDetMultiThreadedAugmenter._finish()'s
        # docstring for why an ungraceful shutdown races the watcher thread
        # against the daemon worker processes dying on their own.
        self._augmenters = []

    # ------------------------------------------------------------------
    # nnU-Net helpers (same pattern as SSLnnUNetDataModule)
    # ------------------------------------------------------------------

    def _get_da_params_from_nnunet(self):
        shim = type("S", (), {})()
        shim.configuration_manager = self.cm
        shim.print_to_log_file = lambda *a, **k: None
        return nnUNetTrainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size(shim)

    def _available_cpu_count(self):
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            return os.cpu_count() or 1

    def _resolve_augmentation_processes(self, world_size):
        cpu_count = max(1, int(self._available_cpu_count()))
        world_size = max(1, int(world_size))

        # One train augmenter per rank here (no separate TrL/TrU pipelines).
        reserved_cpus = max(world_size, math.ceil(0.20 * cpu_count))
        usable_cpus = max(1, cpu_count - reserved_cpus)

        # At least 1 worker -- a hardcoded floor of 4 here would
        # oversubscribe CPUs whenever usable_cpus // world_size comes out
        # below that (e.g. 2 visible CPUs/rank on a 6-GPU job), which is
        # exactly what caused the severe slowdowns/timeouts seen in
        # practice.
        workers_per_augmenter = max(1, usable_cpus // world_size)
        workers_per_augmenter = min(workers_per_augmenter, 8)

        num_cached = max(8, min(16, workers_per_augmenter * 2))

        return workers_per_augmenter, num_cached

    def _make_augmenter(self, loader):
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
        self.ds_class = infer_dataset_class(self.folder)

        trainer = self.trainer

        if trainer is None:
            raise RuntimeError("PretrainDataModule.setup() was called outside Lightning Trainer.")

        world_size = max(1, int(trainer.world_size))
        global_rank = int(trainer.global_rank)

        self.num_processes, self.num_cached = self._resolve_augmentation_processes(world_size=world_size)

        rng = np.random.default_rng(self.seed)
        cases = np.array(self.all_cases)
        rng.shuffle(cases)

        n_val = max(1, min(self.num_val_cases, len(cases) - 1))

        full_val = sorted(cases[:n_val].tolist())
        full_train = sorted(cases[n_val:].tolist())

        train_cases = split_by_rank(full_train, global_rank=global_rank, world_size=world_size)
        val_cases = split_by_rank(full_val, global_rank=global_rank, world_size=world_size)

        # nnUNetDataLoader is an infinite generator (random patch sampling),
        # for both train AND val here -- unlike SSLnnUNetDataModule's val,
        # which uses a real finite raw-case DataLoader. Lightning's fit loop
        # needs an explicit per-epoch batch count for both, and every DDP
        # rank must use the SAME count (a collective op mismatch across
        # ranks would hang), so take the max over all ranks rather than
        # each rank's own (possibly uneven) case count.
        max_rank_train_cases = 0
        max_rank_val_cases = 0
        for rank in range(world_size):
            rank_train_cases = split_by_rank(full_train, global_rank=rank, world_size=world_size)
            rank_val_cases = split_by_rank(full_val, global_rank=rank, world_size=world_size)
            max_rank_train_cases = max(max_rank_train_cases, len(rank_train_cases))
            max_rank_val_cases = max(max_rank_val_cases, len(rank_val_cases))

        self.limit_train_batches = max(1, math.ceil(max_rank_train_cases / self.batch_size))
        self.limit_val_batches = max(1, math.ceil(max_rank_val_cases / self.batch_size))
        trainer.limit_train_batches = self.limit_train_batches
        trainer.limit_val_batches = self.limit_val_batches

        print(
            f"\n[rank {global_rank}/{world_size}] Pretrain train={len(train_cases)} | val={len(val_cases)} | "
            f"limit_train_batches={self.limit_train_batches} | limit_val_batches={self.limit_val_batches} | "
            f"visible_cpus={self._available_cpu_count()} | aug_workers={self.num_processes}",
            flush=True,
        )

        if global_rank == 0:
            print("\n[case ids | rank 0]")
            print("Pretrain train:")
            print(train_cases)
            print("Pretrain val:")
            print(val_cases)
            print()

        # val_cases can be empty on a rank when world_size > len(full_val) --
        # fall back to that rank's own first train case so val_dataloader
        # never has to handle an empty dataset.
        if not val_cases:
            val_cases = train_cases[:1]

        self.dataset_train = self.ds_class(self.folder, train_cases)
        self.dataset_val = self.ds_class(self.folder, val_cases)

    def train_dataloader(self):
        if self.dataset_train is None:
            raise RuntimeError("dataset_train is None. setup() did not run correctly.")

        _, _, init_ps, _ = self._get_da_params_from_nnunet()

        train_tfm = self._build_geometric_transforms(use_spatial_transform=self.transform_geometric)

        loader = nnUNetDataLoader(
            data=self.dataset_train,
            batch_size=self.batch_size,
            patch_size=init_ps,
            final_patch_size=self.cm.patch_size,
            label_manager=self.lm,
            oversample_foreground_percent=0.0,
            sampling_probabilities=None,
            pad_sides=None,
            probabilistic_oversampling=False,
            transforms=train_tfm,
        )

        return self._make_augmenter(loader)

    def val_dataloader(self):
        if self.dataset_val is None:
            raise RuntimeError("dataset_val is None. setup() did not run correctly.")

        # No augmentation for validation -- deterministic center-ish crop
        # straight to final patch_size, so the val reconstruction loss
        # tracks the same un-augmented patches run to run.
        loader = nnUNetDataLoader(
            data=self.dataset_val,
            batch_size=self.batch_size,
            patch_size=self.cm.patch_size,
            final_patch_size=self.cm.patch_size,
            label_manager=self.lm,
            oversample_foreground_percent=0.0,
            sampling_probabilities=None,
            pad_sides=None,
            probabilistic_oversampling=False,
            transforms=None,
        )

        return SingleThreadedAugmenter(loader, None)

    def test_dataloader(self):
        return self.val_dataloader()
