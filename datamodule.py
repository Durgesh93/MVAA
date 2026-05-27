"""
Semi-supervised LightningDataModule for nnU-Net v2.

Simple design:
- TrL and TrU are always split by rank.
- Val is kept full on every rank.
- Uses SingleThreadedAugmenter only.
- Test/predict cases are stored as case IDs only.
- Raw test prediction is handled in prediction_module.py from imagesTs.
"""

import math
import numpy as np
import lightning as L

from lightning.pytorch.utilities.combined_loader import CombinedLoader

from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    isfile,
    load_json,
    save_json,
    maybe_mkdir_p,
)

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


class SSLnnUNetDataModule(L.LightningDataModule):
    def __init__(self, datamodule_cfg):
        super().__init__()

        self.cfg = datamodule_cfg

        self.dataset_id = self.cfg.dataset_id
        self.configuration = self.cfg.configuration
        self.fold = self.cfg.fold
        self.seed = self.cfg.seed
        self.plans_identifier = self.cfg.plans_identifier

        # Set in engine.py before creating datamodule
        self.num_devices = int(getattr(self.cfg, "num_devices", 1))

        self.enable_deep_supervision = True
        self.oversample_fg = 0.33
        self.print_case_ids = True

        # ------------------------------------------------------------
        # Load nnU-Net files
        # ------------------------------------------------------------
        self.dataset_name = maybe_convert_to_dataset_name(self.dataset_id)
        self.base = join(nnUNet_preprocessed, self.dataset_name)

        self.plans = load_json(
            join(self.base, self.plans_identifier + ".json")
        )

        self.dataset_json = load_json(
            join(self.base, "dataset.json")
        )

        self.pm = PlansManager(self.plans)
        self.cm = self.pm.get_configuration(self.configuration)
        self.lm = self.pm.get_label_manager(self.dataset_json)

        self.folder = join(self.base, self.cm.data_identifier)

        self.batch_size = self.cm.batch_size
        self.ds_scales = self._get_deep_supervision_scales_from_nnunet()

        if self.cm.previous_stage_name is not None:
            raise RuntimeError(
                f"Configuration {self.configuration} is cascaded, "
                "but this SSL DataModule does not support cascaded training."
            )

        self.is_cascaded = False
        self.prev_stage_folder = None

        # ------------------------------------------------------------
        # SSL case IDs
        # ------------------------------------------------------------
        ssl = self.dataset_json["ssl_case_ids"]

        self.trl_all = list(ssl["TrL"])
        self.tru_all = list(ssl["TrU"])
        self.ts_all = list(ssl["Ts"])

        self.splits_file = join(self.base, "splits_final_TrL.json")

        # ------------------------------------------------------------
        # Full split + limits
        # ------------------------------------------------------------
        self.full_tr = None
        self.full_tru = None
        self.full_val = None

        self.limit_train_batches = None
        self.limit_val_batches = None

        self._make_full_split_and_epoch_limits()

        # ------------------------------------------------------------
        # Runtime objects
        # ------------------------------------------------------------
        self.ds_class = None

        self.dataset_train_labeled = None
        self.dataset_train_unlabeled = None
        self.dataset_val = None

        # Raw test case IDs only.
        # Actual raw prediction is handled in prediction_module.py.
        self.test_cases = None

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

        return (
            nnUNetTrainer
            .configure_rotation_dummyDA_mirroring_and_inital_patch_size(shim)
        )

    # ------------------------------------------------------------------
    # Full split + epoch limits
    # ------------------------------------------------------------------
    def _make_full_split_and_epoch_limits(self):
        """
        Makes:
            self.full_tr
            self.full_tru
            self.full_val
            self.limit_train_batches
            self.limit_val_batches
        """

        # ------------------------------------------------------------
        # TrL train/val split
        # ------------------------------------------------------------
        if self.fold == "all":
            rng = np.random.default_rng(self.seed)

            trl = np.array(sorted(self.trl_all))
            rng.shuffle(trl)

            # Isolate 5 labeled validation cases
            n_val = min(5, len(trl))

            self.full_val = sorted(trl[:n_val].tolist())
            self.full_tr = sorted(trl[n_val:].tolist())

        else:
            if not isfile(self.splits_file):
                maybe_mkdir_p(self.base)

                splits = generate_crossval_split(
                    list(np.sort(self.trl_all)),
                    seed=self.seed,
                    n_splits=5,
                )

                save_json(splits, self.splits_file)
            else:
                splits = load_json(self.splits_file)

            fold = int(self.fold)

            self.full_tr = list(sorted(splits[fold]["train"]))
            self.full_val = list(sorted(splits[fold]["val"]))

        # ------------------------------------------------------------
        # TrU split
        # ------------------------------------------------------------
        self.full_tru = list(sorted(self.tru_all))

        # ------------------------------------------------------------
        # Simulate rank split for epoch limit
        # ------------------------------------------------------------
        def max_cases_per_rank(case_ids):
            case_ids = list(sorted(case_ids))

            if self.num_devices <= 1:
                return len(case_ids)

            counts = [
                len(case_ids[rank::self.num_devices])
                for rank in range(self.num_devices)
            ]

            return max(counts)

        max_rank_trl = max_cases_per_rank(self.full_tr)
        max_rank_tru = max_cases_per_rank(self.full_tru)

        train_cases = max(max_rank_trl, max_rank_tru)
        val_cases = len(self.full_val)

        self.limit_train_batches = max(
            1,
            math.ceil(train_cases / self.batch_size),
        )

        self.limit_val_batches = max(
            1,
            math.ceil(val_cases / self.batch_size),
        )

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def prepare_data(self):
        cls = infer_dataset_class(self.folder)

        cls.unpack_dataset(
            self.folder,
            overwrite_existing=False,
            num_processes=1,
            verify=True,
        )

    def setup(self, stage=None):
        self.ds_class = infer_dataset_class(self.folder)

        trainer = getattr(self, "trainer", None)

        if trainer is None:
            world_size = 1
            global_rank = 0
        else:
            world_size = getattr(trainer, "world_size", 1)
            global_rank = getattr(trainer, "global_rank", 0)

        def split_by_rank(case_ids):
            case_ids = list(sorted(case_ids))

            if world_size <= 1:
                return case_ids

            return case_ids[global_rank::world_size]

        tr_cases = split_by_rank(self.full_tr)
        tru_cases = split_by_rank(self.full_tru)
        val_cases = list(sorted(self.full_val))
        test_cases = list(sorted(self.ts_all))

        self.test_cases = test_cases

        # ------------------------------------------------------------
        # Print counts from every rank
        # ------------------------------------------------------------
        print(
            f"\n[rank {global_rank}/{world_size}] "
            f"TrL={len(tr_cases)} | "
            f"TrU={len(tru_cases)} | "
            f"Val={len(val_cases)} | "
            f"Ts={len(test_cases)}",
            flush=True,
        )

        # ------------------------------------------------------------
        # Print case IDs only from rank 0
        # ------------------------------------------------------------
        if self.print_case_ids and global_rank == 0:
            print("\n[case ids | rank 0]")

            print("TrL:")
            print(tr_cases)

            print("TrU:")
            print(tru_cases)

            print("Val:")
            print(val_cases)

            print("Test:")
            print(test_cases)

            print()

        self.dataset_train_labeled = self.ds_class(
            self.folder,
            tr_cases,
            folder_with_segs_from_previous_stage=self.prev_stage_folder,
        )

        self.dataset_train_unlabeled = self.ds_class(
            self.folder,
            tru_cases,
            folder_with_segs_from_previous_stage=self.prev_stage_folder,
        )

        self.dataset_val = self.ds_class(
            self.folder,
            val_cases,
            folder_with_segs_from_previous_stage=self.prev_stage_folder,
        )

    # ------------------------------------------------------------------
    # Lightning dataloaders
    # ------------------------------------------------------------------
    def train_dataloader(self):
        lab = self._make_iter(
            dataset=self.dataset_train_labeled,
            batch_size=self.batch_size,
            oversample_fg=self.oversample_fg,
            train=True,
        )

        unl = self._make_iter(
            dataset=self.dataset_train_unlabeled,
            batch_size=self.batch_size,
            oversample_fg=0.0,
            train=True,
        )

        return CombinedLoader(
            {
                "labeled": lab,
                "unlabeled": unl,
            },
            mode="max_size_cycle",
        )

    def val_dataloader(self):
        return self._make_iter(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            oversample_fg=self.oversample_fg,
            train=False,
        )

    # ------------------------------------------------------------------
    # Predict dataloader
    # ------------------------------------------------------------------
    def predict_dataloader(self):
        """
        Dummy predict dataloader.

        We do not use Lightning batches for prediction.
        Actual nnU-Net test prediction is handled in on_predict_start()
        using raw files from nnUNet_raw/<dataset_name>/imagesTs.

        Lightning still requires a predict_dataloader, so we return
        a one-item dummy loader.
        """

        from torch.utils.data import DataLoader, TensorDataset
        import torch

        dummy = TensorDataset(torch.zeros(1, 1))

        return DataLoader(
            dummy,
            batch_size=1,
            num_workers=0,
        )
    # ------------------------------------------------------------------
    # Iterator creation
    # ------------------------------------------------------------------
    def _make_iter(self, dataset, batch_size, oversample_fg, train):
        if dataset is None:
            raise RuntimeError(
                "Dataset is None. setup() did not run correctly."
            )

        rot, dummy_2d, init_ps, mirror = self._get_da_params_from_nnunet()

        if train:
            tfm = nnUNetTrainer.get_training_transforms(
                patch_size=self.cm.patch_size,
                rotation_for_DA=rot,
                deep_supervision_scales=self.ds_scales,
                mirror_axes=mirror,
                do_dummy_2d_data_aug=dummy_2d,
                use_mask_for_norm=self.cm.use_mask_for_norm,
                is_cascaded=self.is_cascaded,
                foreground_labels=self.lm.foreground_labels,
                regions=(
                    self.lm.foreground_regions
                    if self.lm.has_regions
                    else None
                ),
                ignore_label=self.lm.ignore_label,
            )

            patch = init_ps

        else:
            tfm = nnUNetTrainer.get_validation_transforms(
                deep_supervision_scales=self.ds_scales,
                is_cascaded=self.is_cascaded,
                foreground_labels=self.lm.foreground_labels,
                regions=(
                    self.lm.foreground_regions
                    if self.lm.has_regions
                    else None
                ),
                ignore_label=self.lm.ignore_label,
            )

            patch = self.cm.patch_size

        dl = nnUNetDataLoader(
            data=dataset,
            batch_size=batch_size,
            patch_size=patch,
            final_patch_size=self.cm.patch_size,
            label_manager=self.lm,
            oversample_foreground_percent=oversample_fg,
            sampling_probabilities=None,
            pad_sides=None,
            probabilistic_oversampling=False,
            transforms=tfm,
        )

        return SingleThreadedAugmenter(dl, None)
