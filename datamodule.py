"""
Semi-supervised LightningDataModule for nnU-Net v2.
Uses Hydra/OmegaConf datamodule config object.
"""

import numpy as np
import torch
import lightning as L

from lightning.pytorch.utilities.combined_loader import CombinedLoader

from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    isfile,
    load_json,
    save_json,
)

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

from utils import set_nnunet_env


class SSLnnUNetDataModule(L.LightningDataModule):
    def __init__(self, datamodule_cfg):
        super().__init__()

        self.cfg = datamodule_cfg

        # ------------------------------------------------------------------
        # nnU-Net paths
        # ------------------------------------------------------------------
        self.nnunet_raw = self.cfg.paths.nnunet_raw
        self.nnunet_preprocessed = self.cfg.paths.nnunet_preprocessed
        self.nnunet_results = self.cfg.paths.nnunet_results

        set_nnunet_env(
            nnunet_raw=self.nnunet_raw,
            nnunet_preprocessed=self.nnunet_preprocessed,
            nnunet_results=self.nnunet_results,
        )

        self._env = {
            "nnunet_raw": self.nnunet_raw,
            "nnunet_preprocessed": self.nnunet_preprocessed,
            "nnunet_results": self.nnunet_results,
        }

        # ------------------------------------------------------------------
        # Config values
        # ------------------------------------------------------------------
        self.dataset_id = self.cfg.dataset_id
        self.configuration = self.cfg.configuration
        self.fold = self.cfg.fold
        self.seed = self.cfg.get("seed", 12345)

        self.plans_identifier = self.cfg.get("plans_identifier", "nnUNetPlans")

        self.labeled_batch_size = self.cfg.get("labeled_batch_size", None)
        self.unlabeled_batch_size = self.cfg.get("unlabeled_batch_size", None)

        self.enable_deep_supervision = self.cfg.get(
            "enable_deep_supervision",
            True,
        )

        self.ds_scales = self.cfg.get("deep_supervision_scales", None)

        self.oversample_fg = self.cfg.get(
            "oversample_foreground_percent",
            0.33,
        )

        num_processes = self.cfg.get("num_processes", None)
        self.num_processes = (
            num_processes
            if num_processes is not None
            else get_allowed_n_proc_DA()
        )

        # ------------------------------------------------------------------
        # Load plans.json and dataset.json
        # ------------------------------------------------------------------
        self.dataset_name = maybe_convert_to_dataset_name(self.dataset_id)
        self.base = join(nnUNet_preprocessed, self.dataset_name)

        self.plans = load_json(join(self.base, self.plans_identifier + ".json"))
        self.dataset_json = load_json(join(self.base, "dataset.json"))

        # ------------------------------------------------------------------
        # nnU-Net managers
        # ------------------------------------------------------------------
        self.pm = PlansManager(self.plans)
        self.cm = self.pm.get_configuration(self.configuration)
        self.lm = self.pm.get_label_manager(self.dataset_json)

        self.folder = join(self.base, self.cm.data_identifier)

        # ------------------------------------------------------------------
        # Deep supervision scales
        # ------------------------------------------------------------------
        if self.enable_deep_supervision:
            if self.ds_scales is None:
                self.ds_scales = self._get_deep_supervision_scales_from_nnunet()
        else:
            self.ds_scales = None

        # ------------------------------------------------------------------
        # Cascade handling
        # ------------------------------------------------------------------
        self.is_cascaded = self.cm.previous_stage_name is not None
        self.prev_stage_folder = None

        if self.is_cascaded:
            self.prev_stage_folder = join(
                nnUNet_results,
                self.dataset_name,
                "nnUNetTrainer__"
                + self.pm.plans_name
                + "__"
                + self.cm.previous_stage_name,
                "predicted_next_stage",
                self.configuration,
            )

        # ------------------------------------------------------------------
        # SSL case IDs
        # ------------------------------------------------------------------
        ssl = self.dataset_json["ssl_case_ids"]

        self.trl = list(ssl["TrL"])

        tru = ssl.get("TrU")
        self.tru = list(tru) if tru else list(self.trl)

        self.lb = (
            self.labeled_batch_size
            if self.labeled_batch_size is not None
            else self.cm.batch_size
        )

        self.ub = (
            self.unlabeled_batch_size
            if self.unlabeled_batch_size is not None
            else self.cm.batch_size
        )

        self.splits_file = join(self.base, "splits_final_TrL.json")

        self.ds_class = None
        self._tr, self._val = None, None
        self._train_loader, self._val_loader = None, None

    # ------------------------------------------------------------------
    # nnU-Net shim helpers
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

        return nnUNetTrainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size(
            shim
        )

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def prepare_data(self):
        set_nnunet_env(**self._env)

        cls = infer_dataset_class(self.folder)

        cls.unpack_dataset(
            self.folder,
            overwrite_existing=False,
            num_processes=max(1, self.num_processes),
            verify=True,
        )

    def setup(self, stage=None):
        set_nnunet_env(**self._env)

        if self.ds_class is None:
            self.ds_class = infer_dataset_class(self.folder)

        if self._tr is None:
            self._tr, self._val = self._split_trl()

    def train_dataloader(self):
        if self._train_loader is None:
            lab = self._make_iter(
                keys=self._tr,
                batch_size=self.lb,
                oversample_fg=self.oversample_fg,
                train=True,
            )

            unl = self._make_iter(
                keys=self.tru,
                batch_size=self.ub,
                oversample_fg=0.0,
                train=True,
            )

            self._train_loader = CombinedLoader(
                {
                    "labeled": lab,
                    "unlabeled": unl,
                },
                mode="max_size_cycle",
            )

        return self._train_loader

    def val_dataloader(self):
        if self._val_loader is None:
            self._val_loader = self._make_iter(
                keys=self._val,
                batch_size=self.lb,
                oversample_fg=self.oversample_fg,
                train=False,
            )

        return self._val_loader

    def teardown(self, stage=None):
        for it in (self._train_loader, self._val_loader):
            children = (
                it.iterables.values()
                if isinstance(it, CombinedLoader)
                else ([it] if it else [])
            )

            for child in children:
                finish = getattr(child, "_finish", None)

                if callable(finish):
                    try:
                        finish()
                    except Exception:
                        pass

        self._train_loader = None
        self._val_loader = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _split_trl(self):
        if self.fold == "all":
            return self.trl, self.trl

        if not isfile(self.splits_file):
            splits = generate_crossval_split(
                list(np.sort(self.trl)),
                seed=self.seed,
                n_splits=5,
            )
            save_json(splits, self.splits_file)
        else:
            splits = load_json(self.splits_file)

        f = int(self.fold)

        if f < len(splits):
            return splits[f]["train"], splits[f]["val"]

        rnd = np.random.RandomState(seed=self.seed + f)

        keys = np.sort(self.trl)
        idx = rnd.choice(
            len(keys),
            int(len(keys) * 0.8),
            replace=False,
        )

        train_keys = [keys[i] for i in idx]
        val_keys = [keys[i] for i in range(len(keys)) if i not in idx]

        return train_keys, val_keys

    def _make_iter(self, keys, batch_size, oversample_fg, train):
        if self.enable_deep_supervision and self.ds_scales is None:
            raise RuntimeError(
                "enable_deep_supervision=True but self.ds_scales is None. "
                "DownsampleSegForDSTransform will not be added, so target "
                "will not become a list."
            )

        dataset = self.ds_class(
            self.folder,
            keys,
            folder_with_segs_from_previous_stage=self.prev_stage_folder,
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
                regions=self.lm.foreground_regions if self.lm.has_regions else None,
                ignore_label=self.lm.ignore_label,
            )

            patch = init_ps

        else:
            tfm = nnUNetTrainer.get_validation_transforms(
                deep_supervision_scales=self.ds_scales,
                is_cascaded=self.is_cascaded,
                foreground_labels=self.lm.foreground_labels,
                regions=self.lm.foreground_regions if self.lm.has_regions else None,
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

        if self.num_processes == 0:
            return SingleThreadedAugmenter(dl, None)

        if train:
            n_proc = self.num_processes
            n_cache = max(6, self.num_processes // 2)
        else:
            n_proc = max(1, self.num_processes // 2)
            n_cache = max(3, self.num_processes // 4)

        it = NonDetMultiThreadedAugmenter(
            data_loader=dl,
            transform=None,
            num_processes=n_proc,
            num_cached=n_cache,
            seeds=None,
            pin_memory=torch.cuda.is_available(),
            wait_time=0.002,
        )

        _ = next(it)

        return it
