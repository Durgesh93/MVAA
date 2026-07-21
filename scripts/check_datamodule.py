"""
Standalone smoke test for SSLnnUNetDataModule's TrL/TrU dataloaders --
no real training loop, no Lightning Trainer.fit(). Instantiates the
datamodule directly against real preprocessed data, fakes a minimal
trainer so setup() can run, pulls one batch from train_dataloader(),
and asserts shapes/keys match what training_step will eventually expect.

Usage:
    python scripts/check_datamodule.py
"""

from types import SimpleNamespace

import torch

from config import build_config
from utils import set_nnunet_env
from datamodule import SSLnnUNetDataModule

CONFIG_MAP = {"ct": "experiment_CT", "tee": "experiment_TEE", "video": "experiment_video"}


def check_task(task_name, config_name, k_override=3):
    print(f"\n{'=' * 80}\n[{task_name}] config={config_name}\n{'=' * 80}")

    cfg = build_config(
        config_name=config_name,
        overrides=[
            "fold=all",
            f"datamodule.K={k_override}",
        ],
    )
    set_nnunet_env(cfg)

    dm = SSLnnUNetDataModule(cfg.datamodule)
    dm.trainer = SimpleNamespace(world_size=1, global_rank=0, limit_train_batches=None)

    dm.setup()

    loader = dm.train_dataloader()
    batch, batch_idx, dataloader_idx = next(iter(loader))

    labeled = batch["labeled"]
    unlabeled = batch["unlabeled"]

    assert "data" in labeled, f"labeled batch missing 'data' key, got {list(labeled.keys())}"
    assert "data_views" not in labeled, "labeled batch should NOT have 'data_views' (K=1, drop-in shape)"

    labeled_data = labeled["data"]
    print(f"labeled['data'].shape = {tuple(labeled_data.shape)}")
    assert labeled_data.shape[0] == dm.batch_size
    assert labeled_data.shape[1] == dm.num_channels

    assert "data_views" in unlabeled, f"unlabeled batch missing 'data_views' key, got {list(unlabeled.keys())}"
    data_views = unlabeled["data_views"]
    assert isinstance(data_views, list), f"data_views should be a list, got {type(data_views)}"
    assert len(data_views) == dm.K, f"expected {dm.K} views, got {len(data_views)}"

    for v, view in enumerate(data_views):
        print(f"unlabeled['data_views'][{v}].shape = {tuple(view.shape)}")
        assert view.shape == labeled_data.shape, "each TrU view should match labeled 'data' shape"
        assert not torch.isnan(view).any(), f"view {v} contains NaN"
        assert not torch.isinf(view).any(), f"view {v} contains Inf"

    if dm.K >= 2:
        assert not torch.equal(data_views[0], data_views[1]), (
            "view 0 and view 1 are identical -- intensity draws should differ "
            "independently even though they share one geometric draw"
        )

    assert not torch.isnan(labeled_data).any(), "labeled data contains NaN"
    assert not torch.isinf(labeled_data).any(), "labeled data contains Inf"

    print(f"[{task_name}] OK -- TrL={len(dm.trl_all)} TrU={len(dm.tru_all)} K={dm.K}")


def check_shipped_defaults_are_weak_strong(task_name, config_name):
    """
    Shipped defaults: K=2, transform_geometric=True -- TrU intensity
    augmentation and the weak/strong split are both unconditional now
    (single-view self-training was removed). data_views[0] is the weak
    (geometric-only, including rotation/scaling) view, data_views[1] is
    the strong (weak + real intensity aug) view. Confirms the two diverge
    (proving the strong view's intensity draw actually ran).
    """
    print(f"\n{'=' * 80}\n[{task_name}] shipped defaults (weak/strong TrU)\n{'=' * 80}")

    cfg = build_config(config_name=config_name, overrides=["fold=all"])
    set_nnunet_env(cfg)

    dm = SSLnnUNetDataModule(cfg.datamodule)
    dm.trainer = SimpleNamespace(world_size=1, global_rank=0, limit_train_batches=None)
    dm.setup()

    assert dm.transform_geometric is True
    assert dm.K == 2

    loader = dm.train_dataloader()
    batch, batch_idx, dataloader_idx = next(iter(loader))
    data_views = batch["unlabeled"]["data_views"]

    assert len(data_views) == 2, f"expected 2 views (weak + strong), got {len(data_views)}"
    assert not torch.equal(data_views[0], data_views[1]), (
        "weak view (data_views[0]) and strong view (data_views[1]) are identical -- "
        "the strong view's intensity draw should diverge from the untouched weak view"
    )
    print(f"[{task_name}] OK -- shipped defaults confirmed weak/strong split (views diverge as expected)")


if __name__ == "__main__":
    for task_name, config_name in CONFIG_MAP.items():
        check_task(task_name, config_name)

    for task_name, config_name in CONFIG_MAP.items():
        check_shipped_defaults_are_weak_strong(task_name, config_name)

    print("\nAll tasks OK.")
