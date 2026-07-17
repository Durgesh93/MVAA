"""
Standalone smoke test for pseudo-label training_step -- no real
Trainer.fit(). Builds a real SSLnnUNetDataModule + SSLnnUNetLightningModule
for each task, pulls one real batch, and calls training_step directly.

Usage:
    python scripts/check_pseudo_loss.py
"""

from types import SimpleNamespace

import torch

from config import build_config
from utils import set_nnunet_env
from datamodule import SSLnnUNetDataModule
from module import SSLnnUNetLightningModule

CONFIG_MAP = {"ct": "experiment_CT", "tee": "experiment_TEE", "video": "experiment_video"}


def _fake_trainer(world_size=1, global_rank=0, current_epoch=0):
    return SimpleNamespace(
        world_size=world_size,
        global_rank=global_rank,
        is_global_zero=True,
        current_epoch=current_epoch,
        sanity_checking=False,
    )


def check_task(task_name, config_name):
    print(f"\n{'=' * 80}\n[{task_name}] config={config_name}\n{'=' * 80}")

    cfg = build_config(config_name=config_name, overrides=["fold=all"])
    set_nnunet_env(cfg)

    dm = SSLnnUNetDataModule(cfg.datamodule)
    dm.trainer = SimpleNamespace(world_size=1, global_rank=0, limit_train_batches=None)
    dm.setup()

    loader = dm.train_dataloader()
    batch, batch_idx, dataloader_idx = next(iter(loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SSLnnUNetLightningModule(cfg.litmodule)
    model.trainer = _fake_trainer(current_epoch=0)
    model.setup(stage="fit")
    model.to(device)

    # --- warmup: pseudo_loss must be exactly 0 before pseudo_warmup_epochs ---
    total_loss = model.training_step(batch, batch_idx=0)
    assert torch.isfinite(total_loss), f"total_loss not finite: {total_loss}"
    assert total_loss.requires_grad, "total_loss must require grad"
    pseudo_metric = model.metrics.step_metrics["train_pseudo_loss"].compute()
    assert pseudo_metric == 0.0, f"expected pseudo_loss==0 during warmup, got {pseudo_metric}"
    model.metrics.reset_step_metrics()
    print(f"[{task_name}] warmup OK: total_loss={total_loss.item():.4f}")

    # --- after warmup: pseudo_loss generally nonzero, still finite ---
    model.trainer = _fake_trainer(current_epoch=cfg.litmodule.pseudo_warmup_epochs + 5)
    total_loss = model.training_step(batch, batch_idx=0)
    assert torch.isfinite(total_loss)
    assert total_loss.requires_grad
    total_loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.network.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and any(g > 0 for g in grad_norms), "no gradient reached network params"
    pseudo_metric = model.metrics.step_metrics["train_pseudo_loss"].compute()
    print(f"[{task_name}] post-warmup OK: total_loss={total_loss.item():.4f} pseudo_loss={pseudo_metric.item():.4f}")
    model.metrics.reset_step_metrics()
    model.zero_grad()

    # --- zero-confident-pixel edge case: must not NaN ---
    model.pseudo_loss_fn.threshold = 1.1  # softmax max prob is always <= 1.0
    pseudo_loss = model._pseudo_loss(batch["unlabeled"])
    assert torch.isfinite(pseudo_loss) and pseudo_loss.item() == 0.0
    print(f"[{task_name}] zero-confident-pixel edge case OK: pseudo_loss={pseudo_loss.item()}")


if __name__ == "__main__":
    for task_name, config_name in CONFIG_MAP.items():
        check_task(task_name, config_name)

    print("\nAll tasks OK.")
