"""
Lightning trainer builder for SSL nnU-Net experiments.

Usage:
    from lit.trainer import build_trainer

    trainer = build_trainer(cfg)
    trainer.fit(model, datamodule=dm)
"""

import lightning as L


def build_trainer(cfg):
    """
    Build Lightning Trainer from Hydra/OmegaConf config.

    nnU-Net-style controls:
        cfg.trainer.num_train_iters_per_epoch
        cfg.trainer.num_val_iters_per_epoch

    Example:
        max_epochs: 25
        num_train_iters_per_epoch: 250
        num_val_iters_per_epoch: 50

    This gives:
        25 epochs
        250 train batches per epoch
        50 val batches per validation phase
    """

    trainer_cfg = cfg.trainer

    trainer_kwargs = dict(
        max_epochs=trainer_cfg.get("max_epochs", 25),
        accelerator=trainer_cfg.get("accelerator", "gpu"),
        devices=trainer_cfg.get("devices", 1),
        precision=trainer_cfg.get("precision", "16-mixed"),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 1),
        check_val_every_n_epoch=trainer_cfg.get("check_val_every_n_epoch", 1),
    )

    # ------------------------------------------------------------------
    # nnU-Net style epoch length
    # ------------------------------------------------------------------
    trainer_kwargs["limit_train_batches"] = trainer_cfg.get(
        "num_train_iters_per_epoch",
        250,
    )

    trainer_kwargs["limit_val_batches"] = trainer_cfg.get(
        "num_val_iters_per_epoch",
        50,
    )

    # ------------------------------------------------------------------
    # Optional Lightning settings
    # ------------------------------------------------------------------
    if "strategy" in trainer_cfg and trainer_cfg.strategy is not None:
        trainer_kwargs["strategy"] = trainer_cfg.strategy

    if "gradient_clip_val" in trainer_cfg and trainer_cfg.gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = trainer_cfg.gradient_clip_val

    if "accumulate_grad_batches" in trainer_cfg:
        trainer_kwargs["accumulate_grad_batches"] = trainer_cfg.accumulate_grad_batches

    if "num_sanity_val_steps" in trainer_cfg:
        trainer_kwargs["num_sanity_val_steps"] = trainer_cfg.num_sanity_val_steps

    if "enable_checkpointing" in trainer_cfg:
        trainer_kwargs["enable_checkpointing"] = trainer_cfg.enable_checkpointing

    if "enable_progress_bar" in trainer_cfg:
        trainer_kwargs["enable_progress_bar"] = trainer_cfg.enable_progress_bar

    if "deterministic" in trainer_cfg:
        trainer_kwargs["deterministic"] = trainer_cfg.deterministic

    if "default_root_dir" in trainer_cfg:
        trainer_kwargs["default_root_dir"] = trainer_cfg.default_root_dir

    if "logger" in trainer_cfg:
        trainer_kwargs["logger"] = trainer_cfg.logger

    return L.Trainer(**trainer_kwargs)
