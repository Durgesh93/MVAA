"""
Shared helpers for the data_preparation scripts: typer logging, a file
copy, and nnU-Net environment setup.

This module used to carry the whole SSL dataset-writing pipeline --
label detection by scanning every mask, sample/mask dataframe assembly,
sequential case renaming, dummy-mask synthesis for the unlabeled TrU
split, and a generic write_dataset() parameterized by per-task writer
callbacks. All of it existed for the data_task{1,2,3}_*_nnunet.py
scripts, which are gone: MVSeg2023 is fully labeled, so there is no
TrL/TrU/Ts scheme to build and no unlabeled partition to synthesize.
data_mvseg2023_nnunet.py writes its own (much shorter) dataset directly
and needs only the helpers below.
"""

from pathlib import Path
import os
import shutil

import typer


# =============================================================================
# Typer logging helpers
# =============================================================================


def log_section(title):
    typer.echo()
    typer.echo("=" * 80)
    typer.secho(title, fg=typer.colors.CYAN, bold=True)
    typer.echo("=" * 80)


def log_info(message):
    typer.secho(f"[INFO] {message}", fg=typer.colors.CYAN)


def log_ok(message):
    typer.secho(f"[OK] {message}", fg=typer.colors.GREEN)


def log_warn(message):
    typer.secho(f"[WARN] {message}", fg=typer.colors.YELLOW)


def log_flag(message):
    typer.secho(f"[FLAG] {message}", fg=typer.colors.MAGENTA)


def print_group_counts(df, group_cols, title):
    typer.echo()
    typer.secho(title, fg=typer.colors.CYAN, bold=True)

    if df.empty:
        log_warn("No rows to show.")
        return

    grouped = df.groupby(group_cols).size().reset_index(name="count")
    typer.echo(grouped.to_string(index=False))


# =============================================================================
# Common helpers
# =============================================================================


def copy_file(src_path, dst_path):
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def set_nnunet_env(nnunet_raw, nnunet_preprocessed, nnunet_results):
    nnunet_raw = Path(nnunet_raw).resolve()
    nnunet_preprocessed = Path(nnunet_preprocessed).resolve()
    nnunet_results = Path(nnunet_results).resolve()

    nnunet_raw.mkdir(parents=True, exist_ok=True)
    nnunet_preprocessed.mkdir(parents=True, exist_ok=True)
    nnunet_results.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(nnunet_raw)
    os.environ["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    os.environ["nnUNet_results"] = str(nnunet_results)

    log_info(f"nnUNet_raw          : {os.environ['nnUNet_raw']}")
    log_info(f"nnUNet_preprocessed : {os.environ['nnUNet_preprocessed']}")
    log_info(f"nnUNet_results      : {os.environ['nnUNet_results']}")
