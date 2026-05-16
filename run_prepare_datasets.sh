#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────
# nnU-Net paths
# ─────────────────────────────────────────────
export nnUNet_raw="$(realpath dirs/data_storage/raw/MVAA_nnUNET_SSL)"
export nnUNet_preprocessed="$(realpath dirs/data_storage/raw/MVAA_nnUNET_processed_SSL)"
export nnUNet_results="$(realpath dirs/files/nnUNet_results)"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

# ─────────────────────────────────────────────
# Command path
# ─────────────────────────────────────────────
export PATH="/scratch/project_465002860/durgeshk/tmp/pip_userbase/bin:$PATH"

# ─────────────────────────────────────────────
# Thread settings
# ─────────────────────────────────────────────
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1

echo "nnUNet_raw          = $nnUNet_raw"
echo "nnUNet_preprocessed = $nnUNet_preprocessed"
echo "nnUNet_results      = $nnUNet_results"
echo "python              = $(which python)"
echo "nnUNetv2_plan_and_preprocess = $(which nnUNetv2_plan_and_preprocess || true)"

# ─────────────────────────────────────────────
# Dataset preparation
# ─────────────────────────────────────────────

NPROC=64

echo ""
echo "Planning and preprocessing Dataset001_MVAA_CT_SSL..."
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity -np "$NPROC"

echo ""
echo "Planning and preprocessing Dataset002_MVAA_TEE_SSL..."
nnUNetv2_plan_and_preprocess -d 002 --verify_dataset_integrity -np "$NPROC"

echo ""
echo "Planning and preprocessing Dataset003_MVAA_VIDEO_SSL..."
nnUNetv2_plan_and_preprocess -d 003 --verify_dataset_integrity -np "$NPROC"

echo ""
echo "Done: planning and preprocessing finished."