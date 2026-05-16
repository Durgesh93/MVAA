#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────
# nnU-Net paths
# ─────────────────────────────────────────────
export nnUNet_raw="$(realpath dirs/data_storage/raw/MVAA_nnUNET_SSL)"
export nnUNet_preprocessed="$(realpath dirs/data_storage/raw/MVAA_nnUNET_processed_SSL)"
export nnUNet_results="$(realpath dirs/files/nnUNet_results)"

mkdir -p "$nnUNet_preprocessed" "$nnUNet_results"

# ─────────────────────────────────────────────
# Command path
# ─────────────────────────────────────────────
export PATH="/scratch/project_465002860/durgeshk/tmp/pip_userbase/bin:$PATH"

# ─────────────────────────────────────────────
# ROCm / MIOpen settings
# ─────────────────────────────────────────────
export MIOPEN_ENABLE_LOGGING=0
export MIOPEN_ENABLE_LOGGING_CMD=0
export MIOPEN_LOG_LEVEL=1
export AMD_LOG_LEVEL=0
export HIP_LOG_LEVEL=0
export nnUNet_compile=f

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
echo "nnUNetv2_train      = $(which nnUNetv2_train || true)"

# ─────────────────────────────────────────────
# Shared settings
# ─────────────────────────────────────────────
CONFIG="3d_fullres"
FOLD="0"
TRAINER="nnUNetTrainerSSL"
NUM_GPUS=6

NUM_EPOCHS=20
TRAIN_ITERS=100
VAL_ITERS=10
PSEUDO_CONF=0.8
LABELED_FRACTION=0.5

make_kwargs () {
  local lambda_pseudo="$1"

  cat <<EOF
{
  "num_epochs": $NUM_EPOCHS,
  "num_iterations_per_epoch": $TRAIN_ITERS,
  "num_val_iterations_per_epoch": $VAL_ITERS,
  "lambda_pseudo": $lambda_pseudo,
  "pseudo_conf_threshold": $PSEUDO_CONF,
  "pseudo_foreground_only": true,
  "labeled_fraction": $LABELED_FRACTION
}
EOF
}

run_train () {
  local dataset="$1"
  local lambda_pseudo="$2"
  local tag="$3"

  echo ""
  echo "─────────────────────────────────────────────"
  echo "TRAINING | dataset=$dataset | lambda_pseudo=$lambda_pseudo | mode=$tag"
  echo "─────────────────────────────────────────────"

  TRAINER_KWARGS="$(make_kwargs "$lambda_pseudo")"

  nnUNetv2_train "$dataset" "$CONFIG" "$FOLD" \
    -tr "$TRAINER" \
    -num_gpus "$NUM_GPUS" \
    -trainer_kwargs "$TRAINER_KWARGS"
}

run_val_best () {
  local dataset="$1"
  local lambda_pseudo="$2"
  local tag="$3"

  echo ""
  echo "─────────────────────────────────────────────"
  echo "VALIDATION | dataset=$dataset | lambda_pseudo=$lambda_pseudo | mode=$tag | checkpoint_best"
  echo "─────────────────────────────────────────────"

  TRAINER_KWARGS="$(make_kwargs "$lambda_pseudo")"

  nnUNetv2_train "$dataset" "$CONFIG" "$FOLD" \
    -tr "$TRAINER" \
    -num_gpus "$NUM_GPUS" \
    --val \
    --val_best \
    -trainer_kwargs "$TRAINER_KWARGS"
}

# ─────────────────────────────────────────────
# Optional preprocessing
# ─────────────────────────────────────────────
# nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity -np 64
# nnUNetv2_plan_and_preprocess -d 002 --verify_dataset_integrity -np 64
# nnUNetv2_plan_and_preprocess -d 003 --verify_dataset_integrity -np 64

# ─────────────────────────────────────────────
# 1. Training runs first
# ─────────────────────────────────────────────

# Dataset001 CT
run_train 001 0.0 "supervised_only"
run_train 001 0.3 "pseudo_label"

# Dataset002 TEE
run_train 002 0.0 "supervised_only"
run_train 002 0.3 "pseudo_label"

# ─────────────────────────────────────────────
# 2. Full-volume validation after all training
# ─────────────────────────────────────────────

# Dataset001 CT
run_val_best 001 0.0 "supervised_only"
run_val_best 001 0.3 "pseudo_label"

# Dataset002 TEE
run_val_best 002 0.0 "supervised_only"
run_val_best 002 0.3 "pseudo_label"

echo ""
echo "All training and validation runs finished."
