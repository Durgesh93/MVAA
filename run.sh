#!/bin/bash
set -e

# ─────────────────────────────────────────────
# nnU-Net paths
# ─────────────────────────────────────────────
export nnUNet_raw="$(realpath dirs/data_storage/raw/MVAA_nnUNET_SSL)"
export nnUNet_preprocessed="$(realpath dirs/data_storage/raw/MVAA_nnUNET_processed_SSL)"
export nnUNet_results="$(realpath dirs/files/nnUNet_results)"

mkdir -p "$nnUNet_preprocessed"
mkdir -p "$nnUNet_results"


# ─────────────────────────────────────────────
# nnU-Net command path
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

# ─────────────────────────────────────────────
# Debug prints
# ─────────────────────────────────────────────
echo "nnUNet_raw          = $nnUNet_raw"
echo "nnUNet_preprocessed = $nnUNet_preprocessed"
echo "nnUNet_results      = $nnUNet_results"
echo "python              = $(which python)"
echo "nnUNetv2_train      = $(which nnUNetv2_train || true)"


TRAINER_KWARGS='{
  "num_epochs": 1,
  "num_iterations_per_epoch": 100,
  "num_val_iterations_per_epoch": 5,
  "lambda_pseudo": 0.0,
  "pseudo_conf_threshold": 0.8,
  "pseudo_foreground_only": true,
  "labeled_fraction": 0.5
}'

nnUNetv2_train 001 3d_fullres 0 \
  -tr nnUNetTrainerSSL \
  -num_gpus 6 \
  -trainer_kwargs "$TRAINER_KWARGS"

TRAINER_KWARGS='{
  "num_epochs": 20,
  "num_iterations_per_epoch": 100,
  "num_val_iterations_per_epoch": 5,
  "lambda_pseudo": 0.2,
  "pseudo_conf_threshold": 0.8,
  "pseudo_foreground_only": true,
  "labeled_fraction": 0.5
}'
nnUNetv2_train 001 3d_fullres 0 \
  -tr nnUNetTrainerSSL \
  -num_gpus 6 \
  -trainer_kwargs "$TRAINER_KWARGS"

# ─────────────────────────────────────────────
# Choose command
# ─────────────────────────────────────────────
# nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity -np 64
# nnUNetv2_plan_and_preprocess -d 002 --verify_dataset_integrity -np 64
# nnUNetv2_plan_and_preprocess -d 003 --verify_dataset_integrity -np 64


# #!/bin/bash
# set -e

# VIS_SCRIPT="data_preparation/visualize_data.py"
# BASE_DIR="dirs/data_storage/raw/MVAA_nnUNET_SSL"
# OUT_DIR="outputs/mvaa_visualization"

# N_RANDOM=3
# MAX_SLICES=10000
# ALPHA=0.8

# echo "Visualizing CT..."
# python "$VIS_SCRIPT" \
#   --dataset "$BASE_DIR/Dataset001_MVAA_CT_SSL" \
#   --split TrL \
#   --n_random "$N_RANDOM" \
#   --only_mask_slices \
#   --max_slices "$MAX_SLICES" \
#   --alpha "$ALPHA"

# echo "Visualizing TEE..."
# python "$VIS_SCRIPT" \
#   --dataset "$BASE_DIR/Dataset002_MVAA_TEE_SSL" \
#   --split TrL \
#   --n_random "$N_RANDOM" \
#   --only_mask_slices \
#   --max_slices "$MAX_SLICES" \
#   --alpha "$ALPHA"

# echo "Visualizing VIDEO..."
# python "$VIS_SCRIPT" \
#   --dataset "$BASE_DIR/Dataset003_MVAA_VIDEO_SSL" \
#   --split TrL \
#   --n_random "$N_RANDOM" \
#   --max_slices "$MAX_SLICES" \
#   --alpha "$ALPHA"

# echo "Done."
# echo "Figures saved under: $OUT_DIR"