#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────
# Visualization settings
# ─────────────────────────────────────────────
VIS_SCRIPT="data_preparation/visualize_data.py"
BASE_DIR="dirs/data_storage/raw/MVAA_nnUNET_SSL"
OUT_DIR="outputs/mvaa_visualization"

N_RANDOM=3
MAX_SLICES=10000
ALPHA=0.8

mkdir -p "$OUT_DIR"

echo "python      = $(which python)"
echo "VIS_SCRIPT  = $VIS_SCRIPT"
echo "BASE_DIR    = $BASE_DIR"
echo "OUT_DIR     = $OUT_DIR"

# ─────────────────────────────────────────────
# CT visualization
# ─────────────────────────────────────────────
echo ""
echo "Visualizing CT..."
python "$VIS_SCRIPT" \
  --dataset "$BASE_DIR/Dataset001_MVAA_CT_SSL" \
  --split TrL \
  --n_random "$N_RANDOM" \
  --only_mask_slices \
  --max_slices "$MAX_SLICES" \
  --alpha "$ALPHA"

# ─────────────────────────────────────────────
# TEE visualization
# ─────────────────────────────────────────────
echo ""
echo "Visualizing TEE..."
python "$VIS_SCRIPT" \
  --dataset "$BASE_DIR/Dataset002_MVAA_TEE_SSL" \
  --split TrL \
  --n_random "$N_RANDOM" \
  --only_mask_slices \
  --max_slices "$MAX_SLICES" \
  --alpha "$ALPHA"

# ─────────────────────────────────────────────
# VIDEO visualization
# ─────────────────────────────────────────────
echo ""
echo "Visualizing VIDEO..."
python "$VIS_SCRIPT" \
  --dataset "$BASE_DIR/Dataset003_MVAA_VIDEO_SSL" \
  --split TrL \
  --n_random "$N_RANDOM" \
  --max_slices "$MAX_SLICES" \
  --alpha "$ALPHA"

echo ""
echo "Done."
echo "Figures saved under: $OUT_DIR"