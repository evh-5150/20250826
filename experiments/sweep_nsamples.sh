#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:-/path/MMG.dcm}
MODEL=${2:-Results/.../model.pth}
PYTHON=${PYTHON:-python}

for NS in 1 3 5 7; do
  echo "=== n_samples=${NS} ==="
  $PYTHON main.py --mode inference \
    --input_image_path "$INPUT" \
    --upscale_factor 2 \
    --timesteps 100 \
    --n_samples ${NS} \
    --inf_patch_size 384 \
    --inf_overlap 160 \
    --interpolation_mode nearest-exact \
    --model_path "$MODEL" \
    --use_amp
done