#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:-/path/MMG.dcm}
PYTHON=${PYTHON:-python}

$PYTHON main.py --mode full \
  --input_image_path "$INPUT" \
  --upscale_factor 2 \
  --training_steps 1500 \
  --batch_size 4 \
  --patch_size 96 \
  --timesteps 100 \
  --lambda_l1 1.0 \
  --lambda_perceptual 0.1 \
  --interpolation_mode nearest-exact \
  --n_samples 2 \
  --inf_patch_size 512 \
  --inf_overlap 128 \
  --use_amp