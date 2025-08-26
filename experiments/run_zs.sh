#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:-/path/MMG.dcm}
PRIOR=${2:-prior.pth}
PYTHON=${PYTHON:-python}

$PYTHON main.py --mode zs \
  --input_image_path "$INPUT" \
  --upscale_factor 2 \
  --timesteps 100 \
  --prior_path "$PRIOR" \
  --zs_lambda 0.1