#!/bin/bash
# Training launcher called by Slurm job scripts.
# Modules and venv are activated by the job script before calling this.
# Usage: bash main.sh <arch>   where arch is one of: unet | segformer_b2
set -euo pipefail

ARCH="${1:?ERROR: arch argument required. Usage: bash main.sh <arch>}"

# Load WANDB_API_KEY and any other secrets from .env
set -a
source .env
set +a

echo "=== Training run: arch=${ARCH} ==="
date

wandb login

python3 train.py \
    --arch           "${ARCH}" \
    --loss           dice_ce \
    --epochs         160 \
    --batch_size     8 \
    --crop_size      512 1024 \
    --data_root      "${DATA_ROOT}" \
    --checkpoint_dir ./checkpoints \
    --wandb_run_name "${ARCH}" \
    --seed           42

echo "=== Done: arch=${ARCH} ==="
date
