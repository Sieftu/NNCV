#!/bin/bash
# Training entrypoint executed inside the Apptainer container.
# Usage: bash main.sh <arch>   where arch is one of: unet | segformer_b2
set -euo pipefail

ARCH="${1:?arch argument required. Usage: bash main.sh <unet|segformer_b2>}"

case "$ARCH" in
    unet)         TRAIN_SCRIPT="UNetRecipe/train.py" ;;
    segformer_b2) TRAIN_SCRIPT="PeakPerformance/train.py" ;;
    *) echo "Unknown arch: $ARCH" >&2; exit 1 ;;
esac

echo "Training arch=${ARCH} via ${TRAIN_SCRIPT}"
date

wandb login

python3 "$TRAIN_SCRIPT" \
    --arch           "${ARCH}" \
    --loss           dice_ce \
    --epochs         160 \
    --batch_size     8 \
    --crop_size      512 1024 \
    --data_root      ./data/cityscapes \
    --checkpoint_dir ./checkpoints \
    --wandb_run_name "${ARCH}" \
    --seed           42

echo "Done: arch=${ARCH}"
date
