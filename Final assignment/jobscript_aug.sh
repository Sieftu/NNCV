#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=06:00:00

srun apptainer exec --nv --env-file .env container.sif \
    python3 train.py \
        --data-dir       ./data/cityscapes \
        --epochs         50 \
        --batch-size     64 \
        --lr             0.001 \
        --num-workers    8 \
        --augmentation   aug_only \
        --use_amp \
        --eval_every     5 \
        --experiment-id  aug-only-50ep
