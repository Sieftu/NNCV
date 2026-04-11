#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=02:00:00

srun apptainer exec --nv --env-file .env container.sif \
    python3 diagnostics.py \
        --data_root      ./data/cityscapes \
        --checkpoint     best_model-epoch=0055-val_loss=0.28399493731558323.pt \
        --out_dir        diagnostics_out \
        --wandb_project  5lsm0-cityscapes-segmentation \
        --wandb_run_name baseline-diagnostics
