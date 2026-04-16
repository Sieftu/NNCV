#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=00:30:00
#SBATCH --job-name=smoke_test
#SBATCH --output=logs/smoke_test_%j.out
#SBATCH --error=logs/smoke_test_%j.err

mkdir -p logs

# Path to Cityscapes on the cluster — verify with: ls /gpfs/work5/0/jhstue005/JHS_data/
DATA_ROOT="/gpfs/work5/0/jhstue005/JHS_data/CityScapes"
export DATA_ROOT

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Load secrets for wandb
set -a; source .env; set +a
wandb login

srun python3 train.py --arch segformer_b2 --smoke_test \
    --data_root "${DATA_ROOT}" --checkpoint_dir ./checkpoints \
    --wandb_run_name smoke_test_segformer_b2 --seed 42

srun python3 train.py --arch unet --smoke_test \
    --data_root "${DATA_ROOT}" --checkpoint_dir ./checkpoints \
    --wandb_run_name smoke_test_unet --seed 42
