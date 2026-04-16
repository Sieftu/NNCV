#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=10:00:00
#SBATCH --job-name=unet_seg
#SBATCH --output=logs/unet_%j.out
#SBATCH --error=logs/unet_%j.err

mkdir -p logs

srun apptainer exec --nv --env-file .env container.sif /bin/bash main.sh unet
