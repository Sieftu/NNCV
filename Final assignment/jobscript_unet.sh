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

# Path to Cityscapes on the cluster — verify with: ls /gpfs/work5/0/jhstue005/JHS_data/
DATA_ROOT="/gpfs/work5/0/jhstue005/JHS_data/CityScapes"
export DATA_ROOT

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Activate project venv (created once via setup_venv.sh)
source ~/venv_nncv/bin/activate

srun bash main.sh unet
