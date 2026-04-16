#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=20:00:00
#SBATCH --job-name=segformer_b2_seg
#SBATCH --output=logs/segformer_b2_%j.out
#SBATCH --error=logs/segformer_b2_%j.err

mkdir -p logs

# Path to Cityscapes on the cluster — verify with: ls /gpfs/work5/0/jhstue005/JHS_data/
DATA_ROOT="/gpfs/work5/0/jhstue005/JHS_data/CityScapes"
export DATA_ROOT

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

srun bash main.sh segformer_b2
