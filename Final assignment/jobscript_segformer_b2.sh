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

srun apptainer exec --nv \
    --env-file .env \
    --bind "${DATA_ROOT}:./data/cityscapes" \
    container.sif /bin/bash main.sh segformer_b2
