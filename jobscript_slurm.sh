#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=20:00:00
#SBATCH --job-name=nncv_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

# Default: train the SegFormer-B2 Peak submission. Override on the
# command line with `sbatch jobscript_slurm.sh unet` to train the U-Net.
ARCH="${1:-segformer_b2}"

srun apptainer exec --nv \
    --env-file .env \
    container.sif /bin/bash main.sh "$ARCH"
