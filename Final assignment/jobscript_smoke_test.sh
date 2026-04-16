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

srun apptainer exec --nv --env-file .env container.sif /bin/bash -c "
    wandb login
    python3 train.py --arch segformer_b2 --smoke_test --data_root ./data/cityscapes --checkpoint_dir ./checkpoints --wandb_run_name smoke_test_segformer_b2 --seed 42
    python3 train.py --arch unet         --smoke_test --data_root ./data/cityscapes --checkpoint_dir ./checkpoints --wandb_run_name smoke_test_unet         --seed 42
"
