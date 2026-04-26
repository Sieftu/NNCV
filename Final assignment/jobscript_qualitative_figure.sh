#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:30:00
#SBATCH --job-name=qualitative_figure
#SBATCH --output=logs/qualitative_figure_%j.out
#SBATCH --error=logs/qualitative_figure_%j.err

mkdir -p logs

srun apptainer exec --nv \
    --env-file .env \
    container.sif /bin/bash -c "
        python3 generate_qualitative_figure.py \
            --image  data/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_001016_leftImg8bit.png \
            --gt     data/cityscapes/gtFine/val/frankfurt/frankfurt_000000_001016_gtFine_labelIds.png \
            --unet-ckpt       best_model_miou_unet.pt \
            --segformer-ckpt  best_model_miou_segformer.pt \
            --segformer-config segformer_config \
            --output-dir      figures \
            --unet-preprocess starter
    "
