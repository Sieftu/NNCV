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
        set -euo pipefail

        IMAGE=\$(find ./data/cityscapes/leftImg8bit/val -name '*.png' | sort | head -1 || true)
        if [ -z \"\$IMAGE\" ]; then
            echo 'ERROR: no val images found under ./data/cityscapes/leftImg8bit/val' >&2
            exit 1
        fi
        # Derive the matching gtFine labelIds path from the image path
        GT=\$(echo \"\$IMAGE\" \
            | sed 's|leftImg8bit/val|gtFine/val|' \
            | sed 's|_leftImg8bit\\.png|_gtFine_labelIds.png|')
        echo \"Image : \$IMAGE\"
        echo \"GT    : \$GT\"

        python3 generate_qualitative_figure.py \
            --image  \"\$IMAGE\" \
            --gt     \"\$GT\" \
            --unet-ckpt       checkpoints/unet/best_model_miou.pt \
            --segformer-ckpt  checkpoints/segformer_b2/best_model_miou.pt \
            --segformer-config /app/segformer_config \
            --output-dir      figures \
            --unet-preprocess starter
    "
