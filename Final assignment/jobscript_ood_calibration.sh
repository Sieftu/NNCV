#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=01:00:00
#SBATCH --job-name=ood_calibration
#SBATCH --output=logs/ood_calibration_%j.out
#SBATCH --error=logs/ood_calibration_%j.err

mkdir -p logs

srun apptainer exec --nv \
    --env-file .env \
    container.sif /bin/bash -c "
        set -euo pipefail

        echo '=== Step A: Fit temperature scalar ==='
        date
        python3 fit_temperature.py \
            --checkpoint checkpoints/segformer/best_model_miou.pt \
            --data_root  ./data/cityscapes

        TEMPERATURE=\$(python3 -c \"import json; print(json.load(open('temperature.json'))['temperature'])\")
        echo \"Fitted temperature: \$TEMPERATURE\"

        echo ''
        echo '=== Step B: Calibrate all OOD methods ==='
        date
        python3 calibrate_ood.py \
            --checkpoint checkpoints/segformer/best_model_miou.pt \
            --data_root  ./data/cityscapes \
            --temperature \$TEMPERATURE

        echo ''
        echo '=== temperature.json ==='
        cat temperature.json

        echo ''
        echo '=== ood_thresholds.json ==='
        cat ood_thresholds.json

        echo ''
        echo '=== Done ==='
        date
    "
