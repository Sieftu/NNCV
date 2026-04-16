#!/bin/bash
# Run once on the login node to set up the project venv.
# Usage: bash setup_venv.sh
set -euo pipefail

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

python3 -m venv ~/venv_nncv --system-site-packages
source ~/venv_nncv/bin/activate

pip install segmentation_models_pytorch wandb

python3 -c "import torch;       print('torch:      ', torch.__version__)"
python3 -c "import torchvision; print('torchvision:', torchvision.__version__)"
python3 -c "import segmentation_models_pytorch as smp; print('smp:        ', smp.__version__)"
python3 -c "import wandb;       print('wandb:      ', wandb.__version__)"
