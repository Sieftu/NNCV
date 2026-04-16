#!/bin/bash
# Run once on the login node to install missing packages.
# Usage: bash setup_venv.sh
set -euo pipefail

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

pip install --user transformers timm wandb

python3 -c "import torch;        print('torch:       ', torch.__version__)"
python3 -c "import torchvision;  print('torchvision: ', torchvision.__version__)"
python3 -c "import transformers; print('transformers:', transformers.__version__)"
python3 -c "import timm;         print('timm:        ', timm.__version__)"
python3 -c "import wandb;        print('wandb:       ', wandb.__version__)"
