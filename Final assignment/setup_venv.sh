#!/bin/bash
set -euo pipefail

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

python3 -m venv ~/venv_nncv
source ~/venv_nncv/bin/activate

pip install --no-deps segmentation_models_pytorch wandb

echo "--- Verification ---"
python3 -c "import torch; print('torch:       ', torch.__version__)"
python3 -c "import torchvision; print('torchvision:', torchvision.__version__)"
python3 -c "import segmentation_models_pytorch as smp; print('smp:         ', smp.__version__)"
python3 -c "import wandb; print('wandb:       ', wandb.__version__)"
python3 -c "import torch; print('CUDA Ready:  ', torch.cuda.is_available())"