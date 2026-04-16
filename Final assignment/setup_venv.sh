#!/bin/bash
set -euo pipefail

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

echo "=== Packages already available from module ==="
python3 -c "import torch; print('torch:       ', torch.__version__)"
python3 -c "import torchvision; print('torchvision: ', torchvision.__version__)" 2>/dev/null \
    || echo "torchvision:  NOT available from module — will install in venv"

echo ""
echo "=== Checking for packages that need to be installed in venv ==="
for pkg in torchvision segmentation_models_pytorch timm wandb; do
    python3 -c "import ${pkg}; import importlib.metadata as m; print('${pkg}:', m.version('${pkg}'))" 2>/dev/null \
        && echo "  [OK]  ${pkg} already available" \
        || echo "  [--]  ${pkg} NOT available from module — will install in venv"
done

echo ""
echo "=== Creating venv ~/venv_nncv (inherits torch from module) ==="
python3 -m venv ~/venv_nncv --system-site-packages
source ~/venv_nncv/bin/activate

pip install --quiet --upgrade pip
# torchvision is not bundled with the Snellius PyTorch module — install the
# cu121 wheel that matches the CUDA version loaded by PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
pip install --quiet torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --quiet segmentation_models_pytorch timm wandb

echo ""
echo "=== Installed versions ==="
python3 -c "import torchvision; print('torchvision:  ', torchvision.__version__)"
python3 -c "import segmentation_models_pytorch as smp; print('smp:          ', smp.__version__)"
python3 -c "import timm;  print('timm:         ', timm.__version__)"
python3 -c "import wandb; print('wandb:        ', wandb.__version__)"
python3 -c "
import torchvision.transforms.v2 as T
has = hasattr(T, 'RandomResize')
print(f'torchvision.transforms.v2.RandomResize available: {has}')
if not has:
    print('  WARNING: upgrade torchvision in the venv with:')
    print('    pip install --upgrade torchvision')
"

echo ""
echo "=== Setup complete. Activate with: source ~/venv_nncv/bin/activate ==="
