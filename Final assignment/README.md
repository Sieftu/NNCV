# 5LSM0 Final Assignment - Cityscapes Challenge

Code accompanying the final assignment of **5LSM0 Neural Networks for Computer
Vision** at the Department of Electrical Engineering, Eindhoven University of
Technology. The repository contains a baseline U-Net submission, a recipe-trained
U-Net used for the ablation study, a SegFormer-B2 submission for the Peak
Performance benchmark, and four scoring variants for the Out-of-Distribution
benchmark.

- Author: `<<FILL IN: Thor Lastname>>`
- TU/e email: `<<FILL IN: t.lastname@student.tue.nl>>`

## Submission-server usernames

| Username | Server | Experiment | MeanDice / MIoU |
|---|---|---|---|
| `Thor_Baseline_unet` | Peak | Baseline U-Net (starter recipe, 50 ep, 256x256, CE) | 0.520 / 0.414 |
| `Thor_Seg_PP` | Peak | SegFormer-B2 (full recipe, 160 ep, 512x1024, Dice+CE) | 0.573 / 0.468 |
| `Thor_ood_msp` | OOD | SegFormer-B2 + Maximum Softmax Probability | 0.518 / 0.445 |
| `Thor_ood_temp_msp` | OOD | SegFormer-B2 + Temperature-scaled MSP (T = 1.42) | 0.504 / 0.434 |
| `Thor_ood_energy` | OOD | SegFormer-B2 + Energy score (T = 1.42) | 0.608 / 0.518 |
| `Thor_ood_pixel_unc` | OOD | SegFormer-B2 + Pixel Uncertainty Voting | 0.475 / 0.410 |

## Repository structure

```
.
├── README.md                    # this file
├── .gitignore
├── .env.example                 # template for WANDB credentials
├── main.sh                      # training entrypoint executed inside the container
├── jobscript_slurm.sh           # SLURM submission script
├── download_docker_and_data.sh  # one-shot fetch of dataset and Apptainer container
│
├── READMEs/                     # course-provided originals, kept verbatim
│   ├── README.md                # final-assignment overview from the course
│   ├── README-Report.md         # report-writing guidelines
│   ├── README-Submission.md     # Docker contract for the challenge servers
│   ├── README-Installation.md   # local environment setup
│   ├── README-Slurm.md          # HPC workflow
│   └── img/
│
├── Baseline/                    # mandatory baseline submission (Thor_Baseline_unet)
│   ├── model.py                 # course-provided U-Net
│   ├── train.py                 # starter trainer (256x256, CE, 0.5/0.5 normalisation)
│   ├── predict.py               # starter inference matching the baseline checkpoint
│   └── Dockerfile
│
├── UNetRecipe/                  # ablation: U-Net trained with the SegFormer recipe
│   ├── model.py                 # same U-Net architecture as Baseline
│   ├── train.py                 # full recipe (160 ep, 512x1024, Dice+CE, ImageNet norm)
│   ├── predict.py               # native-resolution inference, ImageNet norm
│   └── losses.py                # Dice + DiceCE loss
│
├── PeakPerformance/             # main Peak submission (Thor_Seg_PP)
│   ├── model.py                 # SegFormer-B2 wrapper (HuggingFace)
│   ├── train.py                 # same recipe as UNetRecipe
│   ├── predict.py               # native-resolution inference, ImageNet norm
│   ├── losses.py
│   ├── segformer_config/        # local HF config so the container needs no internet
│   │   └── config.json
│   └── Dockerfile
│
└── OOD/                         # Out-of-Distribution submissions (Thor_ood_*)
    ├── model.py                 # SegFormer-B2 backbone + OOD scoring head
    ├── ood_scoring.py           # msp, energy, pixel_uncertainty, entropy
    ├── predict.py               # OOD inference (mask + predictions.csv)
    ├── calibrate_ood.py         # 99th-percentile thresholds on Cityscapes val
    ├── fit_temperature.py       # NLL-minimising temperature scalar
    ├── configs/
    │   ├── ood_config_msp.json
    │   ├── ood_config_temp_msp.json
    │   ├── ood_config_energy.json
    │   └── ood_config_pixel_uncertainty.json
    ├── segformer_config/
    │   └── config.json
    └── Dockerfile
```

## Environment setup

### HPC cluster (Snellius)

The shared course container `cclaess/5lsm0:v1` carries PyTorch, torchvision, the
HuggingFace stack used by SegFormer, and `wandb`. Pull it once and grab the
Cityscapes data:

```bash
chmod +x download_docker_and_data.sh
sbatch download_docker_and_data.sh
```

This produces `container.sif` and a populated `data/cityscapes/` directory next
to the repository root.

Copy `.env.example` to `.env` and fill in your Weights & Biases credentials:

```bash
cp .env.example .env
$EDITOR .env   # set WANDB_API_KEY and WANDB_DIR
```

Submit a training job with:

```bash
sbatch jobscript_slurm.sh
```

`jobscript_slurm.sh` calls `main.sh segformer_b2` inside the container.

### Local machine

The dependencies imported by `train.py`, `predict.py` and the OOD scripts are:

```bash
pip install torch torchvision pillow transformers timm wandb numpy tqdm matplotlib
```

`tqdm` and `matplotlib` are optional (used only for the calibration progress
bar and figures, respectively).

## Data setup

`download_docker_and_data.sh` calls `huggingface-cli` inside the container to
fetch the dataset from `TimJaspersTue/5LSM0`. The resulting layout is
`./data/cityscapes/{leftImg8bit,gtFine}/{train,val,test}/...` and is what every
training script expects via `--data_root ./data/cityscapes`. No further
preprocessing is required.

## Training

All commands assume the working directory is the repository root.

### Baseline (starter U-Net)

```bash
python Baseline/train.py \
  --data-dir ./data/cityscapes \
  --batch-size 64 \
  --epochs 50 \
  --lr 0.001 \
  --num-workers 10 \
  --seed 42 \
  --experiment-id Thor_Baseline_unet
```

Trains the starter U-Net with the course-provided recipe (256x256 input,
cross-entropy loss, mean/std = 0.5). Best checkpoint is written to
`checkpoints/Thor_Baseline_unet/best_model-epoch=...pt`. Copy it to
`Baseline/model.pt` before building the Docker image.

### UNet Recipe (ablation only - not submitted to the server)

```bash
python UNetRecipe/train.py \
  --arch unet \
  --epochs 160 \
  --batch_size 8 \
  --crop_size 512 1024 \
  --loss dice_ce \
  --data_root ./data/cityscapes \
  --checkpoint_dir ./checkpoints \
  --wandb_run_name unet_recipe \
  --seed 42
```

Best validation mIoU 0.607 on Cityscapes val. Used in the report's ablation
table to isolate the contribution of training recipe versus architecture.

### Peak Performance (SegFormer-B2)

```bash
python PeakPerformance/train.py \
  --arch segformer_b2 \
  --epochs 160 \
  --batch_size 8 \
  --crop_size 512 1024 \
  --loss dice_ce \
  --data_root ./data/cityscapes \
  --checkpoint_dir ./checkpoints \
  --wandb_run_name Thor_Seg_PP \
  --seed 42
```

Best validation mIoU 0.892, test mIoU 0.468 (`Thor_Seg_PP`). The same
checkpoint is reused as the OOD backbone.

## OOD threshold calibration

After training the SegFormer-B2 checkpoint, calibrate the four OOD scoring
methods on the Cityscapes val set:

```bash
cd OOD
python fit_temperature.py \
  --checkpoint ../checkpoints/Thor_Seg_PP/best_model_miou.pt \
  --data_root ../data/cityscapes

python calibrate_ood.py \
  --checkpoint ../checkpoints/Thor_Seg_PP/best_model_miou.pt \
  --data_root ../data/cityscapes \
  --temperature $(python -c "import json; print(json.load(open('temperature.json'))['temperature'])")
```

`fit_temperature.py` writes `temperature.json` (T \approx 1.42 with the
shipped checkpoint). `calibrate_ood.py` writes `ood_thresholds.json`, whose
four entries populate the four `configs/ood_config_*.json` files. Each config
is wired into a separate Docker image at build time (next section).

## Building and submitting Docker images

Each experiment folder is a self-contained build context. Place the trained
checkpoint at `<folder>/model.pt` before building.

### Baseline

```bash
cp checkpoints/Thor_Baseline_unet/best_model-epoch=*.pt Baseline/model.pt
docker build -t nncv:baseline -f Baseline/Dockerfile Baseline/
docker save -o submission_baseline.tar nncv:baseline
```

### Peak Performance

```bash
cp checkpoints/Thor_Seg_PP/best_model_miou.pt PeakPerformance/model.pt
docker build -t nncv:peak -f PeakPerformance/Dockerfile PeakPerformance/
docker save -o submission_peak.tar nncv:peak
```

### OOD (one image per scoring method)

```bash
cp checkpoints/Thor_Seg_PP/best_model_miou.pt OOD/model.pt

docker build -t nncv:ood-msp     --build-arg OOD_CONFIG=ood_config_msp.json                 -f OOD/Dockerfile OOD/
docker build -t nncv:ood-tempmsp --build-arg OOD_CONFIG=ood_config_temp_msp.json            -f OOD/Dockerfile OOD/
docker build -t nncv:ood-energy  --build-arg OOD_CONFIG=ood_config_energy.json              -f OOD/Dockerfile OOD/
docker build -t nncv:ood-pixel   --build-arg OOD_CONFIG=ood_config_pixel_uncertainty.json   -f OOD/Dockerfile OOD/

for tag in ood-msp ood-tempmsp ood-energy ood-pixel; do
  docker save -o submission_${tag}.tar nncv:${tag}
done
```

The `OOD_CONFIG` build argument selects which `configs/*.json` is copied to
`/app/ood_config.json` inside the container. The Python code reads that file
at construction time.

For local testing and the upload step, see
[READMEs/README-Submission.md](READMEs/README-Submission.md).

## Experiment tracking

`train.py` logs to Weights & Biases. The `--wandb_run_name` (or
`--experiment-id` for the Baseline starter) becomes the W&B run name. Set
`WANDB_API_KEY` in `.env` before submitting a SLURM job; the script logs in
automatically inside the container. `.env` is gitignored.

## Hardware

Training was run on the TU/e SLURM cluster, partitions `gpu_a100` (A100 40GB)
and `gpu_h100` (H100 80GB), with `--cpus-per-task=18` and `--gpus=1`.
SegFormer-B2 fits comfortably on a single A100 at batch size 8 with bfloat16
autocast.
