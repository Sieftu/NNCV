"""
generate_qualitative_figure.py

Produces four PNG images for the qualitative comparison figure referenced
in main.tex (\\cref{fig:qualitative}). LaTeX assembles the 1x4 panel; this
script only emits the panel contents.

Outputs (written to --output-dir):
    input.png            : original RGB leftImg8bit
    ground_truth.png     : GT, Cityscapes palette (255 -> black)
    unet_pred.png        : U-Net baseline prediction, Cityscapes palette
    segformer_pred.png   : SegFormer-B2 prediction, Cityscapes palette

Example
-------
$ python generate_qualitative_figure.py \
      --image  data/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_001016_leftImg8bit.png \
      --gt     data/cityscapes/gtFine/val/frankfurt/frankfurt_000000_001016_gtFine_labelIds.png \
      --unet-ckpt       best_model_miou_unet.pt \
      --segformer-ckpt  best_model_miou_segformer.pt \
      --segformer-config segformer_config \
      --output-dir figures \
      --unet-preprocess starter

Notes
-----
- --unet-preprocess controls preprocessing for the U-Net forward pass:
    starter : Resize 256x256, Normalize(mean=0.5, std=0.5)  [matches 0.414 baseline]
    recipe  : native resolution, Normalize(ImageNet stats)  [matches recipe-trained U-Net]
  Pick the one whose checkpoint matches; mismatching distributions yields
  meaningless predictions.
- SegFormer always runs at native resolution with ImageNet normalisation.
- Logits are upsampled to native resolution before argmax for SegFormer
  (HuggingFace SegFormer returns 1/4-scale logits by default).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import (
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToDtype,
    ToImage,
)
# ---------------------------------------------------------------------------
# Optional / project-local imports
#
# The U-Net definition lives in `model_unet.py` per PROJECT_KNOWLEDGE.md §11,
# but during local iteration `model.py` may transiently be the U-Net build
# (build-time switch). Try the canonical name first, then fall back to
# `model.Model`. If neither resolves, fail loudly with a useful message
# rather than a bare ImportError mid-run.
# ---------------------------------------------------------------------------
try:
    from model_unet import Model as UNet  # canonical: vanilla U-Net, class Model
except ImportError:
    try:
        from model import Model as UNet  # fallback if model.py currently holds the U-Net
    except ImportError as e:
        raise ImportError(
            "Could not import the U-Net Model class. Expected either "
            "`model_unet.py` (preferred, see PROJECT_KNOWLEDGE.md §11) or "
            "`model.py` to define a class named `Model`. "
            "Run this script from the directory that contains it."
        ) from e

try:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
except ImportError as e:
    raise ImportError(
        "huggingface `transformers` is required for the SegFormer branch. "
        "Install it in the same environment used for training: "
        "`pip install transformers`."
    ) from e

# ---------------------------------------------------------------------------
# Cityscapes id <-> trainId <-> colour, replicated from train.py
# ---------------------------------------------------------------------------
ID_TO_TRAINID: dict[int, int] = {cls.id: cls.train_id for cls in Cityscapes.classes}
TRAINID_TO_COLOR: dict[int, tuple[int, int, int]] = {
    cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255
}
TRAINID_TO_COLOR[255] = (0, 0, 0)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Colour mapping
# ---------------------------------------------------------------------------
def labelid_to_trainid(arr: np.ndarray) -> np.ndarray:
    """Map raw Cityscapes labelIds (0..33) to trainIds (0..18, 255)."""
    out = np.full_like(arr, 255)
    for label_id, train_id in ID_TO_TRAINID.items():
        out[arr == label_id] = train_id
    return out


def colorize(trainid_map: np.ndarray) -> Image.Image:
    """trainid_map: (H, W) int array with values in {0..18, 255}."""
    h, w = trainid_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for tid, color in TRAINID_TO_COLOR.items():
        rgb[trainid_map == tid] = color
    return Image.fromarray(rgb)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_unet(ckpt_path: Path, device: str) -> nn.Module:
    model = UNet(in_channels=3, n_classes=19)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device)


def load_segformer(ckpt_path: Path, config_dir: Path, device: str) -> nn.Module:
    config = SegformerConfig.from_pretrained(str(config_dir))
    model = SegformerForSemanticSegmentation(config)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _build_transform(resize_to: tuple[int, int] | None,
                     mean: tuple[float, ...],
                     std: tuple[float, ...]) -> Compose:
    steps: list = [ToImage()]
    if resize_to is not None:
        steps.append(Resize(size=resize_to, interpolation=InterpolationMode.BILINEAR))
    steps += [
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=mean, std=std),
    ]
    return Compose(steps)


def preprocess_unet(img: Image.Image, mode: str) -> torch.Tensor:
    if mode == "starter":
        tf = _build_transform((256, 256), (0.5,), (0.5,))
    elif mode == "recipe":
        tf = _build_transform(None, IMAGENET_MEAN, IMAGENET_STD)
    else:
        raise ValueError(f"unknown --unet-preprocess: {mode!r}")
    return tf(img).unsqueeze(0)


def preprocess_segformer(img: Image.Image) -> torch.Tensor:
    tf = _build_transform(None, IMAGENET_MEAN, IMAGENET_STD)
    return tf(img).unsqueeze(0)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_unet(model: nn.Module, img: Image.Image, mode: str, device: str) -> np.ndarray:
    w, h = img.size
    x = preprocess_unet(img, mode).to(device)
    logits = model(x)  # (1, 19, h', w')
    pred = logits.argmax(dim=1, keepdim=True).float()
    pred = nn.functional.interpolate(pred, size=(h, w), mode="nearest")
    return pred.long().squeeze().cpu().numpy()


@torch.no_grad()
def predict_segformer(model: nn.Module, img: Image.Image, device: str) -> np.ndarray:
    w, h = img.size
    x = preprocess_segformer(img).to(device)
    out = model(pixel_values=x)
    logits = out.logits  # (1, 19, h/4, w/4)
    logits = nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False
    )
    return logits.argmax(dim=1).squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True,
                        help="Path to *_leftImg8bit.png")
    parser.add_argument("--gt", type=Path, required=True,
                        help="Path to *_gtFine_labelIds.png")
    parser.add_argument("--unet-ckpt", type=Path, default=Path("best_model_miou_unet.pt"))
    parser.add_argument("--segformer-ckpt", type=Path, default=Path("best_model_miou_segformer.pt"))
    parser.add_argument("--segformer-config", type=Path, default=Path("segformer_config"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--unet-preprocess", choices=("starter", "recipe"),
                        default="starter",
                        help="Match this to the U-Net checkpoint's training recipe.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Fail fast on missing paths rather than mid-inference.
    for label, path in [
        ("--image", args.image),
        ("--gt", args.gt),
        ("--unet-ckpt", args.unet_ckpt),
        ("--segformer-ckpt", args.segformer_ckpt),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"{label}: file not found at {path}")
    if not args.segformer_config.is_dir():
        raise FileNotFoundError(
            f"--segformer-config: directory not found at {args.segformer_config}. "
            "This must point to the local copy of the HuggingFace SegFormer "
            "config (segformer_config/config.json), per PROJECT_KNOWLEDGE.md §11."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Input image
    img = Image.open(args.image).convert("RGB")
    img.save(args.output_dir / "input.png")

    # 2. Ground truth
    gt_label_ids = np.array(Image.open(args.gt))
    gt_train_ids = labelid_to_trainid(gt_label_ids)
    colorize(gt_train_ids).save(args.output_dir / "ground_truth.png")

    # 3. U-Net prediction
    unet = load_unet(args.unet_ckpt, args.device)
    unet_pred = predict_unet(unet, img, args.unet_preprocess, args.device)
    colorize(unet_pred).save(args.output_dir / "unet_pred.png")
    del unet
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # 4. SegFormer prediction
    segformer = load_segformer(args.segformer_ckpt, args.segformer_config, args.device)
    seg_pred = predict_segformer(segformer, img, args.device)
    colorize(seg_pred).save(args.output_dir / "segformer_pred.png")

    print(f"Wrote 4 images to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
