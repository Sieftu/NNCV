"""
Fit a single temperature scalar T on the Cityscapes val set by minimising NLL.

A model trained without explicit calibration is usually mildly overconfident,
so the optimal T is typically slightly above 1. The result is written to
temperature.json and is used by score_msp / score_energy in temp-scaled mode.

Usage:
    python fit_temperature.py --checkpoint model.pt --data_root ./data/cityscapes
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Normalize

from model import Backbone


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_ID_TO_TRAINID = torch.full((256,), 255, dtype=torch.long)
for _c in Cityscapes.classes:
    if 0 <= _c.id < 256:
        _ID_TO_TRAINID[_c.id] = _c.train_id if _c.train_id >= 0 else 255


def apply_trainid(mask):
    return _ID_TO_TRAINID[mask.long()]


class TemperatureScaler(nn.Module):
    def __init__(self, init=1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init)

    def forward(self, logits):
        return logits / self.temperature


def build_preprocess():
    return Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_model(checkpoint, device):
    model = Backbone(in_channels=3, n_classes=19)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model


def nll_at_T(logits, labels, T):
    return F.cross_entropy(logits / T, labels).item()


def main():
    parser = argparse.ArgumentParser(description="Fit temperature scaling on Cityscapes val.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--pixels", type=int, default=100_000,
                        help="Total pixel budget for calibration (default 100,000).")
    parser.add_argument("--init_temp", type=float, default=1.5)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Pixel cap  : {args.pixels:,}")

    model = load_model(args.checkpoint, device)
    preprocess = build_preprocess()

    val_ds = Cityscapes(args.data_root, split="val", mode="fine", target_type="semantic")
    n_images = len(val_ds)
    print(f"Val images : {n_images}")

    pixels_per_image = max(1, args.pixels // n_images)
    print(f"Pixels/img : ~{pixels_per_image:,}")

    try:
        from tqdm import tqdm
        indices = tqdm(range(n_images), desc="Collecting logits", unit="img")
    except ImportError:
        indices = range(n_images)

    logits_list = []
    labels_list = []
    rng = np.random.default_rng(seed=42)

    with torch.no_grad():
        for idx in indices:
            img_pil, mask_pil = val_ds[idx]

            img_t = preprocess(img_pil).unsqueeze(0).to(device)
            mask_t = ToImage()(mask_pil)
            label_t = apply_trainid(mask_t.squeeze(0))

            logits = model(img_t).squeeze(0)

            logits_flat = logits.permute(1, 2, 0).reshape(-1, 19)
            labels_flat = label_t.reshape(-1)

            valid_mask = labels_flat != 255
            logits_valid = logits_flat[valid_mask].cpu()
            labels_valid = labels_flat[valid_mask].cpu()

            n_valid = logits_valid.shape[0]
            if n_valid == 0:
                continue

            if n_valid > pixels_per_image:
                chosen = torch.from_numpy(rng.choice(n_valid, size=pixels_per_image, replace=False))
                logits_valid = logits_valid[chosen]
                labels_valid = labels_valid[chosen]

            logits_list.append(logits_valid)
            labels_list.append(labels_valid)

    logits_all = torch.cat(logits_list, dim=0).to(device)
    labels_all = torch.cat(labels_list, dim=0).to(device)
    print(f"\nPixel sample collected : {logits_all.shape[0]:,}")

    nll_before = nll_at_T(logits_all, labels_all, T=1.0)
    print(f"NLL before scaling (T=1.0) : {nll_before:.6f}")

    scaler = TemperatureScaler(init=args.init_temp).to(device)
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits_all), labels_all)
        loss.backward()
        return loss

    optimizer.step(closure)
    optimal_T = scaler.temperature.item()

    nll_after = nll_at_T(logits_all, labels_all, T=optimal_T)
    print(f"NLL after  scaling (T={optimal_T:.4f}) : {nll_after:.6f}")
    print(f"\nOptimal temperature : {optimal_T:.4f}")

    if not (1.0 <= optimal_T <= 4.0):
        print(f"WARNING: T={optimal_T:.4f} outside expected range [1.0, 4.0].")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "temperature": optimal_T,
        "nll_before": nll_before,
        "nll_after": nll_after,
        "n_pixels": int(logits_all.shape[0]),
    }
    json_path = output_dir / "temperature.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Saved -> {json_path}")


if __name__ == "__main__":
    main()
