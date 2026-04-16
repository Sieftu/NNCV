"""
Fit a single temperature scalar T on the Cityscapes val set by minimising NLL.

The model is likely overconfident (T < 1 during training), so T > 1 is expected.
Saves the optimal T to temperature.json for use in score_msp / score_energy.

Usage:
    python fit_temperature.py --checkpoint /path/to/model.pt --data_root /path/to/cityscapes
    python fit_temperature.py --checkpoint model.pt --data_root ./data/cityscapes --pixels 100000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Normalize

# Allow importing from Final assignment/
sys.path.insert(0, str(Path(__file__).parent / "Final assignment"))
from model_segformer import Model  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Lookup table: raw Cityscapes pixel value → train ID (255 = ignore)
_ID_TO_TRAINID = torch.full((256,), 255, dtype=torch.long)
for _c in Cityscapes.classes:
    if 0 <= _c.id < 256:
        _ID_TO_TRAINID[_c.id] = _c.train_id if _c.train_id >= 0 else 255


def apply_trainid(mask: torch.Tensor) -> torch.Tensor:
    """Map a (H, W) uint8 mask of raw Cityscapes IDs to train IDs 0–18 / 255."""
    return _ID_TO_TRAINID[mask.long()]


# ── Temperature scaler ────────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    def __init__(self, init: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_preprocess():
    return Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_model(checkpoint: str, device: torch.device) -> Model:
    model = Model(in_channels=3, n_classes=19)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model


def nll_before_after(logits: torch.Tensor, labels: torch.Tensor, T: float) -> float:
    """Cross-entropy of logits / T on the given pixel sample."""
    return F.cross_entropy(logits / T, labels).item()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fit temperature scaling T on Cityscapes val set."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to SegFormer-B2 checkpoint (.pt state dict).")
    parser.add_argument("--data_root",  required=True,
                        help="Root directory of the Cityscapes dataset.")
    parser.add_argument("--pixels",     type=int, default=100_000,
                        help="Total pixel budget for calibration (default: 100 000).")
    parser.add_argument("--init_temp",  type=float, default=1.5,
                        help="Initial temperature (default: 1.5).")
    parser.add_argument("--output_dir", type=str,   default=".",
                        help="Directory for temperature.json (default: current dir).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Pixel cap  : {args.pixels:,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model      = load_model(args.checkpoint, device)
    preprocess = build_preprocess()

    # ── Dataset ───────────────────────────────────────────────────────────────
    val_ds   = Cityscapes(args.data_root, split="val", mode="fine",
                          target_type="semantic")
    n_images = len(val_ds)
    print(f"Val images : {n_images}")

    # pixels_per_image: how many to sample from each image so the total stays
    # near args.pixels.  We over-sample slightly and truncate at the end.
    pixels_per_image = max(1, args.pixels // n_images)
    print(f"Pixels/img : ~{pixels_per_image:,}")

    # ── Collect logit / label samples ─────────────────────────────────────────
    try:
        from tqdm import tqdm
        indices = tqdm(range(n_images), desc="Collecting logits", unit="img")
    except ImportError:
        indices = range(n_images)

    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    rng = np.random.default_rng(seed=42)

    with torch.no_grad():
        for idx in indices:
            img_pil, mask_pil = val_ds[idx]

            # ── Preprocess image ──────────────────────────────────────────────
            img_t = preprocess(img_pil).unsqueeze(0).to(device)  # (1, 3, H, W)

            # ── Ground-truth label ────────────────────────────────────────────
            mask_t  = ToImage()(mask_pil)                    # uint8 (1, H, W)
            label_t = apply_trainid(mask_t.squeeze(0))       # (H, W) long

            # ── Forward ───────────────────────────────────────────────────────
            logits = model(img_t)                            # (1, 19, H, W)
            logits = logits.squeeze(0)                       # (19, H, W)

            # ── Flatten and filter ignore pixels ──────────────────────────────
            H, W         = logits.shape[-2:]
            logits_flat  = logits.permute(1, 2, 0).reshape(-1, 19)  # (H*W, 19)
            labels_flat  = label_t.reshape(-1)                       # (H*W,)

            valid_mask   = labels_flat != 255
            logits_valid = logits_flat[valid_mask].cpu()    # keep on CPU to save VRAM
            labels_valid = labels_flat[valid_mask].cpu()

            n_valid = logits_valid.shape[0]
            if n_valid == 0:
                continue

            # ── Random subsample ─────────────────────────────────────────────
            if n_valid > pixels_per_image:
                chosen = rng.choice(n_valid, size=pixels_per_image, replace=False)
                chosen = torch.from_numpy(chosen)
                logits_valid = logits_valid[chosen]
                labels_valid = labels_valid[chosen]

            logits_list.append(logits_valid)
            labels_list.append(labels_valid)

    logits_all = torch.cat(logits_list, dim=0)   # (N, 19)  on CPU
    labels_all = torch.cat(labels_list, dim=0)   # (N,)     on CPU
    print(f"\nPixel sample collected : {logits_all.shape[0]:,}")

    # Move to device for optimisation
    logits_all = logits_all.to(device)
    labels_all = labels_all.to(device)

    # ── NLL before scaling ────────────────────────────────────────────────────
    nll_before = nll_before_after(logits_all, labels_all, T=1.0)
    print(f"NLL before scaling (T=1.0) : {nll_before:.6f}")

    # ── Fit temperature ───────────────────────────────────────────────────────
    scaler    = TemperatureScaler(init=args.init_temp).to(device)
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits_all), labels_all)
        loss.backward()
        return loss

    optimizer.step(closure)

    optimal_T = scaler.temperature.item()

    # ── NLL after scaling ─────────────────────────────────────────────────────
    nll_after = nll_before_after(logits_all, labels_all, T=optimal_T)
    print(f"NLL after  scaling (T={optimal_T:.4f}) : {nll_after:.6f}")
    print(f"\nOptimal temperature : {optimal_T:.4f}")

    if not (1.0 <= optimal_T <= 4.0):
        print(f"WARNING: T={optimal_T:.4f} is outside the expected range [1.0, 4.0]. "
              "Check that the checkpoint is correct and fully trained.")

    # ── Save ──────────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "temperature": optimal_T,
        "nll_before":  nll_before,
        "nll_after":   nll_after,
        "n_pixels":    int(logits_all.shape[0]),
    }

    json_path = output_dir / "temperature.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Saved → {json_path}")


if __name__ == "__main__":
    main()
