"""
Calibrate OOD thresholds on the Cityscapes val set.

Runs the SegFormer-B2 backbone over every val image, computes four scores
per image (msp, temp_msp, energy, pixel_uncertainty), and writes:
  - ood_thresholds.json  : the 99th-percentile threshold per method
  - ood_scores_val.npz   : raw per-image arrays for follow-up analysis

The four output thresholds populate the four configs/ood_config_*.json files.

Usage:
    python calibrate_ood.py --checkpoint model.pt --data_root ./data/cityscapes
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Normalize

from model import Backbone
from ood_scoring import score_msp, score_energy, score_pixel_uncertainty


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_TEMP = 2.0
DEFAULT_CONF_THR = 0.5
THRESHOLD_PERCENTILE = 99


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


def print_stats(name, scores):
    pcts = np.percentile(scores, [50, 75, 90, 95, 99])
    print(f"\n  {name}")
    print(f"    mean={scores.mean():.6f}  std={scores.std():.6f}  "
          f"min={scores.min():.6f}  max={scores.max():.6f}")
    print(f"    p50={pcts[0]:.6f}  p75={pcts[1]:.6f}  p90={pcts[2]:.6f}  "
          f"p95={pcts[3]:.6f}  p99={pcts[4]:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate OOD thresholds on Cityscapes val.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to SegFormer-B2 checkpoint (.pt state dict).")
    parser.add_argument("--data_root", required=True,
                        help="Root directory of the Cityscapes dataset.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMP,
                        help=f"Temperature for temp-scaled MSP (default {DEFAULT_TEMP}).")
    parser.add_argument("--conf_threshold", type=float, default=DEFAULT_CONF_THR,
                        help=f"Confidence threshold for pixel_uncertainty (default {DEFAULT_CONF_THR}).")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")

    model = load_model(args.checkpoint, device)
    preprocess = build_preprocess()

    val_ds = Cityscapes(args.data_root, split="val", mode="fine", target_type="semantic")
    n_images = len(val_ds)
    print(f"Val images : {n_images}")

    scores_msp = np.empty(n_images, dtype=np.float32)
    scores_tmsp = np.empty(n_images, dtype=np.float32)
    scores_energy = np.empty(n_images, dtype=np.float32)
    scores_pu = np.empty(n_images, dtype=np.float32)
    image_names = []

    print(f"\nScoring {n_images} val images ...")

    try:
        from tqdm import tqdm
        indices = tqdm(range(n_images), unit="img")
    except ImportError:
        indices = range(n_images)

    with torch.no_grad():
        for idx in indices:
            img_pil, _ = val_ds[idx]
            img_t = preprocess(img_pil).unsqueeze(0).to(device)
            logits = model(img_t)

            scores_msp[idx] = score_msp(logits, temperature=1.0)
            scores_tmsp[idx] = score_msp(logits, temperature=args.temperature)
            scores_energy[idx] = score_energy(logits, temperature=1.0)
            scores_pu[idx] = score_pixel_uncertainty(logits, conf_threshold=args.conf_threshold)

            image_names.append(str(val_ds.images[idx]))

    print("\n=== Per-method statistics (all ID val images) ===")
    print_stats("msp (T=1.0)", scores_msp)
    print_stats(f"temp_msp (T={args.temperature})", scores_tmsp)
    print_stats("energy (T=1.0)", scores_energy)
    print_stats(f"pixel_uncertainty (thr={args.conf_threshold})", scores_pu)

    thr_msp = float(np.percentile(scores_msp, THRESHOLD_PERCENTILE))
    thr_tmsp = float(np.percentile(scores_tmsp, THRESHOLD_PERCENTILE))
    thr_energy = float(np.percentile(scores_energy, THRESHOLD_PERCENTILE))
    thr_pu = float(np.percentile(scores_pu, THRESHOLD_PERCENTILE))

    print(f"\n=== Thresholds ({THRESHOLD_PERCENTILE}th percentile) ===")
    print(f"  msp               : {thr_msp:.6f}")
    print(f"  temp_msp (T={args.temperature}) : {thr_tmsp:.6f}")
    print(f"  energy            : {thr_energy:.6f}")
    print(f"  pixel_uncertainty : {thr_pu:.6f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = {
        "msp": {"threshold": thr_msp, "temperature": 1.0},
        "temp_msp": {"threshold": thr_tmsp, "temperature": args.temperature},
        "energy": {"threshold": thr_energy, "temperature": 1.0},
        "pixel_uncertainty": {"threshold": thr_pu, "conf_threshold": args.conf_threshold},
    }

    json_path = output_dir / "ood_thresholds.json"
    with open(json_path, "w") as f:
        json.dump(thresholds, f, indent=4)
    print(f"\nSaved thresholds -> {json_path}")

    npz_path = output_dir / "ood_scores_val.npz"
    np.savez(
        npz_path,
        msp=scores_msp,
        temp_msp=scores_tmsp,
        energy=scores_energy,
        pixel_uncertainty=scores_pu,
        image_names=np.array(image_names),
    )
    print(f"Saved raw scores  -> {npz_path}")


if __name__ == "__main__":
    main()
