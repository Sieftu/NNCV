"""
Calibrate OOD thresholds on the Cityscapes val set (all in-distribution).

Runs SegFormer-B2 over every val image, computes four OOD scores per image,
prints per-method statistics, and writes:
  - ood_thresholds.json   (threshold per method, ready for inference)
  - ood_scores_val.npz    (raw per-image arrays for report analysis)

Usage:
    python calibrate_ood.py --checkpoint /path/to/model.pt --data_root /path/to/cityscapes
    python calibrate_ood.py --checkpoint model.pt --data_root ./data/cityscapes --output_dir ./ood
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Normalize

# Allow importing model_segformer from the "Final assignment" sub-directory
sys.path.insert(0, str(Path(__file__).parent / "Final assignment"))
from model_segformer import Model  # noqa: E402 (path must be set first)

from ood_scoring import score_msp, score_energy, score_pixel_uncertainty  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEFAULT_TEMP        = 2.0   # temperature for temp-scaled MSP
DEFAULT_CONF_THR    = 0.5   # confidence threshold for pixel_uncertainty
THRESHOLD_PERCENTILE = 99   # 99th percentile → ~99 % of ID images included


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_preprocess():
    """ImageNet-normalised float32 tensor; no resize (Cityscapes is 1024×2048)."""
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


def print_stats(name: str, scores: np.ndarray) -> None:
    pcts = np.percentile(scores, [50, 75, 90, 95, 99])
    print(f"\n  {name}")
    print(f"    mean={scores.mean():.6f}  std={scores.std():.6f}  "
          f"min={scores.min():.6f}  max={scores.max():.6f}")
    print(f"    p50={pcts[0]:.6f}  p75={pcts[1]:.6f}  p90={pcts[2]:.6f}  "
          f"p95={pcts[3]:.6f}  p99={pcts[4]:.6f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate OOD thresholds on the Cityscapes val set."
    )
    parser.add_argument("--checkpoint",     required=True,
                        help="Path to SegFormer-B2 checkpoint (.pt state dict).")
    parser.add_argument("--data_root",      required=True,
                        help="Root directory of the Cityscapes dataset.")
    parser.add_argument("--temperature",    type=float, default=DEFAULT_TEMP,
                        help=f"Temperature for temp-scaled MSP (default: {DEFAULT_TEMP}).")
    parser.add_argument("--conf_threshold", type=float, default=DEFAULT_CONF_THR,
                        help=f"Confidence threshold for pixel_uncertainty (default: {DEFAULT_CONF_THR}).")
    parser.add_argument("--output_dir",     type=str,   default=".",
                        help="Directory for output files (default: current dir).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Checkpoint : {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    preprocess = build_preprocess()
    val_ds     = Cityscapes(args.data_root, split="val", mode="fine",
                            target_type="semantic")
    n_images   = len(val_ds)
    print(f"Val images : {n_images}")

    # ── Scoring loop ──────────────────────────────────────────────────────────
    scores_msp   = np.empty(n_images, dtype=np.float32)
    scores_tmsp  = np.empty(n_images, dtype=np.float32)
    scores_energy= np.empty(n_images, dtype=np.float32)
    scores_pu    = np.empty(n_images, dtype=np.float32)
    image_names  = []

    print(f"\nScoring {n_images} val images …")

    # Try to import tqdm for a progress bar; fall back to plain iteration.
    try:
        from tqdm import tqdm
        indices = tqdm(range(n_images), unit="img")
    except ImportError:
        indices = range(n_images)

    with torch.no_grad():
        for idx in indices:
            img_pil, _ = val_ds[idx]

            # Preprocess: float32, ImageNet-normalised, no resize
            img_t = preprocess(img_pil).unsqueeze(0).to(device)  # (1, 3, 1024, 2048)

            logits = model(img_t)  # (1, 19, 1024, 2048)

            scores_msp[idx]    = score_msp(logits, temperature=1.0)
            scores_tmsp[idx]   = score_msp(logits, temperature=args.temperature)
            scores_energy[idx] = score_energy(logits, temperature=1.0)
            scores_pu[idx]     = score_pixel_uncertainty(
                logits, conf_threshold=args.conf_threshold
            )

            image_names.append(str(val_ds.images[idx]))

    # ── Statistics ────────────────────────────────────────────────────────────
    print("\n=== Per-method statistics (all ID val images) ===")
    print_stats("msp (T=1.0)",                      scores_msp)
    print_stats(f"temp_msp (T={args.temperature})", scores_tmsp)
    print_stats("energy (T=1.0)",                   scores_energy)
    print_stats(f"pixel_uncertainty (thr={args.conf_threshold})", scores_pu)

    # ── Thresholds ────────────────────────────────────────────────────────────
    # All methods: 99th percentile of the ID distribution.
    #
    # MSP / Energy return negative values (more negative = more ID).
    #   p99 is the least-confident ID image.  At inference, score > threshold → OOD.
    #
    # pixel_uncertainty returns a positive value (higher = more OOD).
    #   p99 is the highest tolerable uncertain-pixel fraction.
    #   At inference, score > threshold → OOD.
    thr_msp    = float(np.percentile(scores_msp,    THRESHOLD_PERCENTILE))
    thr_tmsp   = float(np.percentile(scores_tmsp,   THRESHOLD_PERCENTILE))
    thr_energy = float(np.percentile(scores_energy, THRESHOLD_PERCENTILE))
    thr_pu     = float(np.percentile(scores_pu,     THRESHOLD_PERCENTILE))

    print(f"\n=== Thresholds ({THRESHOLD_PERCENTILE}th percentile) ===")
    print(f"  msp               : {thr_msp:.6f}")
    print(f"  temp_msp (T={args.temperature}) : {thr_tmsp:.6f}")
    print(f"  energy            : {thr_energy:.6f}")
    print(f"  pixel_uncertainty : {thr_pu:.6f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = {
        "msp": {
            "threshold": thr_msp,
            "temperature": 1.0,
        },
        "temp_msp": {
            "threshold": thr_tmsp,
            "temperature": args.temperature,
        },
        "energy": {
            "threshold": thr_energy,
            "temperature": 1.0,
        },
        "pixel_uncertainty": {
            "threshold": thr_pu,
            "conf_threshold": args.conf_threshold,
        },
    }

    json_path = output_dir / "ood_thresholds.json"
    with open(json_path, "w") as f:
        json.dump(thresholds, f, indent=4)
    print(f"\nSaved thresholds → {json_path}")

    npz_path = output_dir / "ood_scores_val.npz"
    np.savez(
        npz_path,
        msp=scores_msp,
        temp_msp=scores_tmsp,
        energy=scores_energy,
        pixel_uncertainty=scores_pu,
        image_names=np.array(image_names),
    )
    print(f"Saved raw scores  → {npz_path}")


if __name__ == "__main__":
    main()
