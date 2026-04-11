#!/usr/bin/env python3
"""
audit.py - D9 architecture audit + D10 Cityscapes class frequencies
Run from the "Final assignment" directory:
    python audit.py [--data-dir ./data/cityscapes] [--skip-data]
"""
import sys
import os
import json
import csv
import time
import subprocess

# Reconfigure stdout to UTF-8 so special chars survive any codec wrapper.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---- ensure fvcore is installed ---------------------------------------------
try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    print("fvcore not found - installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fvcore", "-q"])
    from fvcore.nn import FlopCountAnalysis

import torch
import numpy as np
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import Compose, ToImage, ToDtype

# ---- local imports (same directory as model.py / train.py) -----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from model import Model
from train import convert_to_train_id

OUTPUT_DIR = os.path.join(_HERE, "diagnostics_out")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CITYSCAPES_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic_light", "traffic_sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

# =============================================================================
# Analytical receptive field
# =============================================================================

def compute_receptive_field():
    """
    Trace RF through the encoder to the bottleneck (after down4).

    Formula per layer:
        RF_new  = RF_old + (kernel - 1) * jump
        jump_new = jump  * stride

    Network path to bottleneck:
        inc   : DoubleConv(3->64)   = 2x Conv3x3 s=1
        down1 : MaxPool2 s=2 + DoubleConv(64->128)
        down2 : MaxPool2 s=2 + DoubleConv(128->256)
        down3 : MaxPool2 s=2 + DoubleConv(256->512)
        down4 : MaxPool2 s=2 + DoubleConv(512->512)  <- bottleneck
    """
    rf, jump = 1, 1
    trace = []

    def layer(name, kernel, stride):
        nonlocal rf, jump
        rf   = rf + (kernel - 1) * jump
        jump = jump * stride
        trace.append({"name": name, "kernel": kernel, "stride": stride,
                       "rf": rf, "jump": jump})

    # inc
    layer("inc.conv1", 3, 1)
    layer("inc.conv2", 3, 1)

    # down1 ... down4
    for i in range(1, 5):
        layer(f"down{i}.maxpool", 2, 2)
        layer(f"down{i}.conv1",   3, 1)
        layer(f"down{i}.conv2",   3, 1)

    return rf, trace   # rf = 140


# =============================================================================
# Part A - D9 architecture audit
# =============================================================================

def part_a():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    model = Model().to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    # ---- FLOPs (always on CPU to avoid OOM on small GPUs) -------------------
    flops_shapes = [
        (1, 3,  256,  256),
        (1, 3,  512, 1024),
        (1, 3, 1024, 2048),
    ]
    flops_gflops = {}
    print("  Computing FLOPs (on CPU)...")
    cpu_model = Model().cpu().eval()
    for shape in flops_shapes:
        dummy_cpu = torch.zeros(*shape)
        fa = FlopCountAnalysis(cpu_model, dummy_cpu)
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        gf = fa.total() / 1e9
        key = f"{shape[2]}x{shape[3]}"
        flops_gflops[key] = round(gf, 4)
        print(f"    {key}: {gf:.2f} GFLOPs")
    del cpu_model

    # ---- Latency (CUDA only) ------------------------------------------------
    latency = {}
    lat_shapes = [(1, 3, 512, 1024), (1, 3, 1024, 2048)]
    if torch.cuda.is_available():
        print("  Measuring latency...")
        for shape in lat_shapes:
            key = f"{shape[2]}x{shape[3]}"
            dummy = torch.zeros(*shape, device=device)
            with torch.no_grad():
                for _ in range(5):          # warmup
                    model(dummy)
                torch.cuda.synchronize()
                times_ms = []
                for _ in range(20):         # timed
                    t0 = time.perf_counter()
                    model(dummy)
                    torch.cuda.synchronize()
                    times_ms.append((time.perf_counter() - t0) * 1e3)
            mean_ms = float(np.mean(times_ms))
            std_ms  = float(np.std(times_ms, ddof=1))
            latency[key] = {"mean_ms": round(mean_ms, 3), "std_ms": round(std_ms, 3)}
            print(f"    {key}: {mean_ms:.2f} +/- {std_ms:.2f} ms")
    else:
        print("  No CUDA - latency skipped")
        for shape in lat_shapes:
            latency[f"{shape[2]}x{shape[3]}"] = {"mean_ms": None, "std_ms": None}

    # ---- Receptive field ----------------------------------------------------
    rf, rf_trace = compute_receptive_field()
    print(f"  Bottleneck receptive field: {rf}x{rf} px")

    # ---- Save JSON ----------------------------------------------------------
    result = {
        "n_params":                      n_params,
        "flops_gflops":                  flops_gflops,
        "latency":                       latency,
        "bottleneck_receptive_field_px": rf,
        "device":                        str(device),
    }
    out_path = os.path.join(OUTPUT_DIR, "flops_params_latency.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved -> {out_path}")

    return result, rf, rf_trace


# =============================================================================
# Part B - D10 class frequencies
# =============================================================================

def part_b(data_dir: str):
    target_tf = Compose([ToImage(), ToDtype(torch.int64)])

    dataset = Cityscapes(
        data_dir,
        split="train",
        mode="fine",
        target_type="semantic",
        transform=None,          # skip image loading - we only need labels
        target_transform=target_tf,
    )
    print(f"  {len(dataset)} training images found")

    counts = torch.zeros(19, dtype=torch.int64)

    for idx, (_, label) in enumerate(dataset):
        if idx % 500 == 0:
            print(f"    {idx}/{len(dataset)}", flush=True)

        # label: (1, H, W) int64 with raw Cityscapes class IDs
        label = convert_to_train_id(label)   # -> train IDs 0-18 or 255

        valid = label[(label >= 0) & (label < 19)]
        if valid.numel() > 0:
            bc = torch.bincount(valid.reshape(-1), minlength=19)
            counts += bc[:19]

    total = counts.sum().item()
    freqs = counts.double() / total
    inv_f = 1.0 / freqs.clamp(min=1e-12)
    med_f = freqs.median().item()
    mfw   = med_f / freqs.clamp(min=1e-12)

    # ---- Save CSV -----------------------------------------------------------
    csv_path = os.path.join(OUTPUT_DIR, "class_frequencies.csv")
    rows = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "pixel_count",
                          "fraction", "inverse_freq", "median_freq_weight"])
        for i in range(19):
            row = [
                i,
                CITYSCAPES_NAMES[i],
                int(counts[i].item()),
                f"{freqs[i].item():.6f}",
                f"{inv_f[i].item():.4f}",
                f"{mfw[i].item():.4f}",
            ]
            writer.writerow(row)
            rows.append(row)

    print(f"  Saved -> {csv_path}")
    return rows


# =============================================================================
# Markdown output
# =============================================================================

def print_markdown(result_a, rf, rf_trace, rows_b):
    sep = "=" * 72

    print(f"\n{sep}")
    print("# D9 -- Architecture Audit")
    print(sep)

    print(f"\n**Parameters:** {result_a['n_params']:,}")
    print(f"**Device:**     {result_a['device']}")

    print("\n## FLOPs\n")
    print("| Input (HxW)  | GFLOPs |")
    print("|:-------------|-------:|")
    for k, v in result_a["flops_gflops"].items():
        print(f"| {k:<12} | {v:>6.2f} |")

    print("\n## Latency (CUDA, batch=1)\n")
    print("| Input (HxW)  | Mean ms | Std ms |")
    print("|:-------------|--------:|-------:|")
    for k, v in result_a["latency"].items():
        if v["mean_ms"] is None:
            print(f"| {k:<12} | {'N/A':>7} | {'N/A':>6} |")
        else:
            print(f"| {k:<12} | {v['mean_ms']:>7.2f} | {v['std_ms']:>6.2f} |")

    print(f"\n## Receptive Field at Bottleneck = **{rf}x{rf} px**\n")
    print("Formula: RF_new = RF_old + (k-1)*jump,  jump_new = jump*stride\n")
    print("| Layer              | k | s |  RF | jump |")
    print("|:-------------------|:-:|:-:|----:|-----:|")
    for t in rf_trace:
        print(f"| {t['name']:<18} | {t['kernel']} | {t['stride']} | {t['rf']:>3} |  {t['jump']:>3} |")

    if rows_b:
        print(f"\n{sep}")
        print("# D10 -- Cityscapes Class Frequencies (train split)")
        print(sep)
        print()
        print("| id | class          | pixel_count | fraction | inv_freq | med_freq_w |")
        print("|---:|:---------------|------------:|---------:|---------:|-----------:|")
        for r in rows_b:
            print(f"| {r[0]:>2} | {r[1]:<14} | {r[2]:>11,} | {r[3]:>8} | {r[4]:>8} | {r[5]:>10} |")
    else:
        print("\n*(Part B skipped - Cityscapes data not available)*")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="D9/D10 audit script")
    p.add_argument("--data-dir",  default="./data/cityscapes",
                   help="Root of the Cityscapes dataset")
    p.add_argument("--skip-data", action="store_true",
                   help="Skip Part B (class frequencies)")
    args = p.parse_args()

    # ---- Part A -------------------------------------------------------------
    print("\n## Part A -- Architecture Audit")
    result_a, rf, rf_trace = part_a()

    # ---- Part B -------------------------------------------------------------
    rows_b = None
    print("\n## Part B -- Class Frequencies")
    if args.skip_data:
        print("  Skipped (--skip-data)")
    else:
        try:
            rows_b = part_b(args.data_dir)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"  Could not load Cityscapes from '{args.data_dir}': {exc}")
            print("  Re-run with --data-dir <path> or --skip-data to skip Part B.")

    # ---- Print markdown summary ---------------------------------------------
    print_markdown(result_a, rf, rf_trace, rows_b)
    print(f"\nAll outputs in: {OUTPUT_DIR}")
