#!/usr/bin/env python3
"""
diagnostics.py -- Baseline diagnostics for trained U-Net on Cityscapes.

Produces D1 (per-class IoU), D2 (confusion matrix), D3 (boundary vs interior),
D5 (qualitative), D6 (resolution sweep), D7 (calibration), D11 (train/val gap).
Logs everything to wandb and writes all files to --out_dir.

Usage:
    python diagnostics.py --data_root /path/to/cityscapes [--no-res-sweep]
"""

import sys
import os
import csv
import bisect
import heapq
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from tqdm import tqdm
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import Compose, ToImage, Resize, ToDtype, Normalize

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from model import Model
from train import convert_to_train_id, convert_train_id_to_color

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic_light", "traffic_sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]
N = 19

HARD_CLASSES = {
    "pole": 5, "traffic_light": 6, "person": 11,
    "rider": 12, "train": 16, "motorcycle": 17, "bicycle": 18,
}

VIZ_H, VIZ_W = 256, 512   # thumbnail size for qualitative panels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def update_conf(conf, pred, gt, ignore=255):
    """Accumulate confusion matrix in-place (exact spec signature)."""
    mask = gt != ignore
    idx = N * gt[mask].long() + pred[mask].long()
    conf += torch.bincount(idx, minlength=N * N).reshape(N, N)


def iou_from_conf(conf):
    """Per-class IoU; NaN where a class has no GT pixels."""
    diag = conf.diagonal().float()
    denom = conf.sum(1).float() + conf.sum(0).float() - diag
    iou = diag / (denom + 1e-9)
    iou[denom == 0] = float("nan")
    return iou


def make_preprocess(h, w):
    return Compose([
        ToImage(),
        Resize((h, w)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def load_gt(lbl_pil):
    """PIL label -> (H, W) int64 CPU tensor with train IDs applied."""
    arr = np.array(lbl_pil, dtype=np.int64)          # (H, W)
    t = torch.from_numpy(arr).unsqueeze(0)            # (1, H, W) int64
    convert_to_train_id(t)                            # in-place
    return t.squeeze(0)                               # (H, W)


def resize_nearest(t2d, h, w):
    """(H, W) int64 -> (h, w) int64 via nearest interpolation."""
    return F.interpolate(
        t2d.float().unsqueeze(0).unsqueeze(0), size=(h, w), mode="nearest"
    ).squeeze().long()


# ---------------------------------------------------------------------------
# Top-K / Bottom-K trackers (heap-based, O(log k) per push)
# ---------------------------------------------------------------------------

class TopK:
    """Keep K items with highest score."""
    def __init__(self, k):
        self.k, self._h, self._c = k, [], 0

    def push(self, score, data):
        e = (score, self._c, data); self._c += 1
        if len(self._h) < self.k:
            heapq.heappush(self._h, e)
        elif score > self._h[0][0]:
            heapq.heapreplace(self._h, e)

    def items(self):   # [(score, ctr, data)] descending
        return sorted(self._h, key=lambda e: -e[0])


class BottomK:
    """Keep K items with lowest score."""
    def __init__(self, k):
        self.k, self._h, self._c = k, [], 0

    def push(self, score, data):
        e = (-score, self._c, data); self._c += 1
        if len(self._h) < self.k:
            heapq.heappush(self._h, e)
        elif score < -self._h[0][0]:   # score < current max-in-heap
            heapq.heapreplace(self._h, e)

    def items(self):   # [(score, ctr, data)] ascending (worst first)
        return sorted([(-e[0], e[1], e[2]) for e in self._h], key=lambda e: e[0])


# ---------------------------------------------------------------------------
# Qualitative rendering
# ---------------------------------------------------------------------------

def render_4panel(inp_viz, gt_sm, pred_sm, title, save_path):
    """
    inp_viz : (3, 256, 256) float in [0,1]
    gt_sm, pred_sm : (VIZ_H, VIZ_W) int64 with train IDs
    """
    # Resize input to VIZ aspect ratio
    inp_disp = F.interpolate(
        inp_viz.unsqueeze(0), size=(VIZ_H, VIZ_W), mode="bilinear", align_corners=False
    ).squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()

    def colorize(t2d):
        c = convert_train_id_to_color(t2d.unsqueeze(0).unsqueeze(0))
        return c.squeeze(0).permute(1, 2, 0).numpy()

    H, W = gt_sm.shape
    err = torch.full((3, H, W), 255, dtype=torch.uint8)
    ign = gt_sm == 255
    err[:, ign] = 128                                  # grey = ignore
    bad = (pred_sm != gt_sm) & ~ign
    err[0, bad] = 255; err[1, bad] = 0; err[2, bad] = 0   # red = error
    err_np = err.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, img, lbl in zip(axes,
                             [inp_disp, colorize(gt_sm), colorize(pred_sm), err_np],
                             ["Input (256x256)", "GT", "Prediction", "Error"]):
        ax.imshow(img)
        ax.set_title(lbl, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=9, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main single-pass val evaluation
# ---------------------------------------------------------------------------

def run_val_pass(val_dataset, model, device, desc="val pass"):
    preprocess = make_preprocess(256, 256)

    conf_global = torch.zeros(N, N, dtype=torch.int64)
    conf_bnd = {k: torch.zeros(N, N, dtype=torch.int64) for k in (1, 3, 9)}
    conf_int = {k: torch.zeros(N, N, dtype=torch.int64) for k in (1, 3, 9)}

    n_bins = 15
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    cal_count     = torch.zeros(n_bins)
    cal_conf_sum  = torch.zeros(n_bins)
    cal_corr_sum  = torch.zeros(n_bins)

    best5  = TopK(5)
    worst5 = BottomK(5)
    all_mious_sorted = []
    med_cand = {"miou": None, "data": None}
    hard = {cn: {"iou": float("inf"), "data": None} for cn in HARD_CLASSES}

    for idx in tqdm(range(len(val_dataset)), desc=desc):
        img_pil, lbl_pil = val_dataset[idx]

        # ---- preprocess image ----
        x  = preprocess(img_pil).unsqueeze(0).to(device)   # (1,3,256,256)
        gt = load_gt(lbl_pil)                               # (H,W) int64 CPU

        # ---- forward + upsample ----
        logits    = model(x)
        H_nat, W_nat = gt.shape
        logits_up = F.interpolate(logits, size=(H_nat, W_nat),
                                   mode="bilinear", align_corners=False)
        probs             = logits_up.softmax(1)
        conf_max, pred    = probs.max(1)                    # (1,H,W) on device
        pred     = pred.squeeze(0)                          # (H,W) on device
        conf_max = conf_max.squeeze(0)                      # (H,W) on device
        pred_cpu = pred.cpu()

        # ---- D1/D2: global conf ----
        update_conf(conf_global, pred_cpu, gt)

        # ---- D3: boundary ----
        gt_dev = gt.to(device)
        gt_f   = gt_dev.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        pooled = F.max_pool2d(gt_f, kernel_size=3, stride=1, padding=1)
        bnd_base = (pooled != gt_f).squeeze()               # (H,W) bool on device

        for k in (1, 3, 9):
            bnd_dil = F.max_pool2d(
                bnd_base.float().unsqueeze(0).unsqueeze(0),
                kernel_size=2 * k + 1, stride=1, padding=k
            ).squeeze().bool()
            valid = gt_dev != 255

            gt_b = torch.where(bnd_dil & valid, gt_dev, torch.full_like(gt_dev, 255))
            gt_i = torch.where(~bnd_dil & valid, gt_dev, torch.full_like(gt_dev, 255))
            update_conf(conf_bnd[k], pred_cpu, gt_b.cpu())
            update_conf(conf_int[k], pred_cpu, gt_i.cpu())

        # ---- D7: calibration ----
        n_pix = pred_cpu.numel()
        samp  = torch.randperm(n_pix)[:min(50_000, n_pix)]
        c_s   = conf_max.cpu().reshape(-1)[samp]
        p_s   = pred_cpu.reshape(-1)[samp]
        g_s   = gt.reshape(-1)[samp]
        vmask = g_s != 255
        c_s, p_s, g_s = c_s[vmask], p_s[vmask], g_s[vmask]
        if c_s.numel() > 0:
            bins    = torch.bucketize(c_s, bin_edges[1:-1])
            correct = (p_s == g_s).float()
            cal_count.scatter_add_   (0, bins, torch.ones_like(c_s))
            cal_conf_sum.scatter_add_(0, bins, c_s)
            cal_corr_sum.scatter_add_(0, bins, correct)

        # ---- D5: per-image mIoU ----
        conf_img = torch.zeros(N, N, dtype=torch.int64)
        update_conf(conf_img, pred_cpu, gt)
        img_iou  = iou_from_conf(conf_img)
        valid_cl = ~img_iou.isnan()
        img_miou = img_iou[valid_cl].mean().item() if valid_cl.any() else 0.0

        inp_viz  = (x.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)  # (3,256,256)
        gt_sm    = resize_nearest(gt,       VIZ_H, VIZ_W)
        pred_sm  = resize_nearest(pred_cpu, VIZ_H, VIZ_W)
        fname    = os.path.basename(val_dataset.images[idx])
        viz      = (fname, inp_viz, gt_sm, pred_sm)

        best5.push(img_miou, viz)
        worst5.push(img_miou, viz)

        bisect.insort(all_mious_sorted, img_miou)
        n_seen = len(all_mious_sorted)
        med = (all_mious_sorted[n_seen // 2] if n_seen % 2 == 1
               else (all_mious_sorted[n_seen//2-1] + all_mious_sorted[n_seen//2]) / 2)
        if med_cand["miou"] is None or abs(img_miou - med) < abs(med_cand["miou"] - med):
            med_cand = {"miou": img_miou, "data": viz}

        for cn, ci in HARD_CLASSES.items():
            if (gt == ci).sum().item() == 0:
                continue
            tp = conf_img[ci, ci].item()
            fp = conf_img[:, ci].sum().item() - tp
            fn = conf_img[ci, :].sum().item() - tp
            cls_iou = tp / (tp + fp + fn + 1e-9)
            if cls_iou < hard[cn]["iou"]:
                hard[cn] = {"iou": cls_iou, "data": viz}

    return dict(
        conf_global=conf_global,
        conf_bnd=conf_bnd, conf_int=conf_int,
        cal_count=cal_count, cal_conf_sum=cal_conf_sum, cal_corr_sum=cal_corr_sum,
        bin_edges=bin_edges,
        best5=best5, worst5=worst5, med_cand=med_cand, hard=hard,
    )


# ---------------------------------------------------------------------------
# D1 -- per-class IoU
# ---------------------------------------------------------------------------

def compute_d1(conf_global, out_dir):
    iou     = iou_from_conf(conf_global)
    valid   = ~iou.isnan()
    miou    = iou[valid].mean().item() if valid.any() else 0.0
    diag    = conf_global.diagonal().float()
    pix_acc = (diag.sum() / (conf_global.sum().float() + 1e-9)).item()

    rows = sorted(
        [(CITYSCAPES_CLASSES[i], float(iou[i])) for i in range(N) if not iou[i].isnan()],
        key=lambda r: r[1],
    )
    with open(os.path.join(out_dir, "per_class_iou.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["class", "iou"]] + rows)

    with open(os.path.join(out_dir, "per_class_iou.md"), "w") as f:
        f.write("| Class | IoU |\n|:---|---:|\n")
        for cls, v in rows:
            f.write(f"| {cls} | {v:.4f} |\n")
        f.write(f"\n**mIoU = {miou:.4f}**  |  **pixel_acc = {pix_acc:.4f}**\n")

    wandb.log({"val/mIoU": miou, "val/pixel_acc": pix_acc})
    table = wandb.Table(data=[[c, v] for c, v in rows], columns=["Class", "IoU"])
    wandb.log({
        "val/class_iou_table": table,
        "val/class_iou_bar":   wandb.plot.bar(table, "Class", "IoU", title="Per-Class IoU"),
    })
    print(f"  mIoU={miou:.4f}   pixel_acc={pix_acc:.4f}")
    return miou, pix_acc


# ---------------------------------------------------------------------------
# D2 -- confusion matrix
# ---------------------------------------------------------------------------

def compute_d2(conf_global, out_dir):
    row_sums  = conf_global.sum(1, keepdim=True).float()
    conf_norm = (conf_global.float() / (row_sums + 1e-9)).numpy()

    with open(os.path.join(out_dir, "confusion_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + CITYSCAPES_CLASSES)
        for i in range(N):
            w.writerow([CITYSCAPES_CLASSES[i]] + conf_global[i].tolist())

    annot = np.where(conf_norm >= 0.05,
                     np.vectorize(lambda v: f"{v:.2f}")(conf_norm),
                     np.full_like(conf_norm, "", dtype=object))

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(conf_norm, xticklabels=CITYSCAPES_CLASSES, yticklabels=CITYSCAPES_CLASSES,
                annot=annot, fmt="", cmap="viridis", ax=ax,
                annot_kws={"size": 6}, linewidths=0.3)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=12)
    plt.xticks(fontsize=7, rotation=45, ha="right")
    plt.yticks(fontsize=7, rotation=0)
    plt.tight_layout()
    png_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    wandb.log({"val/confusion_matrix": wandb.Image(png_path)})

    raw = conf_global.clone()
    raw.fill_diagonal_(0)
    flat     = raw.reshape(-1)
    top10_fi = flat.argsort(descending=True)[:10]
    with open(os.path.join(out_dir, "top_confusions.md"), "w") as f:
        f.write("## Top-10 Confusions (raw pixel counts)\n\n")
        f.write("| True | Predicted | Count |\n|:---|:---|---:|\n")
        for fi in top10_fi:
            r, c = fi.item() // N, fi.item() % N
            f.write(f"| {CITYSCAPES_CLASSES[r]} | {CITYSCAPES_CLASSES[c]} | {flat[fi].item():,} |\n")

    return png_path


# ---------------------------------------------------------------------------
# D3 -- boundary vs interior
# ---------------------------------------------------------------------------

def compute_d3(conf_bnd, conf_int, out_dir):
    rows = []
    for k in (1, 3, 9):
        cb, ci = conf_bnd[k], conf_int[k]
        acc_b   = (cb.diagonal().sum() / (cb.sum() + 1e-9)).item()
        acc_i   = (ci.diagonal().sum() / (ci.sum() + 1e-9)).item()
        ratio   = acc_b / (acc_i + 1e-9)
        miou_b_ = iou_from_conf(cb); miou_b = miou_b_[~miou_b_.isnan()].mean().item()
        miou_i_ = iou_from_conf(ci); miou_i = miou_i_[~miou_i_.isnan()].mean().item()
        rows.append((k, acc_b, acc_i, ratio, miou_b, miou_i))

    with open(os.path.join(out_dir, "boundary_vs_interior.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["width_px","acc_boundary","acc_interior","ratio",
                    "miou_boundary","miou_interior"])
        for r in rows:
            w.writerow([r[0]] + [f"{v:.4f}" for v in r[1:]])

    tdata = []
    for width_px, acc_b, acc_i, _, miou_b, miou_i in rows:
        tdata += [[width_px, acc_b,  "acc_boundary"],
                  [width_px, acc_i,  "acc_interior"],
                  [width_px, miou_b, "miou_boundary"],
                  [width_px, miou_i, "miou_interior"]]
    line_table = wandb.Table(data=tdata, columns=["width_px","value","metric"])
    wandb.log({
        "val/boundary_table": wandb.Table(
            data=[[r[0]] + [round(v, 4) for v in r[1:]] for r in rows],
            columns=["width_px","acc_boundary","acc_interior","ratio",
                     "miou_boundary","miou_interior"],
        ),
        "val/boundary_line": wandb.plot.line(
            line_table, "width_px", "value",
            stroke="metric", title="Boundary vs Interior",
        ),
    })
    return rows


# ---------------------------------------------------------------------------
# D7 -- calibration
# ---------------------------------------------------------------------------

def compute_d7(cal_count, cal_conf_sum, cal_corr_sum, bin_edges, out_dir):
    n_bins   = len(cal_count)
    N_total  = cal_count.sum().item()
    bin_acc  = (cal_corr_sum / (cal_count + 1e-12)).numpy()
    bin_conf = (cal_conf_sum / (cal_count + 1e-12)).numpy()
    ece      = float(((cal_count / (N_total + 1e-12)) *
                      torch.tensor(bin_acc - bin_conf).abs()).sum())

    centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).numpy()
    widths  = (bin_edges[1:] - bin_edges[:-1]).numpy()

    with open(os.path.join(out_dir, "calibration.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_low","bin_high","count","mean_conf","accuracy"])
        for b in range(n_bins):
            w.writerow([f"{bin_edges[b].item():.4f}", f"{bin_edges[b+1].item():.4f}",
                        int(cal_count[b].item()),
                        f"{bin_conf[b]:.4f}", f"{bin_acc[b]:.4f}"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8),
                                    gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax1.bar(centers, bin_acc, width=widths, align="center",
            alpha=0.75, color="steelblue", label="Accuracy")
    ax1.plot([0, 1], [0, 1], "r--", lw=1.5, label="Perfect calibration")
    ax1.set_ylabel("Accuracy"); ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)
    ax1.set_title(f"Reliability Diagram  (ECE = {ece:.4f})")

    ax2.bar(centers, cal_count.numpy(), width=widths, align="center",
            color="grey", alpha=0.7)
    ax2.set_xlabel("Confidence"); ax2.set_ylabel("Count")
    plt.tight_layout()
    png_path = os.path.join(out_dir, "reliability_diagram.png")
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    wandb.log({"val/ECE": ece, "val/reliability_diagram": wandb.Image(png_path)})
    wandb.run.summary["val/ECE"] = ece
    print(f"  ECE = {ece:.4f}")
    return ece


# ---------------------------------------------------------------------------
# D5 -- qualitative rendering
# ---------------------------------------------------------------------------

def render_d5(best5, worst5, med_cand, hard, out_dir):
    qual_dir = os.path.join(out_dir, "qualitative")
    os.makedirs(qual_dir, exist_ok=True)
    all_wandb_imgs = []

    def save_one(tag, fname, inp, gt_sm, pred_sm):
        stem  = os.path.splitext(fname)[0]
        path  = os.path.join(qual_dir, f"{tag}_{stem}.png")
        render_4panel(inp, gt_sm, pred_sm, title=f"{tag} | {fname}", save_path=path)
        all_wandb_imgs.append(wandb.Image(path, caption=f"{tag}: {fname}"))

    for rank, (score, _, (fname, inp, gt_sm, pred_sm)) in enumerate(worst5.items()):
        save_one(f"worst_{rank+1}_mIoU={score:.3f}", fname, inp, gt_sm, pred_sm)
    for rank, (score, _, (fname, inp, gt_sm, pred_sm)) in enumerate(best5.items()):
        save_one(f"best_{rank+1}_mIoU={score:.3f}", fname, inp, gt_sm, pred_sm)
    if med_cand["data"] is not None:
        fname, inp, gt_sm, pred_sm = med_cand["data"]
        save_one(f"median_mIoU={med_cand['miou']:.3f}", fname, inp, gt_sm, pred_sm)
    for cn, info in hard.items():
        if info["data"] is not None:
            fname, inp, gt_sm, pred_sm = info["data"]
            save_one(f"hard_{cn}_IoU={info['iou']:.3f}", fname, inp, gt_sm, pred_sm)

    if all_wandb_imgs:
        wandb.log({"qualitative/grid": all_wandb_imgs})
    return qual_dir


# ---------------------------------------------------------------------------
# D6 -- resolution sweep
# ---------------------------------------------------------------------------

def run_res_sweep(val_dataset, model, device, out_dir):
    resolutions  = [(256, 256), (256, 512), (384, 768), (512, 1024), (768, 1536), (1024, 2048)]
    sweep_rows   = []
    NATIVE_H, NATIVE_W = 1024, 2048   # Cityscapes native label resolution

    # ---- pipeline debug: one image at each resolution ----
    print("  [D6 debug] pipeline shapes for val image 0:")
    img_pil0, lbl_pil0 = val_dataset[0]
    gt0 = load_gt(lbl_pil0)
    print(f"    GT (native) shape  : {tuple(gt0.shape)}")
    for h, w in resolutions:
        pre0 = make_preprocess(h, w)
        x0   = pre0(img_pil0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits0  = model(x0)
        logits_up0   = F.interpolate(logits0, size=(NATIVE_H, NATIVE_W),
                                      mode="bilinear", align_corners=False)
        pred0        = logits_up0.argmax(1).squeeze(0)
        print(f"    res {h}x{w}: "
              f"input={tuple(x0.shape)}  "
              f"logits={tuple(logits0.shape)}  "
              f"logits_up={tuple(logits_up0.shape)}  "
              f"pred={tuple(pred0.shape)}  "
              f"gt={tuple(gt0.shape)}")
        assert pred0.shape == (NATIVE_H, NATIVE_W), \
            f"pred shape mismatch: {pred0.shape}"
        assert gt0.shape   == (NATIVE_H, NATIVE_W), \
            f"gt shape mismatch: {gt0.shape}"
    print("  [D6 debug] pipeline OK - all shapes verified.")

    for h, w in resolutions:
        pre  = make_preprocess(h, w)
        conf = torch.zeros(N, N, dtype=torch.int64)
        for idx in tqdm(range(len(val_dataset)), desc=f"sweep {h}x{w}"):
            img_pil, lbl_pil = val_dataset[idx]
            x   = pre(img_pil).unsqueeze(0).to(device)
            gt  = load_gt(lbl_pil)          # (1024, 2048) int64 CPU, never resized
            # upsample logits to native resolution -- GT is never touched
            logits_up = F.interpolate(
                model(x), size=(NATIVE_H, NATIVE_W),
                mode="bilinear", align_corners=False,
            )
            pred = logits_up.argmax(1).squeeze(0).cpu()   # (1024, 2048)
            assert pred.shape == (NATIVE_H, NATIVE_W), \
                f"pred shape {pred.shape} at res {h}x{w}"
            assert gt.shape   == (NATIVE_H, NATIVE_W), \
                f"gt shape {gt.shape} at res {h}x{w}"
            update_conf(conf, pred, gt)
        iou  = iou_from_conf(conf)
        miou = iou[~iou.isnan()].mean().item()
        sweep_rows.append((h, w, miou))
        print(f"  {h}x{w}: mIoU = {miou:.4f}")

    with open(os.path.join(out_dir, "resolution_sweep.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["input_h","input_w","miou"]] + list(sweep_rows))

    pixels = [h * w for h, w, _ in sweep_rows]
    mious  = [m      for _, _, m  in sweep_rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(pixels, mious, "o-", linewidth=2)
    ax.set_xlabel("Input pixels (log scale)"); ax.set_ylabel("mIoU")
    ax.set_title("mIoU vs Input Resolution")
    for (h, w, m), px in zip(sweep_rows, pixels):
        ax.annotate(f"{h}x{w}", (px, m), textcoords="offset points",
                    xytext=(5, 3), fontsize=8)
    plt.tight_layout()
    png_path = os.path.join(out_dir, "resolution_sweep.png")
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    tdata = [[h * w, m, f"{h}x{w}"] for h, w, m in sweep_rows]
    table = wandb.Table(data=tdata, columns=["pixels","mIoU","resolution"])
    wandb.log({
        "res_sweep/plot": wandb.plot.line(table, "pixels", "mIoU",
                                           title="Resolution Sweep mIoU"),
        "res_sweep/png":  wandb.Image(png_path),
    })
    return sweep_rows


# ---------------------------------------------------------------------------
# D11 -- train/val gap
# ---------------------------------------------------------------------------

def run_train_eval(train_dataset, model, device, n):
    # Sort by image filename for deterministic first-N selection
    pairs = sorted(zip(train_dataset.images, train_dataset.targets),
                   key=lambda p: os.path.basename(p[0]))
    train_dataset.images  = [p[0] for p in pairs]
    train_dataset.targets = [p[1] for p in pairs]

    pre  = make_preprocess(256, 256)
    conf = torch.zeros(N, N, dtype=torch.int64)
    n_actual = min(n, len(train_dataset))

    for idx in tqdm(range(n_actual), desc=f"train eval (n={n_actual})"):
        img_pil, lbl_pil = train_dataset[idx]
        x  = pre(img_pil).unsqueeze(0).to(device)
        gt = load_gt(lbl_pil)
        logits_up = F.interpolate(model(x), size=gt.shape,
                                   mode="bilinear", align_corners=False)
        _, pred = logits_up.softmax(1).max(1)
        update_conf(conf, pred.squeeze(0).cpu(), gt)

    iou     = iou_from_conf(conf)
    miou    = iou[~iou.isnan()].mean().item()
    diag    = conf.diagonal().float()
    pix_acc = (diag.sum() / (conf.sum().float() + 1e-9)).item()

    wandb.log({"train_eval/mIoU": miou, "train_eval/pixel_acc": pix_acc})
    print(f"  train eval: mIoU={miou:.4f}   pixel_acc={pix_acc:.4f}")
    return miou, pix_acc, n_actual


# ---------------------------------------------------------------------------
# SUMMARY.md
# ---------------------------------------------------------------------------

def write_summary(out_dir, miou, pix_acc, ece, d3_rows, sweep_rows,
                  train_miou, train_pix_acc, n_train, val_n):
    lines = ["# Baseline Diagnostics Summary\n\n"]

    lines += [
        "## D1 -- Per-Class IoU\n\n",
        f"**mIoU = {miou:.4f}**  |  **Pixel Acc = {pix_acc:.4f}**\n\n",
        "See [per_class_iou.md](per_class_iou.md)\n\n",
    ]

    lines += [
        "## D2 -- Confusion Matrix\n\n",
        "![confusion matrix](confusion_matrix.png)\n\n",
        "See [top_confusions.md](top_confusions.md)\n\n",
    ]

    lines += ["## D3 -- Boundary vs Interior\n\n"]
    lines += ["| width_px | acc_boundary | acc_interior | ratio |"
              " miou_boundary | miou_interior |\n"]
    lines += ["|---:|---:|---:|---:|---:|---:|\n"]
    for r in d3_rows:
        lines.append(f"| {r[0]} | {r[1]:.4f} | {r[2]:.4f} | {r[3]:.4f} |"
                     f" {r[4]:.4f} | {r[5]:.4f} |\n")
    lines.append("\n")

    lines += ["## D5 -- Qualitative Examples\n\n",
              "See [qualitative/](qualitative/) folder.\n\n"]

    if sweep_rows:
        lines += ["## D6 -- Resolution Sweep\n\n"]
        lines += ["| input_h | input_w | mIoU |\n|---:|---:|---:|\n"]
        for h, w, m in sweep_rows:
            lines.append(f"| {h} | {w} | {m:.4f} |\n")
        lines += ["\n![resolution sweep](resolution_sweep.png)\n\n"]
    else:
        lines += ["## D6 -- Resolution Sweep\n\nSkipped (--no-res-sweep).\n\n"]

    lines += [
        f"## D7 -- Calibration  (ECE = {ece:.4f})\n\n",
        "![reliability diagram](reliability_diagram.png)\n\n",
        "See [calibration.csv](calibration.csv)\n\n",
    ]

    lines += ["## D11 -- Train / Val Gap\n\n"]
    lines += ["| split | n_images | mIoU | pixel_acc |\n|:---|---:|---:|---:|\n"]
    lines.append(f"| train (eval) | {n_train} | {train_miou:.4f} | {train_pix_acc:.4f} |\n")
    lines.append(f"| val          | {val_n}   | {miou:.4f} | {pix_acc:.4f} |\n")

    path = os.path.join(out_dir, "SUMMARY.md")
    with open(path, "w") as f:
        f.writelines(lines)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Baseline diagnostics for trained U-Net")
    parser.add_argument("--checkpoint",
                        default="best_model-epoch=0055-val_loss=0.28399493731558323.pt",
                        help="Checkpoint filename (resolved under checkpoints/unet-training/) "
                             "or absolute path.")
    parser.add_argument("--data_root",      default="./data/cityscapes")
    parser.add_argument("--out_dir",        default="diagnostics_out")
    parser.add_argument("--wandb_project",  default="5lsm0-cityscapes-segmentation")
    parser.add_argument("--wandb_run_name", default="baseline-diagnostics")
    parser.add_argument("--res_sweep",      default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Run D6 resolution sweep (4 extra val passes).")
    parser.add_argument("--res_sweep_only", action="store_true", default=False,
                        help="Run ONLY the D6 resolution sweep; skip D1/D2/D3/D5/D7/D11.")
    parser.add_argument("--train_eval_n",   type=int, default=500,
                        help="Number of train images for D11 gap estimate.")
    args = parser.parse_args()

    # ---- resolve paths ----
    ckpt = args.checkpoint
    if not os.path.isabs(ckpt):
        candidate = os.path.join(_HERE, "checkpoints", "unet-training", ckpt)
        ckpt = candidate if os.path.isfile(candidate) else os.path.join(_HERE, ckpt)

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(_HERE, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Checkpoint : {ckpt}")
    print(f"Output dir : {out_dir}")

    # ---- wandb ----
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        job_type="diagnostics",
        config=vars(args),
    )

    # ---- model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = Model().to(device)
    state  = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Model loaded on {device}.")

    # ---- datasets (no transforms -- applied manually in loop) ----
    val_dataset = Cityscapes(args.data_root, split="val",
                              mode="fine", target_type="semantic")
    print(f"Val split  : {len(val_dataset)} images")

    miou = pix_acc = ece = 0.0
    d3_rows = []
    sweep_rows = []
    train_miou = train_pix_acc = 0.0
    n_train = 0

    if not args.res_sweep_only:
        # =====================================================================
        # PASS 1: main val pass (D1, D2, D3, D5, D7)
        # =====================================================================
        print("\n=== Pass 1: main val pass ===")
        with torch.no_grad():
            res = run_val_pass(val_dataset, model, device)

        print("\n--- D1: per-class IoU ---")
        miou, pix_acc = compute_d1(res["conf_global"], out_dir)

        print("\n--- D2: confusion matrix ---")
        compute_d2(res["conf_global"], out_dir)

        print("\n--- D3: boundary vs interior ---")
        d3_rows = compute_d3(res["conf_bnd"], res["conf_int"], out_dir)

        print("\n--- D7: calibration ---")
        ece = compute_d7(res["cal_count"], res["cal_conf_sum"], res["cal_corr_sum"],
                         res["bin_edges"], out_dir)

        print("\n--- D5: qualitative rendering ---")
        render_d5(res["best5"], res["worst5"], res["med_cand"], res["hard"], out_dir)

    # =========================================================================
    # PASS 2 (x4): D6 resolution sweep
    # =========================================================================
    if args.res_sweep_only or args.res_sweep:
        print("\n=== Pass 2: D6 resolution sweep ===")
        with torch.no_grad():
            sweep_rows = run_res_sweep(val_dataset, model, device, out_dir)

    if not args.res_sweep_only:
        # =====================================================================
        # PASS 3: D11 train/val gap
        # =====================================================================
        print("\n=== Pass 3: D11 train/val gap ===")
        train_dataset = Cityscapes(args.data_root, split="train",
                                    mode="fine", target_type="semantic")
        print(f"Train split: {len(train_dataset)} images  (using first {args.train_eval_n})")
        with torch.no_grad():
            train_miou, train_pix_acc, n_train = run_train_eval(
                train_dataset, model, device, args.train_eval_n,
            )

        with open(os.path.join(out_dir, "train_val_gap.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "n_images", "miou", "pixel_acc"])
            w.writerow(["train", n_train,             f"{train_miou:.4f}", f"{train_pix_acc:.4f}"])
            w.writerow(["val",   len(val_dataset),    f"{miou:.4f}",       f"{pix_acc:.4f}"])

        # =====================================================================
        # SUMMARY.md
        # =====================================================================
        print("\n--- Writing SUMMARY.md ---")
        summary_path = write_summary(
            out_dir, miou, pix_acc, ece, d3_rows, sweep_rows,
            train_miou, train_pix_acc, n_train, len(val_dataset),
        )
        print(f"  -> {summary_path}")

    # =========================================================================
    # wandb artifact
    # =========================================================================
    artifact = wandb.Artifact(name="baseline-diagnostics", type="diagnostics")
    artifact.add_dir(out_dir)
    wandb.log_artifact(artifact)

    wandb.finish()
    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
