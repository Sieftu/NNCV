"""
Training script for Cityscapes semantic segmentation.

Shared recipe for both U-Net and SegFormer-B2: 160 epochs, 512x1024 random
crops, Dice+CE loss, AdamW with weight decay, linear warmup followed by
polynomial decay, bfloat16 autocast, gradient clipping.

The Model class is loaded from the local model.py, so each experiment folder
plugs in its own architecture.

Usage:
    python train.py --arch segformer_b2
    python train.py --arch unet
    python train.py --arch segformer_b2 --smoke_test
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, Subset

import wandb
from torchvision.datasets import Cityscapes
from torchvision import tv_tensors
from torchvision.transforms.v2 import (
    ColorJitter,
    Compose,
    InterpolationMode,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResize,
    ToImage,
)

from model import Model
from losses import DiceCELoss


# Lookup: raw Cityscapes label id -> train id (255 means ignore)
_ID_TO_TRAINID = torch.full((256,), 255, dtype=torch.long)
for _c in Cityscapes.classes:
    if 0 <= _c.id < 256:
        _ID_TO_TRAINID[_c.id] = _c.train_id if _c.train_id >= 0 else 255

_trainid_to_name = {}
for _c in Cityscapes.classes:
    if 0 <= _c.train_id <= 18:
        _trainid_to_name[_c.train_id] = _c.name
CLASS_NAMES = [_trainid_to_name[i] for i in range(19)]
assert len(CLASS_NAMES) == 19


def apply_trainid(mask):
    return _ID_TO_TRAINID[mask.long()]


_IMAGENET_NORM = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class CityscapesSegDataset(Dataset):
    def __init__(self, root, split, joint_transform=None):
        self.ds = Cityscapes(root, split=split, mode="fine", target_type="semantic")
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img_pil, mask_pil = self.ds[idx]

        img = tv_tensors.Image(ToImage()(img_pil))
        mask = tv_tensors.Mask(ToImage()(mask_pil))

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        img_f = _IMAGENET_NORM(img.float() / 255.0)
        label = apply_trainid(mask.squeeze(0))

        return img_f, label


def make_train_transform(crop_h, crop_w):
    return Compose([
        RandomResize(
            min_size=int(0.5 * crop_h),
            max_size=int(2.0 * crop_h),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        RandomCrop(
            size=(crop_h, crop_w),
            pad_if_needed=True,
            fill={tv_tensors.Image: 0, tv_tensors.Mask: 255},
        ),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    ])


def update_conf(conf, preds, labels):
    n = conf.shape[0]
    valid = labels != 255
    p, l = preds[valid].long(), labels[valid].long()
    conf += torch.bincount(n * l + p, minlength=n * n).reshape(n, n)


def conf_to_metrics(conf):
    inter = conf.diag().float()
    union = (conf.sum(1) + conf.sum(0) - conf.diag()).float().clamp(min=1)
    iou = (inter / union).cpu()
    miou = iou.mean().item()
    pacc = (inter.sum() / conf.sum().float().clamp(min=1)).item()
    return miou, pacc, iou


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    n = len(CLASS_NAMES)
    conf = torch.zeros(n, n, dtype=torch.long, device=device)
    total = 0.0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(imgs)
            loss = criterion(logits, labels)
        total += loss.item()
        update_conf(conf, logits.argmax(dim=1), labels)

    avg_loss = total / len(loader)
    miou, pacc, iou = conf_to_metrics(conf)
    return avg_loss, miou, pacc, iou


def get_args():
    p = argparse.ArgumentParser(description="Cityscapes segmentation training")
    p.add_argument("--arch", required=True, choices=["unet", "segformer_b2"])
    p.add_argument("--loss", default="dice_ce", choices=["ce", "dice_ce"])
    p.add_argument("--lr", type=float, default=None,
                   help="default 6e-5 for segformer_b2, 1e-3 for unet")
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--crop_size", type=int, nargs=2, default=[512, 1024],
                   metavar=("H", "W"))
    p.add_argument("--data_root", type=str, default="./data/cityscapes")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke_test", action="store_true")
    args = p.parse_args()

    if args.lr is None:
        args.lr = 6e-5 if args.arch == "segformer_b2" else 1e-3
    if args.wandb_run_name is None:
        args.wandb_run_name = args.arch
    if args.smoke_test:
        args.epochs = 5

    return args


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_model = Model().to(device)
    n_params = sum(p.numel() for p in raw_model.parameters())

    model = raw_model
    try:
        model = torch.compile(raw_model, mode="default")
        print(f"torch.compile enabled for arch={args.arch}")
    except Exception:
        print(f"torch.compile failed for arch={args.arch}, falling back to eager")
        model = raw_model

    crop_h, crop_w = args.crop_size

    train_ds = CityscapesSegDataset(
        args.data_root, "train", make_train_transform(crop_h, crop_w)
    )
    val_ds = CityscapesSegDataset(args.data_root, "val")

    train_eval_ds = CityscapesSegDataset(args.data_root, "train")
    sorted_train_idxs = sorted(range(len(train_ds.ds)), key=lambda i: train_ds.ds.images[i])
    train_eval_sub = Subset(train_eval_ds, sorted_train_idxs[:500])

    _dl = dict(num_workers=8, persistent_workers=True, pin_memory=True, prefetch_factor=4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **_dl)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **_dl)
    train_eval_loader = DataLoader(train_eval_sub, batch_size=1, shuffle=False, **_dl)

    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss(ignore_index=255)
    else:
        criterion = DiceCELoss(ignore_index=255)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    warmup_iters = 1500
    total_iters = args.epochs * len(train_loader)
    remaining_iters = max(total_iters - warmup_iters, 1)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_iters),
            PolynomialLR(optimizer, total_iters=remaining_iters, power=1.0),
        ],
        milestones=[warmup_iters],
    )

    ckpt_dir = Path(args.checkpoint_dir) / args.wandb_run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=args.wandb_run_name,
        config={**vars(args), "n_params": n_params},
    )

    best_miou, best_epoch, best_iou_per_class = 0.0, 0, None
    global_step = 0
    first_step = True

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            if first_step:
                try:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        logits = model(imgs)
                        loss = criterion(logits, labels)
                except Exception:
                    print(f"torch.compile failed for arch={args.arch}, falling back to eager")
                    model = raw_model
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        logits = model(imgs)
                        loss = criterion(logits, labels)
                first_step = False
            else:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = model(imgs)
                    loss = criterion(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1
            wandb.log(
                {"train/loss": loss.item(), "train/lr": optimizer.param_groups[0]["lr"]},
                step=global_step,
            )

        avg_loss = epoch_loss / len(train_loader)

        if epoch % 5 == 0 or epoch == args.epochs:
            val_loss, val_miou, val_pacc, val_iou = evaluate(
                model, val_loader, criterion, device
            )
            te_loss, te_miou, te_pacc, te_iou = evaluate(
                model, train_eval_loader, criterion, device
            )

            val_log = {
                "val/loss": val_loss,
                "val/mIoU": val_miou,
                "val/pixel_acc": val_pacc,
                **{f"val/iou_{n}": val_iou[i].item() for i, n in enumerate(CLASS_NAMES)},
            }
            te_log = {
                "train_eval/loss": te_loss,
                "train_eval/mIoU": te_miou,
                "train_eval/pixel_acc": te_pacc,
                **{f"train_eval/iou_{n}": te_iou[i].item() for i, n in enumerate(CLASS_NAMES)},
            }
            wandb.log({**val_log, **te_log}, step=global_step)

            if val_miou > best_miou:
                best_miou, best_epoch, best_iou_per_class = val_miou, epoch, val_iou
                torch.save(raw_model.state_dict(), ckpt_dir / "best_model_miou.pt")

            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"loss={avg_loss:.4f} | "
                f"val_loss={val_loss:.4f}  val_mIoU={val_miou:.4f} | "
                f"best={best_miou:.4f} (ep {best_epoch})"
            )
            model.train()
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f}")

    torch.save(raw_model.state_dict(), ckpt_dir / "final_model.pt")

    if best_iou_per_class is not None:
        print(f"\nBest val mIoU: {best_miou:.4f} at epoch {best_epoch}")
        print(f"{'Class':<22} {'IoU':>8}")
        for name, v in zip(CLASS_NAMES, best_iou_per_class):
            print(f"{name:<22} {v.item():>8.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
