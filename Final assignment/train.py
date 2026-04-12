"""
This script implements a training loop for the model. It is designed to be flexible,
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console.
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block.
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision import tv_tensors
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ColorJitter,
)

from model import Model


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def compute_confusion_matrix(pred, gt, num_classes=19, ignore_index=255):
    """pred, gt: [N] or [B,H,W] long tensors on same device. Returns [C,C] int64."""
    mask = gt != ignore_index
    pred = pred[mask]
    gt = gt[mask]
    idx = num_classes * gt + pred
    return torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics_from_confusion(conf):
    """conf: [C,C]. Returns (per_class_iou: [C] float tensor with NaN for absent classes, miou: float, pixel_acc: float)."""
    conf = conf.float()
    diag = conf.diag()
    row = conf.sum(dim=1)
    col = conf.sum(dim=0)
    union = row + col - diag
    iou = torch.where(union > 0, diag / union, torch.full_like(diag, float('nan')))
    miou = torch.nanmean(iou).item()
    pixel_acc = (diag.sum() / conf.sum().clamp(min=1)).item()
    return iou, miou, pixel_acc


CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',
    'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle',
]


class AugDataset(torch.utils.data.Dataset):
    """Wraps a Cityscapes dataset to apply joint image+mask augmentation per sample."""
    def __init__(self, base, joint_transform):
        self.base = base
        self.joint_transform = joint_transform
        self.images  = base.images
        self.targets = base.targets

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, lbl = self.base[idx]
        img_tv = tv_tensors.Image(img)
        lbl_tv = tv_tensors.Mask(lbl)
        img_tv, lbl_tv = self.joint_transform(img_tv, lbl_tv)
        return img_tv.as_subclass(torch.Tensor), lbl_tv.as_subclass(torch.Tensor)


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    parser.add_argument("--augmentation", type=str, choices=["none", "aug_only"], default="none",
                        help="none=baseline unchanged; aug_only=RandomResizedCrop+HFlip+ColorJitter at 512x512")
    parser.add_argument("--use_amp", action="store_true",
                        help="Wrap training forward+loss in torch.autocast(cuda, bfloat16)")
    parser.add_argument("--use_compile", action="store_true",
                        help="Apply torch.compile(model, mode='default') after instantiation")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Run val pass every N epochs; always runs on the final epoch")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Override epochs=2 and exit cleanly after; use to validate the pipeline")

    return parser


def main(args):
    # ---- smoke test: cap epochs ----
    if args.smoke_test:
        args.epochs = 2

    # ---- auto-derive experiment_id for aug_only ----
    if args.augmentation == "aug_only" and args.experiment_id == "unet-training":
        args.experiment_id = "aug-only"

    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=args.experiment_id,
        config=vars(args),
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Transforms
    # -------------------------------------------------------------------------
    if args.augmentation == "aug_only":
        # Train base: convert to tensor only -- crop/flip/colour/norm applied jointly
        img_transform = Compose([
            ToImage(),
            ToDtype(torch.float32, scale=True),
        ])
        target_transform = Compose([
            ToImage(),
            ToDtype(torch.int64),
        ])
        # Joint per-sample augmentation (image gets bilinear crop; mask gets nearest)
        joint_aug = Compose([
            RandomResizedCrop(
                size=(512, 512),
                scale=(0.5, 2.0),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        # Baseline transforms (unchanged)
        img_transform = Compose([
            ToImage(),
            Resize((256, 256)),
            ToDtype(torch.float32, scale=True),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        target_transform = Compose([
            ToImage(),
            Resize((256, 256), interpolation=InterpolationMode.NEAREST),
            ToDtype(torch.int64),
        ])
        joint_aug = None

    # Val transform is always the baseline 256x256 (never augmented)
    val_img_transform = Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_target_transform = Compose([
        ToImage(),
        Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.int64),
    ])

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------
    _train_base = Cityscapes(
        args.data_dir,
        split="train",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=target_transform,
    )
    train_dataset = AugDataset(_train_base, joint_aug) if joint_aug is not None else _train_base

    valid_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=val_img_transform,
        target_transform=val_target_transform,
    )

    # -------------------------------------------------------------------------
    # Dataloader kwargs (always applied)
    # -------------------------------------------------------------------------
    _nw = args.num_workers
    _dl_kwargs = dict(
        num_workers=_nw,
        persistent_workers=_nw > 0,
        pin_memory=True,
        prefetch_factor=4 if _nw > 0 else None,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **_dl_kwargs,
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = Model(
        in_channels=3,
        n_classes=19,
    ).to(device)

    # -------------------------------------------------------------------------
    # OOM check (aug_only: 512x512 crops may not fit at the requested batch_size).
    # Tests with a full forward+backward to account for activation memory during
    # training (inference alone uses ~3-4x less memory).
    # Bisects batch_size down: requested -> /2 -> /4 -> /8 (floor at 8).
    # -------------------------------------------------------------------------
    train_batch_size = args.batch_size
    if args.augmentation == "aug_only" and torch.cuda.is_available():
        _crit_oom = nn.CrossEntropyLoss(ignore_index=255)
        test_bs = train_batch_size
        while True:
            try:
                torch.cuda.empty_cache()
                model.train()
                _img = torch.randn(test_bs, 3, 512, 512, device=device)
                _lbl = torch.randint(0, 19, (test_bs, 512, 512), device=device)
                _loss = _crit_oom(model(_img), _lbl)
                _loss.backward()
                model.zero_grad(set_to_none=True)
                del _img, _lbl, _loss
                torch.cuda.empty_cache()
                train_batch_size = test_bs
                if test_bs < args.batch_size:
                    print(
                        f"WARNING: OOM at batch_size={args.batch_size} with 512x512 "
                        f"training, falling back to batch_size={train_batch_size}. "
                        f"LR unchanged."
                    )
                break
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                model.zero_grad(set_to_none=True)
                if test_bs <= 8:
                    raise RuntimeError(
                        f"OOM even at batch_size=8 with 512x512 training on this GPU."
                    ) from e
                test_bs //= 2

    # -------------------------------------------------------------------------
    # Optional torch.compile
    # -------------------------------------------------------------------------
    if args.use_compile:
        try:
            model = torch.compile(model, mode="default")
        except Exception as e:
            print(f"WARNING: torch.compile failed ({e}), proceeding without compile.")

    # -------------------------------------------------------------------------
    # Training dataloader (uses OOM-adjusted batch size)
    # -------------------------------------------------------------------------
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        **_dl_kwargs,
    )

    # -------------------------------------------------------------------------
    # Loss + optimizer
    # -------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # -------------------------------------------------------------------------
    # Eval transforms: image at 256x256 (model input); label at native resolution
    # -------------------------------------------------------------------------
    eval_img_transform = Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    eval_target_transform = Compose([
        ToImage(),
        ToDtype(torch.int64),  # no resize -- keep native 1024x2048
    ])

    val_eval_dataset = Cityscapes(
        args.data_dir, split="val", mode="fine", target_type="semantic",
        transform=eval_img_transform, target_transform=eval_target_transform,
    )
    val_eval_dataloader = DataLoader(
        val_eval_dataset, batch_size=1, shuffle=False, num_workers=_nw,
    )

    train_eval_dataset = Cityscapes(
        args.data_dir, split="train", mode="fine", target_type="semantic",
        transform=eval_img_transform, target_transform=eval_target_transform,
    )
    _sorted_pairs = sorted(
        zip(train_eval_dataset.images, train_eval_dataset.targets),
        key=lambda p: os.path.basename(p[0]),
    )
    train_eval_dataset.images  = [p[0] for p in _sorted_pairs]
    train_eval_dataset.targets = [p[1] for p in _sorted_pairs]
    train_eval_dataloader = DataLoader(
        Subset(train_eval_dataset, range(min(500, len(train_eval_dataset)))),
        batch_size=1, shuffle=False, num_workers=_nw,
    )

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    best_valid_loss = float('inf')
    current_best_model_path = None
    best_miou = 0.0
    current_best_miou_path = None
    valid_loss = float('inf')  # initialised so final-save filename is always defined

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # ---- Train ----
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # CPU, in-place
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)

            optimizer.zero_grad()
            if args.use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)

        # ---- Validation (gated by eval_every; always runs on the final epoch) ----
        run_val = ((epoch + 1) % args.eval_every == 0) or (epoch + 1 == args.epochs)
        if run_val:
            model.eval()
            with torch.no_grad():
                losses = []
                for i, (images, labels) in enumerate(valid_dataloader):

                    labels = convert_to_train_id(labels)
                    images, labels = images.to(device), labels.to(device)
                    labels = labels.long().squeeze(1)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    losses.append(loss.item())

                    if i == 0:
                        predictions = outputs.softmax(1).argmax(1)

                        predictions = predictions.unsqueeze(1)
                        labels = labels.unsqueeze(1)

                        predictions = convert_train_id_to_color(predictions)
                        labels = convert_train_id_to_color(labels)

                        predictions_img = make_grid(predictions.cpu(), nrow=8)
                        labels_img = make_grid(labels.cpu(), nrow=8)

                        predictions_img = predictions_img.permute(1, 2, 0).numpy()
                        labels_img = labels_img.permute(1, 2, 0).numpy()

                        wandb.log({
                            "predictions": [wandb.Image(predictions_img)],
                            "labels": [wandb.Image(labels_img)],
                        }, step=(epoch + 1) * len(train_dataloader) - 1)

                valid_loss = sum(losses) / len(losses)

                # Full-res mIoU evaluation on the entire val split
                conf_val = torch.zeros(19, 19, dtype=torch.int64)
                for images_e, labels_e in val_eval_dataloader:
                    labels_e = convert_to_train_id(labels_e)
                    images_e = images_e.to(device)
                    labels_e_dev = labels_e.to(device).squeeze(1)
                    outputs_e = model(images_e)
                    h_nat, w_nat = labels_e_dev.shape[-2], labels_e_dev.shape[-1]
                    outputs_e = F.interpolate(
                        outputs_e, size=(h_nat, w_nat), mode='bilinear', align_corners=False,
                    )
                    preds_e = outputs_e.argmax(dim=1)
                    conf_val += compute_confusion_matrix(preds_e.cpu(), labels_e.squeeze(1))

                iou, miou, pixel_acc = metrics_from_confusion(conf_val)
                log_dict = {"val/loss": valid_loss, "val/mIoU": miou, "val/pixel_acc": pixel_acc}
                for ci, name in enumerate(CITYSCAPES_CLASSES):
                    log_dict[f"val/iou_{name}"] = iou[ci].item() if not torch.isnan(iou[ci]) else float('nan')
                wandb.log(log_dict, step=(epoch + 1) * len(train_dataloader) - 1)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    if current_best_model_path:
                        os.remove(current_best_model_path)
                    current_best_model_path = os.path.join(
                        output_dir,
                        f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
                    )
                    torch.save(model.state_dict(), current_best_model_path)

                # mIoU-based checkpoint
                if miou > best_miou:
                    best_miou = miou
                    if current_best_miou_path:
                        os.remove(current_best_miou_path)
                    current_best_miou_path = os.path.join(
                        output_dir,
                        f"best_model_miou-epoch={epoch:04}-miou={miou:.4f}.pt",
                    )
                    torch.save(model.state_dict(), current_best_miou_path)

        # ---- Train-slice evaluation every 5 epochs ----
        if (epoch + 1) % 5 == 0:
            conf_tr = torch.zeros(19, 19, dtype=torch.int64)
            model.eval()
            with torch.no_grad():
                for images_t, labels_t in train_eval_dataloader:
                    labels_t = convert_to_train_id(labels_t)
                    images_t = images_t.to(device)
                    labels_t_dev = labels_t.to(device).squeeze(1)
                    outputs_t = model(images_t)
                    h_t, w_t = labels_t_dev.shape[-2], labels_t_dev.shape[-1]
                    outputs_t = F.interpolate(
                        outputs_t, size=(h_t, w_t), mode='bilinear', align_corners=False,
                    )
                    preds_t = outputs_t.argmax(dim=1)
                    conf_tr += compute_confusion_matrix(preds_t.cpu(), labels_t.squeeze(1))
            _, train_miou, train_pix_acc = metrics_from_confusion(conf_tr)
            wandb.log({
                "train_eval/mIoU":      train_miou,
                "train_eval/pixel_acc": train_pix_acc,
            }, step=(epoch + 1) * len(train_dataloader) - 1)

        # ---- Smoke test exit after last epoch ----
        if args.smoke_test and (epoch + 1 == args.epochs):
            print(
                f"\nSmoke test complete. "
                f"train_loss={loss.item():.4f}  "
                f"val_loss={valid_loss:.4f}  "
                f"best_mIoU={best_miou:.4f}"
            )
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"smoke_model-epoch={epoch:04}.pt"),
            )
            wandb.finish()
            return

    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
