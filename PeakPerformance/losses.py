import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, C, H, W), targets: (B, H, W) long
        num_classes = logits.shape[1]
        mask = targets != self.ignore_index
        targets_clean = targets.clone()
        targets_clean[~mask] = 0

        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets_clean, num_classes).permute(0, 3, 1, 2).float()
        mask_expanded = mask.unsqueeze(1).float()

        probs = probs * mask_expanded
        one_hot = one_hot * mask_expanded

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        cardinality = probs.sum(dims) + one_hot.sum(dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class DiceCELoss(nn.Module):
    def __init__(self, ignore_index=255, dice_weight=0.4, ce_weight=0.6):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, targets):
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)
