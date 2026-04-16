import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import model_segformer

# Method-specific safe defaults used when no config file is present
_DEFAULT_THRESHOLDS = {
    "msp": -0.85,     # negative of mean max softmax probability; tune on validation set
    "temp_msp": -0.85,
    "energy": -8.0,   # negative energy score; tune on validation set
}

_CONFIG_PATH = "/app/ood_config.json"


def _load_config():
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    return {}


class Model(nn.Module):
    """
    SegFormer segmentation model with OOD detection.

    Returns a tuple (seg_logits, include_decision) where include_decision is
    True (in-distribution) or False (OOD).
    """

    def __init__(
        self,
        in_channels=3,
        n_classes=19,
        ood_method="msp",
        threshold=None,
        temperature=1.0,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            n_classes (int): Number of output classes.
            ood_method (str): One of 'msp', 'temp_msp', 'energy'.
            threshold (float | None): OOD score threshold. Loaded from
                /app/ood_config.json when None; falls back to a safe default.
            temperature (float): Temperature for temp_msp / energy scoring.
                Also loaded from config when not explicitly provided.
        """
        super().__init__()

        if ood_method not in ("msp", "temp_msp", "energy"):
            raise ValueError(f"Unknown ood_method '{ood_method}'. Choose from: msp, temp_msp, energy")

        self.segmentation_model = model_segformer.Model(in_channels=in_channels, n_classes=n_classes)
        self.ood_method = ood_method

        # Load baked-in config (may be empty dict if file absent)
        cfg = _load_config()

        self.temperature = temperature if temperature != 1.0 else float(cfg.get("temperature", temperature))
        if threshold is not None:
            self.threshold = float(threshold)
        else:
            self.threshold = float(
                cfg.get("threshold", _DEFAULT_THRESHOLDS[ood_method])
            )

    def _ood_score(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute a scalar OOD score for the batch (lower = more in-distribution)."""
        if self.ood_method == "msp":
            # score = -(mean max softmax probability over all pixels)
            probs = F.softmax(logits, dim=1)
            score = -probs.max(dim=1).values.mean()

        elif self.ood_method == "temp_msp":
            probs = F.softmax(logits / self.temperature, dim=1)
            score = -probs.max(dim=1).values.mean()

        elif self.ood_method == "energy":
            score = -(self.temperature * torch.logsumexp(logits / self.temperature, dim=1)).mean()

        return score

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            tuple:
                seg_logits (torch.Tensor): Logits (B, n_classes, H, W).
                include_decision (bool): True if in-distribution, False if OOD.
        """
        seg_logits = self.segmentation_model(x)
        score = self._ood_score(seg_logits)
        include_decision = bool(score.item() < self.threshold)
        return seg_logits, include_decision


if __name__ == "__main__":
    model = Model(in_channels=3, n_classes=19, ood_method="msp")
    x = torch.randn(2, 3, 512, 1024)
    seg_logits, include_decision = model(x)
    n_params = sum(p.numel() for p in model.parameters())

    # Recompute score for display
    with torch.no_grad():
        probs = F.softmax(seg_logits, dim=1)
        score = -probs.max(dim=1).values.mean().item()

    print(f"Segmentation output shape: {seg_logits.shape}")
    print(f"Include decision (in-distribution): {include_decision}")
    print(f"OOD score (msp): {score:.4f}  |  threshold: {model.threshold}")
    print(f"Parameter count: {n_params:,}")
