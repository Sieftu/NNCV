import json
import os

import torch
import torch.nn as nn

from model_segformer import Model as SegformerModel
from ood_scoring import score_msp, score_energy, score_pixel_uncertainty

_CONFIG_PATH = "/app/ood_config.json"


class Model(nn.Module):
    """
    SegFormer-B2 with OOD detection.

    Reads /app/ood_config.json at construction time to determine the scoring
    method and threshold.  Falls back to safe defaults when the file is absent
    (useful for local testing without a Docker environment).

    Forward returns (seg_logits, include_decision):
        seg_logits       -- (B, n_classes, H, W) float tensor
        include_decision -- bool; True = in-distribution, False = OOD

    load_state_dict delegates to the inner SegformerModel so the same
    best_model_miou.pt produced by train.py works without modification.
    """

    def __init__(self, in_channels=3, n_classes=19):
        super().__init__()
        self.segmenter = SegformerModel(in_channels, n_classes)

        if os.path.exists(_CONFIG_PATH):
            with open(_CONFIG_PATH) as f:
                cfg = json.load(f)
            self.ood_method    = cfg["method"]
            self.threshold     = cfg["threshold"]
            self.temperature   = cfg.get("temperature", 1.0)
            self.conf_threshold = cfg.get("conf_threshold", 0.5)
        else:
            self.ood_method    = "msp"
            self.threshold     = -0.5
            self.temperature   = 1.0
            self.conf_threshold = 0.5

    def forward(self, x: torch.Tensor):
        logits = self.segmenter(x)

        with torch.no_grad():
            if self.ood_method == "msp":
                score = score_msp(logits)
            elif self.ood_method == "temp_msp":
                score = score_msp(logits, self.temperature)
            elif self.ood_method == "energy":
                score = score_energy(logits)
            elif self.ood_method == "pixel_uncertainty":
                score = score_pixel_uncertainty(logits, self.conf_threshold)
            else:
                score = score_msp(logits)

        # msp / temp_msp / energy: score is negative; threshold is negative.
        #   score > threshold  →  less confident  →  OOD
        # pixel_uncertainty: score is positive; threshold is positive.
        #   score > threshold  →  more uncertain  →  OOD
        # In both cases: score < threshold  →  include (ID).
        include = score < self.threshold

        return logits, include

    def load_state_dict(self, state_dict, strict=True):
        # Delegate to the inner segmenter so best_model_miou.pt works as-is.
        return self.segmenter.load_state_dict(state_dict, strict=strict)


if __name__ == "__main__":
    model = Model(in_channels=3, n_classes=19)
    x = torch.randn(1, 3, 512, 1024)
    with torch.no_grad():
        seg_logits, include_decision = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Segmentation output shape : {seg_logits.shape}")
    print(f"Include decision          : {include_decision}")
    print(f"OOD method                : {model.ood_method}")
    print(f"Threshold                 : {model.threshold}")
    print(f"Parameter count           : {n_params:,}")
