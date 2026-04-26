import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig

from ood_scoring import score_msp, score_energy, score_pixel_uncertainty

_CONFIG_PATH = "/app/ood_config.json"


class Backbone(nn.Module):
    def __init__(self, in_channels=3, n_classes=19):
        super().__init__()
        local_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "segformer_config")
        if os.path.exists(local_config):
            config = SegformerConfig.from_pretrained(local_config)
        else:
            config = SegformerConfig.from_pretrained("/app/segformer_config")
        config.num_labels = n_classes
        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x):
        logits = self.model(pixel_values=x).logits
        logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logits


class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19):
        super().__init__()
        self.segmenter = Backbone(in_channels, n_classes)

        if os.path.exists(_CONFIG_PATH):
            with open(_CONFIG_PATH) as f:
                cfg = json.load(f)
            self.ood_method = cfg["method"]
            self.threshold = cfg["threshold"]
            self.temperature = cfg.get("temperature", 1.0)
            self.conf_threshold = cfg.get("conf_threshold", 0.5)
        else:
            self.ood_method = "msp"
            self.threshold = -0.5
            self.temperature = 1.0
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

        # MSP / Temp-MSP / Energy return negative scores; pixel_uncertainty returns
        # a positive fraction. In both conventions, a smaller score means more in-distribution.
        include = score < self.threshold

        return logits, include

    def load_state_dict(self, state_dict, strict=True):
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
