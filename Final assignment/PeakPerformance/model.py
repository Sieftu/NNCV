import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class Model(nn.Module):

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
        logits = self.model(pixel_values=x).logits  # (B, n_classes, H/4, W/4)
        logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logits


if __name__ == "__main__":
    model = Model(in_channels=3, n_classes=19)
    x = torch.randn(2, 3, 512, 1024)
    with torch.no_grad():
        logits = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape:    {logits.shape}")
    print(f"Parameter count: {n_params:,}")
