import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation


class Model(nn.Module):
    """
    SegFormer-B2 segmentation model via HuggingFace Transformers.
    Initialised from nvidia/segformer-b2-finetuned-cityscapes-1024-1024
    (ImageNet + Cityscapes pretrained). The decode head is reinitialised
    when n_classes differs from the checkpoint's 19 classes.
    """

    def __init__(self, in_channels=3, n_classes=19):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
            num_labels=n_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        logits = self.model(pixel_values=x).logits  # (B, n_classes, H/4, W/4)
        logits = nn.functional.interpolate(
            logits, size=x.shape[-2:], mode='bilinear', align_corners=False
        )
        return logits


if __name__ == "__main__":
    model = Model(in_channels=3, n_classes=19)
    x = torch.randn(2, 3, 512, 1024)
    with torch.no_grad():
        logits = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape:    {logits.shape}")
    print(f"Parameter count: {n_params:,}")
