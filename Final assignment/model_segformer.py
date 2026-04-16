import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class Model(nn.Module):
    """
    SegFormer-based segmentation model using mit_b2 encoder pretrained on ImageNet.
    Upsamples the /4-stride output back to input resolution.
    """

    def __init__(self, in_channels=3, n_classes=19):
        """
        Args:
            in_channels (int): Number of input channels. Default is 3 for RGB images.
            n_classes (int): Number of output classes. Default is 19 for the Cityscapes dataset.
        """
        super().__init__()
        self.segformer = smp.Segformer(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=n_classes,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, n_classes, H, W).
        """
        logits = self.segformer(x)
        # SegFormer decoder outputs at H/4, W/4 — upsample back to input resolution
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logits


if __name__ == "__main__":
    model = Model(in_channels=3, n_classes=19)
    x = torch.randn(2, 3, 512, 1024)
    logits = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {logits.shape}")
    print(f"Parameter count: {n_params:,}")
