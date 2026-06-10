"""Backbone and classifier factories for the supervised experiments."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


# Replace the final layer of a torchvision model with a task-specific head.
def create_classifier(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a ResNet18 or ViT-B/32 classifier."""
    backbone = backbone.lower()
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if backbone in {"vit_b_32", "vit"}:
        weights = models.ViT_B_32_Weights.DEFAULT if pretrained else None
        model = models.vit_b_32(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported backbone: {backbone}")


# Return the feature dimension used by the selected backbone.
def backbone_feature_dim(backbone: str) -> int:
    """Return the default feature dimension for supported backbones."""
    backbone = backbone.lower()
    if backbone == "resnet18":
        return 512
    if backbone in {"vit_b_32", "vit"}:
        return 768
    raise ValueError(f"Unsupported backbone: {backbone}")


class EncoderOnly(nn.Module):
    """Feature extractor wrapper used by multimodal SSL."""

    def __init__(self, backbone: str = "vit_b_32", pretrained: bool = True) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.model = create_classifier(backbone, num_classes=2, pretrained=pretrained)
        if backbone.lower() == "resnet18":
            self.model.fc = nn.Identity()
        elif backbone.lower() in {"vit_b_32", "vit"}:
            self.model.heads.head = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features without a classification head."""
        return self.model(x)

