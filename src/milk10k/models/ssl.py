"""Multimodal contrastive model used for clinical-dermoscopic SSL."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from milk10k.models.backbones import EncoderOnly, backbone_feature_dim


class MultimodalContrastiveModel(nn.Module):
    """Shared encoder plus projection head for paired-modality SSL."""

    def __init__(self, backbone: str = "vit_b_32", projection_dim: int = 128, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = EncoderOnly(backbone=backbone, pretrained=pretrained)
        feature_dim = backbone_feature_dim(backbone)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def encode_project(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images and return L2-normalized projected embeddings."""
        return F.normalize(self.projector(self.encoder(x)), dim=1)

    def forward(self, clinical: torch.Tensor, dermoscopic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return projected clinical and dermoscopic embeddings."""
        return self.encode_project(clinical), self.encode_project(dermoscopic)


# Compute the symmetric bidirectional NT-Xent objective used in the thesis.
def nt_xent_multimodal_loss(z_clinical: torch.Tensor, z_dermoscopic: torch.Tensor, temperature: float = 0.10) -> torch.Tensor:
    """Compute symmetric cross-entropy over the clinical-dermoscopic similarity matrix."""
    logits = z_clinical @ z_dermoscopic.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
