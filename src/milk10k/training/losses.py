"""Loss functions used by the supervised MILK10k experiments."""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


class StandardCrossEntropy(nn.Module):
    """Thin wrapper that matches the shared loss interface."""

    def forward(self, logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute standard cross-entropy from logits and labels."""
        return F.cross_entropy(logits, batch["label"])


class SampleWeightedCrossEntropy(nn.Module):
    """Cross-entropy weighted by valid class and skin-tone pair."""

    def __init__(self, pair_weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("pair_weights", pair_weights.float())

    def forward(self, logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply precomputed pair weights to per-sample cross-entropy."""
        losses = F.cross_entropy(logits, batch["label"], reduction="none")
        weights = self.pair_weights[batch["label"], batch["skin_tone"].clamp(min=0, max=self.pair_weights.shape[1] - 1)]
        return (losses * weights).mean()


# Compute effective-number weights over class x skin-tone pairs.
def build_pair_weights(
    df: pd.DataFrame,
    class_to_idx: dict[str, int],
    task: str,
    beta: float,
    max_skin_tone: int = 5,
) -> torch.Tensor:
    """Return normalized class-skin-tone weights for valid skin tones 1..max_skin_tone."""
    from milk10k.data.labels import make_binary_label

    weights = torch.zeros((len(class_to_idx), max_skin_tone + 1), dtype=torch.float32)
    temp = df.copy()
    temp["label_name"] = temp["class_name"].map(make_binary_label) if task == "2f" else temp["class_name"]
    temp["label_idx"] = temp["label_name"].map(class_to_idx)
    temp["skin_tone_class"] = temp["skin_tone_class"].fillna(0).astype(int)

    positive_values: list[float] = []
    for (label_idx, skin_tone), group in temp.groupby(["label_idx", "skin_tone_class"]):
        # skin_tone_class=0 is unknown/missing; its pair weight remains zero.
        if pd.isna(label_idx) or skin_tone <= 0 or skin_tone > max_skin_tone:
            continue
        n = len(group)
        raw = (1.0 - beta) / (1.0 - beta**n)
        weights[int(label_idx), int(skin_tone)] = raw
        positive_values.append(raw)

    if positive_values:
        weights = weights / float(torch.tensor(positive_values).mean())
    return weights
