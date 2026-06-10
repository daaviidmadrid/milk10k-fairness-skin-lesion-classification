"""Class-skin-tone sample-weighted loss strategy."""

from __future__ import annotations

import pandas as pd

from milk10k.training.common import make_supervised_datasets, run_supervised, split_paths
from milk10k.training.losses import SampleWeightedCrossEntropy, build_pair_weights


# Build the final class x skin-tone weighted loss used in the thesis.
def make_loss(config: dict, class_to_idx: dict[str, int]) -> SampleWeightedCrossEntropy:
    """Create SampleWeightedCrossEntropy from training-set pair counts."""
    train_csv = split_paths(config["data"]["split_dir"], config["data"]["modality"])["train"]
    train_df = pd.read_csv(train_csv)
    weights = build_pair_weights(train_df, class_to_idx, config["task"], beta=config["beta"])
    return SampleWeightedCrossEntropy(weights)


# Run supervised fine-tuning with class-skin-tone reweighting.
def run(config: dict) -> dict:
    """Execute the custom loss experiment."""
    _, class_to_idx = make_supervised_datasets(config, augmented=False)
    return run_supervised(config, loss_fn=make_loss(config, class_to_idx), augmented=False)

