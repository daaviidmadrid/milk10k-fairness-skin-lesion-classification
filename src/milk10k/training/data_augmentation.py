"""Class-aware online data augmentation strategy."""

from __future__ import annotations

from milk10k.training.common import run_supervised


# Run supervised fine-tuning with class-aware virtual repetition and online augmentation.
def run(config: dict) -> dict:
    """Execute the data augmentation experiment."""
    return run_supervised(config, augmented=True)

