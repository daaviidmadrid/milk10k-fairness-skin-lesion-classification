"""Baseline supervised training strategy."""

from __future__ import annotations

from milk10k.training.common import run_supervised


# Run ImageNet-initialized supervised fine-tuning with standard cross-entropy.
def run(config: dict) -> dict:
    """Execute the baseline experiment."""
    return run_supervised(config, augmented=False)

