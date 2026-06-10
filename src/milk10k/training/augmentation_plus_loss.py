"""Combined data augmentation and custom loss strategy."""

from __future__ import annotations

from milk10k.training.common import make_supervised_datasets, run_supervised
from milk10k.training.custom_loss import make_loss


# Run class-aware online augmentation together with class-skin-tone weighted loss.
def run(config: dict) -> dict:
    """Execute the DA+CL experiment."""
    _, class_to_idx = make_supervised_datasets(config, augmented=True)
    return run_supervised(config, loss_fn=make_loss(config, class_to_idx), augmented=True)

