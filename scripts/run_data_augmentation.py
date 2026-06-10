"""Run class-aware online data augmentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milk10k.training import data_augmentation
from milk10k.utils.config import load_config


# Load a YAML config and execute the augmentation strategy.
def main() -> None:
    """CLI entrypoint for data augmentation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data_augmentation.yaml"))
    args = parser.parse_args()
    data_augmentation.run(load_config(args.config))


if __name__ == "__main__":
    main()

