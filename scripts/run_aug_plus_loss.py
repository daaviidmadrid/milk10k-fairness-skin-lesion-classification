"""Run the combined data augmentation and custom loss experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milk10k.training import augmentation_plus_loss
from milk10k.utils.config import load_config


# Load a YAML config and execute DA+CL.
def main() -> None:
    """CLI entrypoint for the combined strategy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/aug_plus_loss.yaml"))
    args = parser.parse_args()
    augmentation_plus_loss.run(load_config(args.config))


if __name__ == "__main__":
    main()

