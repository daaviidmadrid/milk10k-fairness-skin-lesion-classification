"""Run class-skin-tone custom loss training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milk10k.training import custom_loss
from milk10k.utils.config import load_config


# Load a YAML config and execute the custom loss strategy.
def main() -> None:
    """CLI entrypoint for custom loss training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/custom_loss.yaml"))
    args = parser.parse_args()
    custom_loss.run(load_config(args.config))


if __name__ == "__main__":
    main()

