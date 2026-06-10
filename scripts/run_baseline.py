"""Run the supervised baseline experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milk10k.training import baseline
from milk10k.utils.config import load_config


# Load a YAML config and execute baseline fine-tuning.
def main() -> None:
    """CLI entrypoint for the baseline experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = parser.parse_args()
    baseline.run(load_config(args.config))


if __name__ == "__main__":
    main()

