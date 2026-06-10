"""Run multimodal SSL pretraining followed by DA+CL fine-tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milk10k.training import ssl_multimodal
from milk10k.utils.config import load_config


# Load a YAML config and execute SSL+DA+CL.
def main() -> None:
    """CLI entrypoint for multimodal SSL."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ssl.yaml"))
    args = parser.parse_args()
    ssl_multimodal.run(load_config(args.config))


if __name__ == "__main__":
    main()

