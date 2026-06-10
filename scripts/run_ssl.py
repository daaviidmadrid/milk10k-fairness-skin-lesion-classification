"""Run multimodal SSL pretraining followed by DA+CL fine-tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def resolve_input_path(path: Path) -> Path:
    """Resolve CLI paths from cwd first, then from the repository root."""
    return path if path.is_absolute() or path.exists() else REPO_ROOT / path


# Load a YAML config and execute SSL+DA+CL.
def main() -> None:
    """CLI entrypoint for multimodal SSL."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "ssl.yaml")
    args = parser.parse_args()

    from milk10k.training import ssl_multimodal
    from milk10k.utils.config import load_config

    ssl_multimodal.run(load_config(resolve_input_path(args.config)))


if __name__ == "__main__":
    main()
