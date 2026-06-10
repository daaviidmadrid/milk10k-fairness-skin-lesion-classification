"""Run class-aware online data augmentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def resolve_input_path(path: Path) -> Path:
    """Resolve CLI paths from cwd first, then from the repository root."""
    return path if path.is_absolute() or path.exists() else REPO_ROOT / path


# Load a YAML config and execute the augmentation strategy.
def main() -> None:
    """CLI entrypoint for data augmentation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "data_augmentation.yaml")
    args = parser.parse_args()

    from milk10k.training import data_augmentation
    from milk10k.utils.config import load_config

    data_augmentation.run(load_config(resolve_input_path(args.config)))


if __name__ == "__main__":
    main()
