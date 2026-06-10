"""Build MILK10k processed CSV splits and optional image folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


# Parse arguments and run the dataset builder.
def main() -> None:
    """CLI entrypoint for dataset generation."""
    parser = argparse.ArgumentParser(description="Build MILK10k splits and task folders from raw files.")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-copy-images", action="store_true")
    args = parser.parse_args()

    from milk10k.data.build_dataset import build_milk10k

    build_milk10k(args.raw_dir, args.output_dir, seed=args.seed, copy_images=not args.no_copy_images)


if __name__ == "__main__":
    main()
