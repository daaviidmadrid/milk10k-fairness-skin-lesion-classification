"""Generate the basic EDA figures used to describe MILK10k."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milk10k.eda.basic import plot_class_distribution, plot_metadata_distribution, plot_paired_examples


# Parse arguments and generate all EDA figures.
def main() -> None:
    """CLI entrypoint for EDA figure generation."""
    parser = argparse.ArgumentParser(description="Generate basic MILK10k EDA figures.")
    parser.add_argument("--master-csv", type=Path, default=Path("data/processed/master_table.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eda"))
    args = parser.parse_args()

    plot_class_distribution(args.master_csv, args.output_dir / "class_distribution.png")
    plot_metadata_distribution(args.master_csv, args.output_dir / "metadata_distribution.png")
    plot_paired_examples(args.master_csv, args.output_dir / "paired_examples.png")


if __name__ == "__main__":
    main()
