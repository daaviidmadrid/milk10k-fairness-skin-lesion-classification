from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def find_repo_root(start: str | Path) -> Path:
    """Find the repository root from a file or directory inside the project."""
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / "src" / "milk10k").exists():
            return candidate
    return current


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve relative paths from the repo root so scripts work even when
    # launched from another current directory.
    repo_root = find_repo_root(path)
    if "output_dir" in config:
        config["output_dir"] = str((repo_root / config["output_dir"]).resolve())
    if "data" in config and "split_dir" in config["data"]:
        config["data"]["split_dir"] = str((repo_root / config["data"]["split_dir"]).resolve())
    return config


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
