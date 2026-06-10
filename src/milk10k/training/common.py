"""Common setup helpers shared by all supervised strategies."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from milk10k.data.datasets import ImageCsvDataset, RepeatedAugmentationDataset, build_class_mapping, class_names_from_mapping, task_label_name
from milk10k.data.transforms import eval_transform, online_augmentation_transform, train_transform
from milk10k.evaluation.metrics import per_class_report
from milk10k.models.backbones import create_classifier
from milk10k.training.engine import evaluate, fit_supervised
from milk10k.training.losses import StandardCrossEntropy
from milk10k.utils.config import ensure_dir
from milk10k.utils.seed import set_seed


# Resolve train, validation and test CSV paths for one modality.
def split_paths(split_dir: str | Path, modality: str) -> dict[str, Path]:
    """Return split CSV paths for the selected modality."""
    split_dir = Path(split_dir)
    return {name: split_dir / f"{modality}_{name}.csv" for name in ["train", "val", "test"]}


# Create standard train/val/test datasets for supervised experiments.
def make_supervised_datasets(config: dict, augmented: bool = False) -> tuple[dict, dict[str, int]]:
    """Build supervised datasets and the shared class mapping."""
    paths = split_paths(config["data"]["split_dir"], config["data"]["modality"])
    train_df = pd.read_csv(paths["train"])
    class_to_idx = build_class_mapping(train_df, config["task"])

    base_transform = train_transform(config["image_size"])
    if augmented:
        train_ds = RepeatedAugmentationDataset(
            train_df,
            task=config["task"],
            base_transform=base_transform,
            aug_transform=online_augmentation_transform(config["image_size"]),
            class_to_idx=class_to_idx,
            exponent=config.get("repeat_exponent", 0.5),
            cap=config.get("repeat_cap", 6),
        )
    else:
        train_ds = ImageCsvDataset(paths["train"], config["task"], transform=base_transform, class_to_idx=class_to_idx)

    datasets = {
        "train": train_ds,
        "val": ImageCsvDataset(paths["val"], config["task"], transform=eval_transform(config["image_size"]), class_to_idx=class_to_idx),
        "test": ImageCsvDataset(paths["test"], config["task"], transform=eval_transform(config["image_size"]), class_to_idx=class_to_idx),
    }
    return datasets, class_to_idx


# Build the class-dependent sampler described in the thesis.
def make_class_weighted_sampler(dataset) -> WeightedRandomSampler:
    """Create weights proportional to the inverse square root of class frequency."""
    class_names = getattr(dataset, "sampling_class_names", None)
    if class_names is None:
        class_names = [task_label_name(row, dataset.task) for _, row in dataset.df.iterrows()]
    class_counts = getattr(dataset, "sampling_class_counts", None) or Counter(class_names)
    weights = torch.tensor([1.0 / (float(class_counts[name]) ** 0.5) for name in class_names], dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


# Create PyTorch dataloaders from datasets.
def make_loaders(datasets: dict, batch_size: int, num_workers: int = 0, weighted_sampler: bool = False) -> dict[str, DataLoader]:
    """Build dataloaders for train, validation and test splits."""
    train_sampler = make_class_weighted_sampler(datasets["train"]) if weighted_sampler else None
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
        ),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


# Save scalar and per-class results for one completed experiment.
def save_results(output_dir: str | Path, metrics: dict, class_names: list[str]) -> None:
    """Write summary metrics and per-class report to CSV files."""
    output_dir = ensure_dir(output_dir)
    scalar = {k: v for k, v in metrics.items() if k not in {"y_true", "y_prob"}}
    pd.DataFrame([scalar]).to_csv(output_dir / "summary.csv", index=False)
    per_class_report(metrics["y_true"], metrics["y_prob"], class_names).to_csv(output_dir / "per_class.csv", index=False)


# Run a complete supervised train-val-test experiment.
def run_supervised(config: dict, loss_fn=None, augmented: bool = False, initial_state: dict | None = None) -> dict:
    """Train a supervised model and evaluate the best checkpoint on test."""
    set_seed(config.get("seed", 123))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    datasets, class_to_idx = make_supervised_datasets(config, augmented=augmented)
    loaders = make_loaders(
        datasets,
        config["batch_size"],
        config.get("num_workers", 0),
        weighted_sampler=config.get("weighted_sampler", False),
    )
    class_names = class_names_from_mapping(class_to_idx)

    model = create_classifier(config["backbone"], len(class_names), pretrained=config.get("pretrained", True))
    if initial_state is not None:
        model.load_state_dict(initial_state, strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0.01))
    loss_fn = loss_fn or StandardCrossEntropy()

    output_dir = ensure_dir(config["output_dir"])
    fit_supervised(
        model,
        loaders["train"],
        loaders["val"],
        loss_fn,
        optimizer,
        device,
        len(class_names),
        config["epochs"],
        config["patience"],
        output_dir,
        alpha=config.get("alpha", 0.4),
    )
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, loaders["test"], loss_fn, device, len(class_names), alpha=config.get("alpha", 0.4))
    save_results(output_dir, test_metrics, class_names)
    return test_metrics
