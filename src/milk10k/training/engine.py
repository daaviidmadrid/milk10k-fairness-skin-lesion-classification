"""Shared supervised training and evaluation loops.

The strategy files configure datasets, samplers and losses; this module only
handles optimization, checkpoint selection and metric computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from milk10k.evaluation.metrics import classification_metrics, equalized_odds_gap, fairness_performance_score
from milk10k.utils.config import ensure_dir


# Move tensor values in a batch to the selected device.
def move_batch(batch: dict, device: torch.device) -> dict:
    """Move tensor batch values to device while leaving metadata untouched."""
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# Run one supervised training epoch.
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: torch.device,
) -> float:
    """Train for one epoch and return mean loss."""
    model.train()
    losses: list[float] = []
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["image"])
        loss = loss_fn(logits, batch)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


# Evaluate a supervised classifier and collect predictions plus metadata.
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    device: torch.device,
    num_classes: int,
    alpha: float = 0.4,
) -> dict:
    """Evaluate a model with predictive metrics and Equalized Odds Gaps."""
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_prob: list[np.ndarray] = []
    skin: list[int] = []
    sex: list[str] = []
    age_group: list[str] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch(batch, device)
        logits = model(batch["image"])
        loss = loss_fn(logits, batch)
        probs = torch.softmax(logits, dim=1)
        losses.append(float(loss.detach().cpu()))
        y_true.extend(batch["label"].detach().cpu().numpy().tolist())
        y_prob.extend(probs.detach().cpu().numpy())
        skin.extend(batch["skin_tone"].detach().cpu().numpy().tolist())
        sex.extend(list(batch["sex"]))
        ages = batch["age"].detach().cpu().numpy()
        age_group.extend(["unknown" if age < 0 else str(int(age // 10 * 10)) for age in ages])

    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    y_pred_arr = y_prob_arr.argmax(axis=1)
    metrics = classification_metrics(y_true_arr, y_prob_arr, num_classes)
    # skin_tone_class=0 denotes unknown/missing skin tone and is excluded from EO-Skin.
    eo = {
        "skin": equalized_odds_gap(y_true_arr, y_pred_arr, np.asarray(skin), num_classes, ignore_values={0}),
        "sex": equalized_odds_gap(y_true_arr, y_pred_arr, np.asarray(sex), num_classes),
        "age": equalized_odds_gap(y_true_arr, y_pred_arr, np.asarray(age_group), num_classes),
    }
    metrics.update(
        {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "eo_skin": eo["skin"],
            "eo_sex": eo["sex"],
            "eo_age": eo["age"],
            "one_minus_avg_eo": 1.0 - float(np.mean(list(eo.values()))),
            "fp_score": fairness_performance_score(metrics["auc"], eo, alpha=alpha),
            "y_true": y_true_arr,
            "y_prob": y_prob_arr,
        }
    )
    return metrics


# Fit a classifier using validation AUC for checkpoint selection.
def fit_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    epochs: int,
    patience: int,
    output_dir: str | Path,
    alpha: float = 0.4,
) -> dict:
    """Train with early stopping based on validation AUC."""
    output_dir = ensure_dir(output_dir)
    model.to(device)
    if isinstance(loss_fn, nn.Module):
        loss_fn.to(device)
    best_score = -float("inf")
    best_metrics: dict | None = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device, num_classes, alpha=alpha)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = train_loss

        if val_metrics["auc"] > best_score:
            best_score = val_metrics["auc"]
            best_metrics = val_metrics
            stale_epochs = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    return best_metrics or {}
