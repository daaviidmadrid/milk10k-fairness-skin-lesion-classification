"""Evaluation metrics used by the MILK10k experiments.

This module keeps predictive and fairness metrics in one place so every
training strategy is evaluated with the same protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float:
    """Compute binary or macro one-vs-rest AUC depending on the task."""
    if num_classes == 2:
        return float(roc_auc_score(y_true, y_prob[:, 1]))
    return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> dict[str, float]:
    """Return accuracy and AUC from integer labels and class probabilities."""
    y_pred = y_prob.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": compute_auc(y_true, y_prob, num_classes),
    }


def per_class_report(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """Return precision, recall and F1-score for every class."""
    y_pred = y_prob.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class_name": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def equalized_odds_gap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    num_classes: int,
    ignore_values: set | None = None,
) -> float:
    """Compute mean one-vs-rest Equalized Odds Gap using both TPR and FPR."""
    if ignore_values:
        keep = np.asarray([group not in ignore_values for group in groups], dtype=bool)
        y_true = y_true[keep]
        y_pred = y_pred[keep]
        groups = groups[keep]

    class_gaps: list[float] = []
    valid_groups = [g for g in pd.Series(groups).dropna().unique().tolist()]
    if len(valid_groups) < 2:
        return 0.0

    for class_idx in range(num_classes):
        tprs: list[float] = []
        fprs: list[float] = []
        for group in valid_groups:
            mask = groups == group
            yt = (y_true[mask] == class_idx).astype(int)
            yp = (y_pred[mask] == class_idx).astype(int)
            tp = int(((yt == 1) & (yp == 1)).sum())
            fn = int(((yt == 1) & (yp == 0)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum())
            tn = int(((yt == 0) & (yp == 0)).sum())
            tprs.append(tp / (tp + fn) if (tp + fn) else 0.0)
            fprs.append(fp / (fp + tn) if (fp + tn) else 0.0)

        # Equalized Odds depends on both true-positive and false-positive rates.
        tpr_mean = float(np.mean(tprs))
        fpr_mean = float(np.mean(fprs))
        class_gaps.append(
            0.5
            * (
                float(np.mean([abs(v - tpr_mean) for v in tprs]))
                + float(np.mean([abs(v - fpr_mean) for v in fprs]))
            )
        )
    return float(np.mean(class_gaps))


def fairness_performance_score(auc: float, eo_gaps: dict[str, float], alpha: float = 0.4) -> float:
    """Combine AUC and average EO Gap into the thesis selection score."""
    avg_eo = float(np.mean(list(eo_gaps.values()))) if eo_gaps else 0.0
    return float(alpha * auc + (1.0 - alpha) * (1.0 - avg_eo))
