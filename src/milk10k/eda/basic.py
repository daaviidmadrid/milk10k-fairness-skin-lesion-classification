from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from milk10k.data.labels import make_binary_label



def _annotate_bars(ax, fontsize: int = 10) -> None:
    heights = [p.get_height() for p in ax.patches if p.get_height() > 0]
    if not heights:
        return
    ymax = max(heights)
    for patch in ax.patches:
        h = patch.get_height()
        if h > 0:
            ax.text(patch.get_x() + patch.get_width() / 2, h + ymax * 0.01, f"{int(h):,}", ha="center", fontsize=fontsize)
    ax.set_ylim(0, ymax * 1.12)


def plot_class_distribution(master_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(master_csv)
    counts_11f = df.groupby("class_name").size().reset_index(name="count").sort_values("count", ascending=False)
    df["binary_label"] = df["class_name"].map(make_binary_label)
    counts_2f = df.groupby("binary_label").size().reset_index(name="count")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2.4, 1]})
    sns.barplot(data=counts_11f, x="class_name", y="count", color="#7AA6A1", ax=axes[0])
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Number of images")
    _annotate_bars(axes[0], fontsize=9)

    sns.barplot(data=counts_2f, x="binary_label", y="count", order=["malignant", "benign"], palette=["#C96F5E", "#7A9E77"], ax=axes[1])
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Number of images")
    _annotate_bars(axes[1], fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metadata_distribution(master_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(master_csv)
    df["skin_tone_class"] = df["skin_tone_class"].fillna(0).astype(int)
    df["sex"] = df["sex"].fillna("unknown").astype(str).str.title()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.countplot(data=df, x="skin_tone_class", color="#A97C50", ax=axes[0])
    sns.countplot(data=df, x="sex", color="#8EA8C3", ax=axes[1])
    sns.histplot(data=df, x="age_approx", bins=np.arange(0, 91, 10), color="#6F8DB5", edgecolor="black", ax=axes[2])

    for ax in axes[:2]:
        _annotate_bars(ax, fontsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_xlabel("Skin tone class")
    axes[1].set_xlabel("Sex")
    axes[2].set_xlabel("Age")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_paired_examples(master_csv: Path, output_path: Path, pairs_per_tone: int = 3) -> None:
    df = pd.read_csv(master_csv)
    pivot = df.pivot_table(
        index=["lesion_id", "skin_tone_class"],
        columns="image_type",
        values="img_path",
        aggfunc="first",
    ).reset_index()
    cols = [c for c in pivot.columns if isinstance(c, str)]
    clinical_col = next(c for c in cols if "clin" in c.lower())
    derm_col = next(c for c in cols if "derm" in c.lower())
    tones = sorted(pivot["skin_tone_class"].dropna().astype(int).unique().tolist())

    fig, axes = plt.subplots(len(tones), pairs_per_tone * 2, figsize=(14, 2.2 * len(tones)))
    for row_idx, tone in enumerate(tones):
        rows = pivot[pivot["skin_tone_class"] == tone].dropna(subset=[clinical_col, derm_col]).head(pairs_per_tone)
        for pair_idx in range(pairs_per_tone):
            for offset, col in enumerate([clinical_col, derm_col]):
                ax = axes[row_idx, pair_idx * 2 + offset]
                ax.axis("off")
                if pair_idx < len(rows):
                    ax.imshow(Image.open(rows.iloc[pair_idx][col]).convert("RGB"))
                    if row_idx == 0:
                        ax.set_title("Clinical" if offset == 0 else "Dermoscopic")
        axes[row_idx, 0].set_ylabel(f"Skin tone {tone}", rotation=90, labelpad=24, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

