# build_dataset.py - Build the MILK10k dataset with lesion-aware splits and export CSVs and image folders.

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from milk10k.data.labels import make_binary_label


# Extract the ZIP file containing the images to the specified output directory. If the images have already been extracted, it will skip the extraction process.
def extract_zip(zip_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / ".extracted"
    if marker.exists():
        return output_dir
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_dir)
    marker.write_text("ok", encoding="utf-8")
    return output_dir


# Find the image path for a given lesion ID and ISIC ID within the specified root directory.
def find_image_path(root: Path, lesion_id: str, isic_id: str) -> Path | None:
    lesion_dir = root / str(lesion_id)
    if not lesion_dir.exists():
        return None
    matches = list(lesion_dir.glob(f"{isic_id}.*"))
    if not matches:
        matches = list(lesion_dir.glob(f"*{isic_id}*"))
    return matches[0] if matches else None


# Load the master table by merging the metadata and ground truth CSV files, and add the image paths for each lesion.
def load_master_table(raw_dir: Path, image_root: Path) -> pd.DataFrame:
    metadata = pd.read_csv(raw_dir / "MILK10k_Training_Metadata.csv")
    ground_truth = pd.read_csv(raw_dir / "MILK10k_Training_GroundTruth.csv")
    label_cols = [c for c in ground_truth.columns if c != "lesion_id"]

    gt = ground_truth.copy()
    gt["class_idx"] = gt[label_cols].values.argmax(axis=1)
    gt["class_name"] = gt["class_idx"].map({i: name for i, name in enumerate(label_cols)})

    master = metadata.merge(gt[["lesion_id", "class_idx", "class_name"]], on="lesion_id", how="left")
    # MILK10k uses skin_tone_class=0 for unknown/missing skin-tone information.
    master["skin_tone_class"] = master["skin_tone_class"].fillna(0).astype(int)
    master["sex"] = master["sex"].fillna("unknown").astype(str).str.lower()
    master["img_path"] = [
        find_image_path(image_root, row.lesion_id, row.isic_id) for row in master.itertuples(index=False)
    ]
    master = master[master["img_path"].notna()].copy()
    master["img_path"] = master["img_path"].astype(str)
    return master


# Perform a lesion-aware split of the dataset into train, validation, and test sets.
def lesion_aware_split(
    master: pd.DataFrame,
    seed: int = 123,
    min_stratum_count: int = 5,
) -> dict[str, set[str]]:
    lesion_table = (
        master.groupby("lesion_id", as_index=False)
        .agg({"class_idx": "first", "skin_tone_class": "first"})
    )
    lesion_table["stratum"] = (
        lesion_table["class_idx"].astype(str) + "_" + lesion_table["skin_tone_class"].astype(str)
    )
    counts = lesion_table["stratum"].value_counts()
    rare = counts[counts < min_stratum_count].index
    lesion_table.loc[lesion_table["stratum"].isin(rare), "stratum"] = (
        lesion_table["class_idx"].astype(str) + "_other"
    )

    if (lesion_table["stratum"].value_counts() < 2).any():
        lesion_table["stratum"] = lesion_table["class_idx"].astype(str)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, temp_idx = next(sss.split(lesion_table["lesion_id"], lesion_table["stratum"]))

    train = set(lesion_table.iloc[train_idx]["lesion_id"])
    temp = lesion_table.iloc[temp_idx].reset_index(drop=True)

    sss_temp = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(sss_temp.split(temp["lesion_id"], temp["stratum"]))
    return {
        "train": train,
        "val": set(temp.iloc[val_idx]["lesion_id"]),
        "test": set(temp.iloc[test_idx]["lesion_id"]),
    }


# Export the split CSV files for each image type and split, containing the specified columns.
def export_split_csvs(master: pd.DataFrame, lesion_splits: dict[str, set[str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = ["img_path", "lesion_id", "isic_id", "image_type", "class_idx", "class_name", "skin_tone_class", "sex", "age_approx"]
    for image_type, image_df in master.groupby("image_type"):
        for split_name, lesion_ids in lesion_splits.items():
            split_df = image_df[image_df["lesion_id"].isin(lesion_ids)].copy()
            split_df["img_path"] = split_df["img_path"].apply(
                lambda path: os.path.relpath(Path(path), start=output_dir)
            )
            split_df[[c for c in cols if c in split_df.columns]].to_csv(
                output_dir / f"{image_type}_{split_name}.csv",
                index=False,
            )


# Export the image folders for each image type and split.
def export_image_folders(master: pd.DataFrame, lesion_splits: dict[str, set[str]], output_dir: Path) -> None:
    for task in ["11f", "2f"]:
        for image_type, image_df in master.groupby("image_type"):
            for split_name, lesion_ids in lesion_splits.items():
                split_df = image_df[image_df["lesion_id"].isin(lesion_ids)].copy()
                labels = split_df["class_name"] if task == "11f" else split_df["class_name"].map(make_binary_label)
                for row, label in zip(split_df.itertuples(index=False), labels):
                    dst_dir = output_dir / task / image_type / split_name / label
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(row.img_path, dst_dir / Path(row.img_path).name)


# Main function to build the MILK10k dataset by extracting images, loading the master table, performing lesion-aware splits, and exporting CSVs and image folders.
def build_milk10k(raw_dir: Path, output_dir: Path, seed: int = 123, copy_images: bool = True) -> None:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    image_zip = raw_dir / "MILK10k_Training_Input.zip"
    image_root = extract_zip(image_zip, output_dir / "extracted" / "MILK10k_Training_Input")
    nested = image_root / "MILK10k_Training_Input"
    if nested.exists():
        image_root = nested

    master = load_master_table(raw_dir, image_root)
    lesion_splits = lesion_aware_split(master, seed=seed)

    table_dir = output_dir / "splits_80_10_10_by_type"
    export_split_csvs(master, lesion_splits, table_dir)
    master.to_csv(output_dir / "master_table.csv", index=False)
    if copy_images:
        export_image_folders(master, lesion_splits, output_dir / "datasets")
