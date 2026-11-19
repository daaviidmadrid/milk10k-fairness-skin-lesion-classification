# Reproduction notes

## Environment and data

Python ≥ 3.10. A CUDA-enabled GPU is recommended for ViT and SSL runs. Install a compatible PyTorch build for the target hardware, then install the project from the repository root:

```bash
pip install -e .
```

Alternatively, use `pip install -r requirements.txt`.

Download the training files from the [official ISIC data page](https://challenge.isic-archive.com/data/):

```text
data/raw/
├── MILK10k_Training_Input.zip
├── MILK10k_Training_Metadata.csv
└── MILK10k_Training_GroundTruth.csv
```

Build the processed dataset and optional EDA outputs:

```bash
python scripts/build_milk10k_dataset.py --raw-dir data/raw --output-dir data/processed
python scripts/run_eda.py --master-csv data/processed/master_table.csv --output-dir outputs/eda
```

The builder creates `master_table.csv`, lesion-level 80/10/10 splits in `data/processed/splits_80_10_10_by_type/`, and image-folder exports for both tasks. Clinical and dermoscopic views of a lesion stay in the same split. Experiments use the labelled training data rather than the official challenge test set, whose labels are not public.

## Experiments

Run the retained 11-class experiments:

```bash
python scripts/run_baseline.py --config configs/baseline.yaml
python scripts/run_data_augmentation.py --config configs/data_augmentation.yaml
python scripts/run_custom_loss.py --config configs/custom_loss.yaml
python scripts/run_aug_plus_loss.py --config configs/aug_plus_loss.yaml
python scripts/run_ssl.py --config configs/ssl.yaml
```

For binary classification:

```bash
python scripts/run_baseline.py --config configs/baseline_2f.yaml
python scripts/run_data_augmentation.py --config configs/data_augmentation_2f.yaml
python scripts/run_custom_loss.py --config configs/custom_loss_2f.yaml
python scripts/run_aug_plus_loss.py --config configs/aug_plus_loss_2f.yaml
python scripts/run_ssl.py --config configs/ssl_2f.yaml
```

The scripts resolve their defaults from the repository location. Relative paths within YAML configurations resolve from the repository root.

All retained configurations use dermoscopic ViT-B/32, image size 224, learning rate 3e-5, seed 123, up to 20 supervised epochs and patience 3.

| Configuration | Task | Batch size | Strategy settings |
| --- | --- | ---: | --- |
| [`baseline.yaml`](../configs/baseline.yaml) | 11-class | 32 | CE |
| [`baseline_2f.yaml`](../configs/baseline_2f.yaml) | Binary | 16 | CE |
| [`data_augmentation.yaml`](../configs/data_augmentation.yaml) | 11-class | 128 | Repeat cap 6; weighted sampling |
| [`data_augmentation_2f.yaml`](../configs/data_augmentation_2f.yaml) | Binary | 64 | Repeat cap 6; weighted sampling |
| [`custom_loss.yaml`](../configs/custom_loss.yaml) | 11-class | 32 | β = 0.999 |
| [`custom_loss_2f.yaml`](../configs/custom_loss_2f.yaml) | Binary | 16 | β = 0.99 |
| [`aug_plus_loss.yaml`](../configs/aug_plus_loss.yaml) | 11-class | 128 | DA + CL; β = 0.999 |
| [`aug_plus_loss_2f.yaml`](../configs/aug_plus_loss_2f.yaml) | Binary | 64 | DA + CL; β = 0.99 |
| [`ssl.yaml`](../configs/ssl.yaml) | 11-class | 128 | SSL + DA + CL; τ = 0.10 |
| [`ssl_2f.yaml`](../configs/ssl_2f.yaml) | Binary | 64 | SSL + DA + CL; τ = 0.10 |

SSL uses only training pairs, a shared encoder and a 128-dimensional projection head, with batch size 64 for 10 pretraining epochs. Positive pairs are the two modalities of the same lesion; other lesions in the batch supply negatives. Fine-tuning retains the encoder and discards the projection head.

## Evaluation details

The implementation is in [`metrics.py`](../src/milk10k/evaluation/metrics.py) and [`engine.py`](../src/milk10k/training/engine.py):

- Binary AUC uses the malignant-class probability; 11-class AUC is macro one-vs-rest.
- Accuracy and per-class precision, recall and F1 use argmax predictions.
- EO Gap averages the mean absolute deviations of group TPR and FPR from their respective group means, then averages across one-vs-rest classes. A rate with a zero denominator is set to zero.
- EO-Skin, EO-Sex and EO-Age use skin-tone categories, sex and decade age bins respectively. The current code excludes skin-tone category `0` from EO-Skin and assigns it zero weight in the custom loss. The [official dataset schema](https://api.isic-archive.com/doi/milk10k/) defines `0` as the darkest skin-tone category; interpret these results with that implementation choice in mind.
- FP Score is `0.4 * AUC + 0.6 * (1 - mean(EO-Skin, EO-Sex, EO-Age))`. Validation AUC controls checkpoint selection and early stopping.

The numbers in the README are the retained thesis results. Raw outputs and checkpoints are excluded from version control. Exact reproduction can vary with hardware, library versions and nondeterministic GPU operations.

## Outputs and notebooks

Each experiment writes `best_model.pt`, aggregate metrics in `summary.csv`, and class-level metrics in `per_class.csv` under its configured `outputs/` directory. SSL also writes `ssl_pretraining/best_encoder.pt`.

The original execution notebooks are retained for traceability:

| Notebook | Experiment |
| --- | --- |
| [`MILK10k_EDA.ipynb`](../notebooks/MILK10k_EDA.ipynb) | Dataset exploration and figures |
| [`model_training.ipynb`](../notebooks/model_training.ipynb) | Backbone, modality and task comparison |
| [`model_training_online_augmentation.ipynb`](../notebooks/model_training_online_augmentation.ipynb) | DA |
| [`model_training_custom.ipynb`](../notebooks/model_training_custom.ipynb) | CL |
| [`model_training_custom_loss_online_augmentation.ipynb`](../notebooks/model_training_custom_loss_online_augmentation.ipynb) | DA + CL |
| [`multimodal_contrastive_self_supervised_learning.ipynb`](../notebooks/multimodal_contrastive_self_supervised_learning.ipynb) | SSL + DA + CL |

Use [`configs/`](../configs/), [`scripts/`](../scripts/) and [`src/`](../src/) for new runs. Raw data, processed data, checkpoints and generated outputs are ignored by Git. The repository is intended for research, not clinical diagnosis.
