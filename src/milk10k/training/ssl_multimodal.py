"""Multimodal self-supervised pretraining plus downstream DA+CL fine-tuning."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from milk10k.data.datasets import PairedModalDataset
from milk10k.data.transforms import train_transform
from milk10k.models.ssl import MultimodalContrastiveModel, nt_xent_multimodal_loss
from milk10k.training.common import make_supervised_datasets, run_supervised, split_paths
from milk10k.training.custom_loss import make_loss
from milk10k.utils.config import ensure_dir
from milk10k.utils.seed import set_seed


# Train the contrastive multimodal SSL model on paired clinical-dermoscopic lesions.
def pretrain_ssl(config: dict) -> dict:
    """Run multimodal contrastive pretraining and return the encoder state."""
    set_seed(config.get("seed", 123))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    split_dir = Path(config["data"]["split_dir"])
    clinical_csv = split_dir / "clinical_train.csv"
    dermoscopic_csv = split_dir / "dermoscopic_train.csv"

    dataset = PairedModalDataset(clinical_csv, dermoscopic_csv, transform=train_transform(config["image_size"]))
    loader = DataLoader(dataset, batch_size=config["ssl_batch_size"], shuffle=True, num_workers=config.get("num_workers", 0))
    model = MultimodalContrastiveModel(
        backbone=config["backbone"],
        projection_dim=config.get("projection_dim", 128),
        pretrained=config.get("pretrained", True),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["ssl_lr"], weight_decay=config.get("weight_decay", 0.01))

    ssl_dir = ensure_dir(Path(config["output_dir"]) / "ssl_pretraining")
    best_loss = float("inf")
    best_encoder_state = None

    for epoch in range(1, config["ssl_epochs"] + 1):
        model.train()
        losses: list[float] = []
        for batch in tqdm(loader, desc=f"ssl epoch {epoch}", leave=False):
            clinical = batch["clinical"].to(device)
            dermoscopic = batch["dermoscopic"].to(device)
            optimizer.zero_grad(set_to_none=True)
            z_c, z_d = model(clinical, dermoscopic)
            loss = nt_xent_multimodal_loss(z_c, z_d, temperature=config.get("temperature", 0.10))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        epoch_loss = sum(losses) / max(1, len(losses))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_encoder_state = model.encoder.model.state_dict()
            torch.save(best_encoder_state, ssl_dir / "best_encoder.pt")

    return best_encoder_state or model.encoder.model.state_dict()


# Run SSL pretraining and then downstream DA+CL fine-tuning.
def run(config: dict) -> dict:
    """Execute the final SSL+DA+CL experiment."""
    encoder_state = pretrain_ssl(config)
    _, class_to_idx = make_supervised_datasets(config, augmented=True)
    loss_fn = make_loss(config, class_to_idx)
    return run_supervised(config, loss_fn=loss_fn, augmented=True, initial_state=encoder_state)
