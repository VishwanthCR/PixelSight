"""Minimal reproducible paired-raster training runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
import rasterio

from ..data.raster import pair_files, read_reflectance
from ..models import MultiTaskUrbanSR
from .losses import MultiTaskLoss


def train_from_config(config_path: str | Path) -> Path:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskUrbanSR(**config.get("model", {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("training", {}).get("learning_rate", 2e-4)))
    criterion = MultiTaskLoss(**config.get("loss", {}))
    epochs = int(config.get("training", {}).get("epochs", 1))
    root = config.get("data", {}).get("root", "data")
    pairs = pair_files(root, "train")
    if not pairs:
        raise RuntimeError(f"No training pairs found under {root}/train/lr")
    model.train()
    for _ in range(epochs):
        for lr_path, hr_path, label_path in pairs:
            lr, _ = read_reflectance(lr_path); hr, _ = read_reflectance(hr_path)
            lr_tensor = torch.from_numpy(lr).unsqueeze(0).to(device); hr_tensor = torch.from_numpy(hr).unsqueeze(0).to(device)
            labels = None
            if label_path:
                with rasterio.open(label_path) as label_source:
                    label = label_source.read(1)
                labels = torch.from_numpy(label[None, : hr.shape[1], : hr.shape[2]].astype(np.int64)).to(device)
            optimizer.zero_grad(); losses = criterion(model(lr_tensor), hr_tensor, labels); losses["total"].backward(); optimizer.step()
    checkpoint = Path(config.get("training", {}).get("checkpoint", "checkpoints/urban_satellite_super_resolution.pt"))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint)
    return checkpoint
