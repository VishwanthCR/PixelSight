"""Modular residual multi-task CNN baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout), nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class MultiTaskUrbanSR(nn.Module):
    """4x SR with urban probabilities and uncertainty from shared features."""

    def __init__(self, in_channels: int = 4, out_channels: int = 4, urban_classes: int = 5, features: int = 48, blocks: int = 6, dropout: float = 0.15, scale: int = 4) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ValueError("scale must be 2 or 4")
        self.scale = scale
        self.stem = nn.Conv2d(in_channels, features, 3, padding=1)
        self.body = nn.Sequential(*(ResidualBlock(features, dropout) for _ in range(blocks)))
        self.fusion = nn.Conv2d(features, features, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(features, features * scale * scale, 3, padding=1), nn.PixelShuffle(scale), nn.ReLU(inplace=True)
        )
        self.sr_head = nn.Conv2d(features, out_channels, 3, padding=1)
        self.urban_head = nn.Sequential(nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True), nn.Dropout2d(dropout), nn.Conv2d(features, urban_classes, 1))
        self.logvar_head = nn.Sequential(nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True), nn.Dropout2d(dropout), nn.Conv2d(features, out_channels, 1))

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.fusion(self.body(F.relu(self.stem(inputs))))
        features = features + self.stem(inputs)
        upsampled = self.upsample(features)
        baseline = F.interpolate(inputs, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return {
            "super_resolved": torch.clamp(baseline + self.sr_head(upsampled), 0.0, 1.0),
            "urban_logits": self.urban_head(upsampled),
            "log_variance": self.logvar_head(upsampled),
            "features": upsampled,
        }
