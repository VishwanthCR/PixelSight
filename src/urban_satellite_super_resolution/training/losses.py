"""Configurable losses for multi-task urban SR."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def charbonnier(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3, mask: torch.Tensor | None = None) -> torch.Tensor:
    values = torch.sqrt((prediction - target).pow(2) + epsilon**2)
    if mask is not None:
        values = values * mask
        return values.sum() / mask.sum().clamp_min(1.0)
    return values.mean()


def spectral_angle(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    pred = prediction.permute(0, 2, 3, 1).flatten(0, 2)
    true = target.permute(0, 2, 3, 1).flatten(0, 2)
    cosine = (pred * true).sum(-1) / (pred.norm(dim=-1) * true.norm(dim=-1)).clamp_min(epsilon)
    return torch.acos(cosine.clamp(-1.0 + epsilon, 1.0 - epsilon)).mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255, smooth: float = 1.0) -> torch.Tensor:
    classes = logits.shape[1]
    valid = target != ignore_index
    safe_target = target.clamp(0, classes - 1)
    one_hot = F.one_hot(safe_target, classes).permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
    probabilities = torch.softmax(logits, dim=1) * valid.unsqueeze(1)
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    return (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()


def gaussian_nll(prediction: torch.Tensor, target: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    variance = log_variance.clamp(-10.0, 5.0).exp()
    return (0.5 * ((prediction - target).pow(2) / variance + log_variance)).mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, sr_weight: float = 1.0, ssim_weight: float = 0.1, sam_weight: float = 0.05, segmentation_weight: float = 1.0, uncertainty_weight: float = 0.1, ignore_index: int = 255) -> None:
        super().__init__()
        self.weights = (sr_weight, ssim_weight, sam_weight, segmentation_weight, uncertainty_weight)
        self.ignore_index = ignore_index

    def forward(self, outputs: dict[str, torch.Tensor], sr_target: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        sr_loss = charbonnier(outputs["super_resolved"], sr_target)
        sam_loss = spectral_angle(outputs["super_resolved"], sr_target)
        uncertainty_loss = gaussian_nll(outputs["super_resolved"], sr_target, outputs["log_variance"])
        segmentation_loss = torch.zeros((), device=sr_target.device)
        if labels is not None:
            segmentation_loss = F.cross_entropy(outputs["urban_logits"], labels, ignore_index=self.ignore_index) + dice_loss(outputs["urban_logits"], labels, self.ignore_index)
        total = self.weights[0] * sr_loss + self.weights[2] * sam_loss + self.weights[3] * segmentation_loss + self.weights[4] * uncertainty_loss
        return {"total": total, "sr": sr_loss, "sam": sam_loss, "segmentation": segmentation_loss, "uncertainty": uncertainty_loss}
