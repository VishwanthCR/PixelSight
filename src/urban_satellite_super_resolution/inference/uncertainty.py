"""Monte-Carlo dropout inference and confidence classification."""

from __future__ import annotations

import torch
from torch import nn


def enable_dropout(model: nn.Module) -> None:
    """Enable dropout modules while keeping normalization layers in eval mode."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def mc_predict(model: nn.Module, inputs: torch.Tensor, passes: int = 20) -> dict[str, torch.Tensor]:
    """Return mean SR, predictive std, urban probabilities, and confidence classes."""
    if passes < 2:
        raise ValueError("At least two stochastic passes are required")
    model.eval()
    enable_dropout(model)
    sr_samples, urban_samples = [], []
    with torch.no_grad():
        for _ in range(passes):
            output = model(inputs)
            sr_samples.append(output["super_resolved"])
            urban_samples.append(torch.softmax(output["urban_logits"], dim=1))
    sr_stack = torch.stack(sr_samples)
    urban_stack = torch.stack(urban_samples)
    mean_sr = sr_stack.mean(dim=0)
    std_sr = sr_stack.std(dim=0, unbiased=True)
    mean_urban = urban_stack.mean(dim=0)
    urban_std = urban_stack.std(dim=0, unbiased=True).mean(dim=1, keepdim=True)
    confidence = torch.where(urban_std < 0.05, torch.zeros_like(urban_std), torch.where(urban_std < 0.15, torch.ones_like(urban_std), torch.full_like(urban_std, 2)))
    return {"super_resolved": mean_sr, "uncertainty_std": std_sr, "urban_probabilities": mean_urban, "confidence_classes": confidence.to(torch.uint8)}
