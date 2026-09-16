"""Base model protocol and interface for super-resolution model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class SuperResolutionModel(Protocol):
    """Protocol that all super-resolution model adapters must satisfy."""

    name: str
    scale: int

    def predict(self, lr: torch.Tensor | np.ndarray, **kwargs: Any) -> dict[str, Any]:
        """Run super-resolution prediction on a batch or single image."""
        ...

    def predict_scene(
        self,
        input_path: str | Path,
        output_path: str | Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run tiled georeferenced prediction preserving metadata and updating transform."""
        ...


class BaseSRAdapter(ABC):
    """Abstract base class providing standard utilities for SR adapters."""

    def __init__(self, name: str, scale: int = 4, device: str | None = None) -> None:
        self.name = name
        self.scale = scale
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def predict(self, lr: torch.Tensor | np.ndarray, **kwargs: Any) -> dict[str, Any]:
        pass

    @abstractmethod
    def predict_scene(
        self,
        input_path: str | Path,
        output_path: str | Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        pass
