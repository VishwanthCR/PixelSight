"""LDSR-S2 model adapter conforming to the canonical framework interface."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch

from .base import BaseSRAdapter


class LDSRS2Adapter(BaseSRAdapter):
    """Production LDSR-S2 4x super-resolution adapter."""

    def __init__(
        self,
        device: str | None = None,
        scale: int = 4,
        sampling_steps: int = 100,
        checkpoint: str | None = None,
    ) -> None:
        super().__init__(name="LDSR-S2", scale=scale, device=device)
        self.sampling_steps = sampling_steps
        self.checkpoint = checkpoint
        self._model = None

    def _get_inference_module(self) -> Any:
        return importlib.import_module("scripts.plan2.infer_ldsr_s2")

    def load_model(self) -> Any:
        if self._model is None:
            module = self._get_inference_module()
            self._model = module.load_model(self.device)
        return self._model

    def predict(self, lr: torch.Tensor | np.ndarray, **kwargs: Any) -> dict[str, Any]:
        """Run single-tile or batched prediction through LDSR-S2."""
        model = self.load_model()
        module = self._get_inference_module()

        if isinstance(lr, torch.Tensor):
            array = lr.detach().cpu().numpy()
            if array.ndim == 4:
                array = array[0]
            if array.shape[0] == 4 and array.shape[-1] != 4:
                array = np.moveaxis(array, 0, -1)
        else:
            array = np.asarray(lr)

        output = module.infer_tile(model, array, self.device)
        return {
            "super_resolved": output,
            "scale": self.scale,
            "model": self.name,
        }

    def predict_scene(
        self,
        input_path: str | Path,
        output_path: str | Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute full tiled scene inference with georeferencing and boundary blending."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        module = self._get_inference_module()
        model = self.load_model()

        with rasterio.open(input_path) as src:
            profile = src.profile.copy()
            bounds = src.bounds
            image = np.moveaxis(src.read(), 0, -1)

        output = module.run_inference(model, image, self.device)
        module.save_output(output, output_path, profile)
        module.verify_output(output_path, bounds)

        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "scale": self.scale,
            "device": self.device,
            "shape": list(output.shape),
            "model": self.name,
        }
