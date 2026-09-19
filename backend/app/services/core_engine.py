"""
PixelSight Shared Core Engine
=============================
Consolidated super-resolution and uncertainty engine for Earth Observation data.
Shared by Research, Crop Monitoring, Urban Analysis, and Disaster Management pipelines.

Architecture:
Satellite Input -> Inspection -> Preprocessing -> Tiling -> LDSR-S2 (100 steps)
-> Uncertainty -> Stitching (preserving CRS & GeoTransform)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import Affine

from backend.app.schemas import InspectionResponse
from backend.app.services.raster import inspect_raster, preprocess_raster
from backend.app.services.uncertainty import generate_uncertainty_map
from backend.app.services.visualization import save_rgb_preview

_CACHED_MODEL = None
_CACHED_MODEL_DEVICE = None


class PixelSightEngine:
    """
    Canonical Core Engine interface for Sentinel-2 4x super-resolution and uncertainty.
    Guarantees:
    - 4x spatial enhancement (~2.5 m equivalent representation)
    - 100 diffusion sampling steps via LDSR-S2
    - Preservation of georeferencing, CRS, and B02/B03/B04/B08 multi-spectral bands
    - Completely decoupled from application-specific metrics/interpretations
    """

    SCALE = 4
    PATCH_SIZE = 128
    SAMPLING_STEPS = 100
    OVERLAP = 12
    EXPECTED_BANDS = ("B02", "B03", "B04", "B08")

    @classmethod
    def inspect(cls, input_path: Path, filename: str | None = None) -> InspectionResponse:
        """Inspect input raster for dimensions, CRS, and band compatibility."""
        return inspect_raster(input_path, filename)

    @classmethod
    def preprocess(cls, input_path: Path, output_path: Path) -> tuple[InspectionResponse, list[dict[str, Any]]]:
        """Normalize input raster to float32 reflectance in [0, 1] across standard bands."""
        return preprocess_raster(input_path, output_path)

    @classmethod
    def enhance(
        cls,
        normalized_path: Path,
        output_sr_path: Path,
        preview_path: Path | None = None,
        device: str | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> tuple[dict[str, Any], str]:
        """
        Execute 4x super-resolution using LDSR-S2 with 100 diffusion steps.
        Saves output GeoTIFF preserving CRS and updated geotransform.
        """
        import torch

        chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        test_mode = os.environ.get("PIXELSIGHT_TEST_MODE") == "1"

        output_sr_path.parent.mkdir(parents=True, exist_ok=True)

        if test_mode:
            # Fast test/mock mode for automated CI/unit tests without downloading 1GB weights
            with rasterio.open(normalized_path) as src:
                profile = src.profile.copy()
                data = src.read().astype(np.float32)
                c, h, w = data.shape
                out_h, out_w = h * cls.SCALE, w * cls.SCALE
                resampled = np.zeros((c, out_h, out_w), dtype=np.float32)
                for b in range(c):
                    img = Image.fromarray(np.clip(data[b] * 255.0, 0, 255).astype(np.uint8), mode="L")
                    resized = img.resize((out_w, out_h), Image.BICUBIC)
                    resampled[b] = np.asarray(resized, dtype=np.float32) / 255.0

                new_transform = src.transform * Affine.scale(1.0 / cls.SCALE, 1.0 / cls.SCALE)
                profile.update(
                    height=out_h,
                    width=out_w,
                    transform=new_transform,
                    dtype="float32",
                    compress="deflate",
                )
                with rasterio.open(output_sr_path, "w", **profile) as dst:
                    dst.write(resampled)
                    for idx, name in enumerate(cls.EXPECTED_BANDS[:c], start=1):
                        dst.set_band_description(idx, name)

            if preview_path:
                save_rgb_preview(output_sr_path, preview_path)

            if progress_callback:
                progress_callback(90.0)

            return {
                "model": "LDSR-S2 (TEST MODE)",
                "scale": cls.SCALE,
                "sampling_steps": cls.SAMPLING_STEPS,
                "device": chosen_device,
                "is_test_mode": True,
            }, chosen_device

        # Standard real inference using LDSR-S2
        import importlib
        inference = importlib.import_module("scripts.plan2.infer_ldsr_s2")

        with rasterio.open(normalized_path) as src:
            profile = src.profile.copy()
            bounds = src.bounds
            image = np.moveaxis(src.read(), 0, -1)

        global _CACHED_MODEL, _CACHED_MODEL_DEVICE
        if _CACHED_MODEL is None or _CACHED_MODEL_DEVICE != chosen_device:
            _CACHED_MODEL = inference.load_model(chosen_device)
            _CACHED_MODEL_DEVICE = chosen_device
        model = _CACHED_MODEL

        output = inference.run_inference(model, image, chosen_device)
        inference.save_output(output, output_sr_path, profile)
        inference.verify_output(output_sr_path, bounds)

        if preview_path:
            save_rgb_preview(output_sr_path, preview_path)

        if progress_callback:
            progress_callback(90.0)

        return {
            "model": "LDSR-S2",
            "scale": cls.SCALE,
            "sampling_steps": cls.SAMPLING_STEPS,
            "patch_size": cls.PATCH_SIZE,
            "overlap": cls.OVERLAP,
            "device": chosen_device,
            "is_test_mode": False,
        }, chosen_device

    @classmethod
    def uncertainty(
        cls,
        normalized_path: Path,
        sr_path: Path,
        output_tif: Path,
        output_png: Path,
    ) -> dict[str, Any]:
        """
        Generate pixel-level super-resolution uncertainty map.
        Returns uncertainty statistics dictionary.
        """
        return generate_uncertainty_map(
            normalized_path,
            sr_path,
            output_tif,
            output_png,
        )
