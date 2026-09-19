"""
Synthetic Paired Benchmark Generator
====================================
Constructs a controlled synthetic benchmark when real or external HR satellite
references are unavailable.

Methodology:
Known High-Quality Target (Reference)
         │
         ▼
Documented Degradation Model:
- Gaussian Point-Spread Function (PSF) blur
- 4x Spatial Downsampling
- Sensor-like noise injection (SNR controlled)
- Valid data masking
- Reproducible random seed
         │
         ▼
Degraded Low-Resolution Observation (Input)
         │
         ▼
Model Reconstruction (4x Super-Resolution)
         │
         ▼
Quantitative Evaluation against Known Ground Truth Reference

Note: Labeled strictly as "synthetic_hr". Does NOT constitute validation against
an independently observed 2.5 m satellite image.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

try:
    import rasterio
    from rasterio.transform import Affine
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False


@dataclass
class DegradationConfig:
    """Documented degradation configuration for reproducibility."""
    reference_resolution: float = 10.0
    degradation_scale: int = 4
    degraded_resolution: float = 40.0
    blur_kernel: str = "gaussian_psf"
    blur_sigma: float = 1.2
    noise_level: float = 0.005  # additive Gaussian noise std
    random_seed: int = 42
    normalization: str = "min_max_clipped_0_1"
    bands: list[str] = field(default_factory=lambda: ["B02", "B03", "B04", "B08"])
    scientific_disclaimer: str = (
        "This is a controlled synthetic reconstruction benchmark and does not "
        "constitute validation against an independently observed 2.5 m satellite image."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SyntheticBenchmarkGenerator:
    """Generates paired synthetic reference and degraded observation."""

    def __init__(self, config: DegradationConfig | None = None) -> None:
        self.config = config or DegradationConfig()

    def generate_from_array(
        self,
        reference_arr: np.ndarray,
        output_dir: str | Path,
    ) -> tuple[np.ndarray, np.ndarray, DegradationConfig]:
        """Generate degraded input from high-quality reference array.

        Parameters
        ----------
        reference_arr : np.ndarray, shape (C, H, W) or (H, W, C)
            Known high-resolution target.
        output_dir : Path
            Destination for degraded array and metadata.

        Returns
        -------
        reference : np.ndarray, shape (C, H, W)
        degraded : np.ndarray, shape (C, H // scale, W // scale)
        config : DegradationConfig
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ref = np.asarray(reference_arr, dtype=np.float32)
        if ref.ndim == 3 and ref.shape[2] in (3, 4) and ref.shape[0] not in (3, 4):
            ref = np.transpose(ref, (2, 0, 1))
        if ref.ndim == 2:
            ref = ref[np.newaxis]

        # Clip reference to valid range [0, 1]
        ref = np.clip(ref, 0.0, 1.0)
        c, h, w = ref.shape
        scale = self.config.degradation_scale

        np.random.seed(self.config.random_seed)

        degraded = np.zeros((c, h // scale, w // scale), dtype=np.float32)

        for b in range(c):
            # 1. Apply Point-Spread Function (PSF) blur approximation
            blurred = gaussian_filter(ref[b], sigma=self.config.blur_sigma, mode="reflect")
            # 2. Downsample spatially
            downsampled = zoom(blurred, 1.0 / scale, order=3)
            # 3. Add sensor-like noise
            if self.config.noise_level > 0:
                noise = np.random.normal(0, self.config.noise_level, downsampled.shape).astype(np.float32)
                downsampled = downsampled + noise
            degraded[b] = np.clip(downsampled, 0.0, 1.0)

        # Save arrays and configuration
        np.save(output_dir / "reference.npy", ref)
        np.save(output_dir / "degraded_input.npy", degraded)
        with open(output_dir / "degradation_config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        return ref, degraded, self.config

    def generate_from_geotiff(
        self,
        reference_tif_path: str | Path,
        output_dir: str | Path,
    ) -> tuple[Path, Path, DegradationConfig]:
        """Generate GeoTIFF pairs: reference.tif and degraded_input.tif."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ref_tif_out = output_dir / "reference.tif"
        deg_tif_out = output_dir / "degraded_input.tif"

        if not _RASTERIO_AVAILABLE:
            raise ImportError("rasterio is required to generate synthetic GeoTIFF benchmarks.")

        with rasterio.open(reference_tif_path) as src:
            ref_data = src.read().astype(np.float32)
            meta_ref = src.meta.copy()
            crs = src.crs
            transform_ref = src.transform
            res_ref = float(src.res[0])

        # Normalize data to [0, 1] if stored in 10000 scale
        if ref_data.max() > 2.0:
            ref_data = ref_data / 10000.0
        ref_data = np.clip(ref_data, 0.0, 1.0)

        # Update config resolutions
        self.config.reference_resolution = res_ref
        self.config.degraded_resolution = res_ref * self.config.degradation_scale

        ref_clean, degraded_arr, config = self.generate_from_array(ref_data, output_dir)

        # Save reference.tif
        meta_ref.update({
            "dtype": "float32",
            "count": ref_clean.shape[0],
            "height": ref_clean.shape[1],
            "width": ref_clean.shape[2],
        })
        with rasterio.open(ref_tif_out, "w", **meta_ref) as dst:
            dst.write(ref_clean)

        # Calculate transform for degraded input
        deg_scale = float(self.config.degradation_scale)
        transform_deg = transform_ref * Affine.scale(deg_scale, deg_scale)

        meta_deg = meta_ref.copy()
        meta_deg.update({
            "width": degraded_arr.shape[2],
            "height": degraded_arr.shape[1],
            "transform": transform_deg,
        })
        with rasterio.open(deg_tif_out, "w", **meta_deg) as dst:
            dst.write(degraded_arr)

        return ref_tif_out, deg_tif_out, config
