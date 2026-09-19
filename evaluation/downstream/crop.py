"""
Crop / Agriculture Downstream Evaluation Component
==================================================
Evaluates vegetation vitality and crop monitoring fidelity using NDVI:
NDVI = (B08 - B04) / (B08 + B04)

Features:
- Multi-representation comparison: Native, Bicubic, PixelSight, Reference (when available).
- Cropland mask filtering (e.g. ESA WorldCover class 40 = Cropland).
- Distribution statistics: mean, std, min, max, MAE, RMSE.
- Spatial delta analysis.
- Generates crop_analysis.json and crop_ndvi_comparison.png.

Scientific Principle:
Do NOT claim PixelSight improves agriculture unless the experiment demonstrates it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import zoom

from evaluation.downstream.ndvi import compute_ndvi


CROPLAND_CLASS_VAL = 40  # ESA WorldCover cropland class


@dataclass
class CropRepresentationMetrics:
    """Crop NDVI metrics for a single representation."""
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    mae_vs_native: float | None = None
    rmse_vs_native: float | None = None
    mae_vs_reference: float | None = None
    rmse_vs_reference: float | None = None
    cropland_pixel_count: int = 0


@dataclass
class CropAnalysisResult:
    """Aggregated crop and agricultural monitoring report."""
    native: CropRepresentationMetrics = field(default_factory=CropRepresentationMetrics)
    bicubic: CropRepresentationMetrics = field(default_factory=CropRepresentationMetrics)
    pixelsight: CropRepresentationMetrics = field(default_factory=CropRepresentationMetrics)
    reference: CropRepresentationMetrics | None = None

    cropland_masked: bool = False
    reference_available: bool = False
    mae_reduction_vs_bicubic_pct: float | None = None
    scientific_note: str = (
        "Agricultural evaluation measures NDVI fidelity across cropland parcels. "
        "Native 10 m NDVI represents the physical observed baseline. "
        "Higher spatial resolution resolves parcel boundaries, but does NOT "
        "automatically indicate improved crop yield estimation without in-situ ground truth."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_crop_monitoring(
    native: np.ndarray,
    sr: np.ndarray,
    bicubic: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    cropland_mask: np.ndarray | None = None,
    output_dir: str | Path | None = None,
) -> CropAnalysisResult:
    """Perform crop monitoring and NDVI parcel analysis.

    Parameters
    ----------
    native : np.ndarray, shape (4, H_n, W_n) or (H_n, W_n, 4)
    sr : np.ndarray, shape (4, H_s, W_s) or (H_s, W_s, 4)
    bicubic : np.ndarray, optional
    reference : np.ndarray, optional
    cropland_mask : np.ndarray, boolean or integer mask
    output_dir : Path, optional
    """
    result = CropAnalysisResult()

    # Compute NDVI arrays
    ndvi_nat = compute_ndvi(native)
    ndvi_sr = compute_ndvi(sr)

    if bicubic is not None:
        ndvi_bic = compute_ndvi(bicubic)
    else:
        scale = ndvi_sr.shape[0] / ndvi_nat.shape[0]
        ndvi_bic = zoom(ndvi_nat, scale, order=3)

    ndvi_ref = compute_ndvi(reference) if reference is not None else None
    result.reference_available = (ndvi_ref is not None)

    # Align native to SR resolution for pixel-wise comparison
    scale_y = ndvi_sr.shape[0] / ndvi_nat.shape[0]
    scale_x = ndvi_sr.shape[1] / ndvi_nat.shape[1]
    if scale_y != 1.0 or scale_x != 1.0:
        ndvi_nat_up = zoom(ndvi_nat, (scale_y, scale_x), order=1)
    else:
        ndvi_nat_up = ndvi_nat

    # Build crop mask
    h, w = ndvi_sr.shape
    if cropland_mask is not None:
        mask = np.asarray(cropland_mask)
        if mask.shape != (h, w):
            m_scale_y = h / mask.shape[0]
            m_scale_x = w / mask.shape[1]
            mask = zoom(mask, (m_scale_y, m_scale_x), order=0)
        # Check if boolean or class value
        if mask.dtype == bool:
            crop_bool = mask
        else:
            crop_bool = (mask == CROPLAND_CLASS_VAL)
        result.cropland_masked = bool(np.any(crop_bool))
    else:
        crop_bool = np.ones((h, w), dtype=bool)
        result.cropland_masked = False

    # Valid finite mask
    valid = crop_bool & np.isfinite(ndvi_nat_up) & np.isfinite(ndvi_sr) & np.isfinite(ndvi_bic)
    if ndvi_ref is not None:
        valid &= np.isfinite(ndvi_ref)

    if not np.any(valid):
        return result

    # Population helper
    def _populate(arr: np.ndarray, base_nat: np.ndarray, base_ref: np.ndarray | None) -> CropRepresentationMetrics:
        sub = arr[valid]
        m = CropRepresentationMetrics(
            mean=float(np.mean(sub)),
            std=float(np.std(sub)),
            min=float(np.min(sub)),
            max=float(np.max(sub)),
            cropland_pixel_count=int(np.sum(valid)),
            mae_vs_native=float(np.mean(np.abs(arr[valid] - base_nat[valid]))),
            rmse_vs_native=float(np.sqrt(np.mean((arr[valid] - base_nat[valid])**2))),
        )
        if base_ref is not None:
            m.mae_vs_reference = float(np.mean(np.abs(arr[valid] - base_ref[valid])))
            m.rmse_vs_reference = float(np.sqrt(np.mean((arr[valid] - base_ref[valid])**2)))
        return m

    result.native = _populate(ndvi_nat_up, ndvi_nat_up, ndvi_ref)
    result.bicubic = _populate(ndvi_bic, ndvi_nat_up, ndvi_ref)
    result.pixelsight = _populate(ndvi_sr, ndvi_nat_up, ndvi_ref)

    if ndvi_ref is not None:
        result.reference = _populate(ndvi_ref, ndvi_nat_up, ndvi_ref)
        if result.bicubic.mae_vs_reference and result.pixelsight.mae_vs_reference:
            bic_err = result.bicubic.mae_vs_reference
            sr_err = result.pixelsight.mae_vs_reference
            result.mae_reduction_vs_bicubic_pct = float(((bic_err - sr_err) / bic_err) * 100.0)

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        with open(out_p / "crop_analysis.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    return result
