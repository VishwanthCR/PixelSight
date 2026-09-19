"""
Disaster & Bi-Temporal Change Analysis Component
================================================
Implements research-ready before/after change detection on super-resolved imagery.

Scientific Principle:
Clearly separate "change detection" from "damage detection".
Never claim disaster damage without independent ground truth validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.downstream.ndvi import compute_ndvi


@dataclass
class ChangeStatistics:
    """Bi-temporal change detection metrics."""
    mean_absolute_difference: float = 0.0
    max_absolute_difference: float = 0.0
    significant_change_pixels: int = 0
    total_analyzed_pixels: int = 0
    significant_change_fraction: float = 0.0
    affected_area_hectares: float = 0.0
    high_confidence_change_fraction: float = 0.0  # changes outside high-uncertainty zones


@dataclass
class DisasterAnalysisResult:
    """Complete research-ready bi-temporal change analysis."""
    multispectral_change: ChangeStatistics = field(default_factory=ChangeStatistics)
    ndvi_change: ChangeStatistics = field(default_factory=ChangeStatistics)
    pixel_resolution_m: float = 2.5
    change_threshold: float = 0.15
    uncertainty_filtered: bool = False
    scientific_classification: str = "BITEMPORAL_SURFACE_REFLECTANCE_CHANGE_DETECTION"
    disclaimer: str = (
        "This evaluation detects physical surface reflectance and spectral index changes "
        "between two acquisition dates. It does NOT constitute automated structural "
        "damage detection or disaster impact verification without validated ground surveys."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_bitemporal_change(
    before_sr: np.ndarray,
    after_sr: np.ndarray,
    before_uncertainty: np.ndarray | None = None,
    after_uncertainty: np.ndarray | None = None,
    pixel_resolution_m: float = 2.5,
    change_threshold: float = 0.15,
    output_dir: str | Path | None = None,
) -> tuple[DisasterAnalysisResult, np.ndarray]:
    """Analyze change between temporally matched super-resolved imagery.

    Parameters
    ----------
    before_sr, after_sr : np.ndarray, shape (C, H, W)
    before_uncertainty, after_uncertainty : np.ndarray, optional
    pixel_resolution_m : float
    change_threshold : float
    output_dir : Path, optional

    Returns
    -------
    (DisasterAnalysisResult, change_magnitude_map)
    """
    b_sr = np.asarray(before_sr, dtype=np.float32)
    a_sr = np.asarray(after_sr, dtype=np.float32)

    if b_sr.ndim == 3 and b_sr.shape[2] in (3, 4) and b_sr.shape[0] not in (3, 4):
        b_sr = np.transpose(b_sr, (2, 0, 1))
    if a_sr.ndim == 3 and a_sr.shape[2] in (3, 4) and a_sr.shape[0] not in (3, 4):
        a_sr = np.transpose(a_sr, (2, 0, 1))

    # Pixel area in hectares (1 hectare = 10,000 m^2)
    pixel_area_ha = (pixel_resolution_m ** 2) / 10000.0

    # 1. Multi-spectral absolute difference map (mean across bands)
    diff_map = np.mean(np.abs(a_sr - b_sr), axis=0)

    # 2. NDVI difference map
    ndvi_before = compute_ndvi(b_sr)
    ndvi_after = compute_ndvi(a_sr)
    ndvi_diff = np.abs(ndvi_after - ndvi_before)

    valid = np.isfinite(diff_map) & np.isfinite(ndvi_diff)
    total_px = int(np.sum(valid))

    res = DisasterAnalysisResult(
        pixel_resolution_m=pixel_resolution_m,
        change_threshold=change_threshold,
    )

    if total_px == 0:
        return res, diff_map

    # Multi-spectral stats
    sig_change_mask = (diff_map > change_threshold) & valid
    sig_px = int(np.sum(sig_change_mask))
    res.multispectral_change = ChangeStatistics(
        mean_absolute_difference=float(np.mean(diff_map[valid])),
        max_absolute_difference=float(np.max(diff_map[valid])),
        significant_change_pixels=sig_px,
        total_analyzed_pixels=total_px,
        significant_change_fraction=float(sig_px / total_px),
        affected_area_hectares=float(sig_px * pixel_area_ha),
    )

    # NDVI stats
    sig_ndvi_mask = (ndvi_diff > change_threshold) & valid
    sig_ndvi_px = int(np.sum(sig_ndvi_mask))
    res.ndvi_change = ChangeStatistics(
        mean_absolute_difference=float(np.mean(ndvi_diff[valid])),
        max_absolute_difference=float(np.max(ndvi_diff[valid])),
        significant_change_pixels=sig_ndvi_px,
        total_analyzed_pixels=total_px,
        significant_change_fraction=float(sig_ndvi_px / total_px),
        affected_area_hectares=float(sig_ndvi_px * pixel_area_ha),
    )

    # Filter changes by uncertainty consensus if uncertainty maps supplied
    if before_uncertainty is not None and after_uncertainty is not None:
        u_comb = np.maximum(before_uncertainty, after_uncertainty)
        u_thresh = float(np.percentile(u_comb[valid], 75))
        high_conf_mask = sig_change_mask & (u_comb <= u_thresh)
        res.multispectral_change.high_confidence_change_fraction = float(np.sum(high_conf_mask) / total_px)
        res.uncertainty_filtered = True

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        with open(out_p / "disaster_analysis.json", "w") as f:
            json.dump(res.to_dict(), f, indent=2)
        np.save(out_p / "change_difference_map.npy", diff_map)

    return res, diff_map
