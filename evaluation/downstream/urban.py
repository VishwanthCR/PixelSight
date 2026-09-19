"""
Urban spatial analysis.

Scientific conventions
----------------------
* Gradient energy measures spatial detail richness.
* Higher gradient energy in the SR vs bicubic result indicates more
  reconstructed fine-scale structure.
* This does NOT by itself establish improved detection accuracy.
* No claims about hallucination or artificial detail are made.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class UrbanSpatialResult:
    """Urban spatial detail analysis across three representations."""

    gradient_energy_native: float | None = None
    gradient_energy_bicubic: float | None = None
    gradient_energy_sr: float | None = None
    sr_vs_bicubic_ratio: float | None = None
    sr_vs_bicubic_pct: float | None = None    # (sr - bic) / bic * 100
    sr_vs_native_ratio: float | None = None
    bicubic_vs_native_ratio: float | None = None

    diagnostic_note: str = (
        "Gradient energy measures spatial detail richness. "
        "LDSR SR having higher gradient energy than bicubic indicates more "
        "reconstructed fine-scale structure.  It does NOT establish improved "
        "building-detection or land-cover classification accuracy.  "
        "Validation requires geographically separated paired data."
    )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def gradient_energy(array: np.ndarray) -> float:
    """Mean squared gradient magnitude over all bands.

    Parameters
    ----------
    array : np.ndarray, shape (C, H, W) or (H, W)
    """
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[np.newaxis]
    total = 0.0
    for band in array:
        gy = np.diff(band, axis=0)
        gx = np.diff(band, axis=1)
        # Match shapes for element-wise addition
        h = min(gy.shape[0], gx.shape[0])
        w = min(gy.shape[1], gx.shape[1])
        grad_sq = gy[:h, :w] ** 2 + gx[:h, :w] ** 2
        total += float(np.mean(grad_sq))
    return total / array.shape[0]


def urban_spatial_analysis(
    native: np.ndarray,
    sr: np.ndarray,
    bicubic: np.ndarray | None = None,
) -> UrbanSpatialResult:
    """Compute urban spatial detail metrics across representations.

    Parameters
    ----------
    native : np.ndarray, shape (C, H_n, W_n)
        Native 10 m input.
    sr : np.ndarray, shape (C, H_s, W_s)
        PixelSight SR output.
    bicubic : np.ndarray, shape (C, H_s, W_s), optional
        Bicubic interpolation to the same resolution as sr.

    Returns
    -------
    UrbanSpatialResult
    """
    result = UrbanSpatialResult()

    result.gradient_energy_native = gradient_energy(native)
    result.gradient_energy_sr = gradient_energy(sr)

    if result.gradient_energy_native and result.gradient_energy_native > 0:
        result.sr_vs_native_ratio = result.gradient_energy_sr / result.gradient_energy_native

    if bicubic is not None:
        result.gradient_energy_bicubic = gradient_energy(bicubic)
        if result.gradient_energy_bicubic and result.gradient_energy_bicubic > 0:
            result.sr_vs_bicubic_ratio = (
                result.gradient_energy_sr / result.gradient_energy_bicubic
            )
            result.sr_vs_bicubic_pct = (
                (result.gradient_energy_sr - result.gradient_energy_bicubic)
                / result.gradient_energy_bicubic * 100.0
            )
        if result.gradient_energy_native and result.gradient_energy_native > 0:
            result.bicubic_vs_native_ratio = (
                result.gradient_energy_bicubic / result.gradient_energy_native  # type: ignore[operator]
            )

    return result
