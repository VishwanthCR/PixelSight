"""
NDVI downstream evaluation.

Scientific conventions
----------------------
* NDVI = (B08 - B04) / (B08 + B04)
* Band indices: B04 (Red) = index 2, B08 (NIR) = index 3 in the 4-band stack.
* B08 must come from genuine Sentinel-2 data.
  Do NOT synthesise NIR from RGB channels.
* Native 10 m NDVI is used as the reference baseline for inter-method comparisons.
  This is a consistency diagnostic, NOT a reconstruction accuracy metric.
* Genuine HR-reference NDVI comparisons require an independent HR source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

try:
    from scipy.ndimage import zoom as _zoom
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class NDVIResult:
    """NDVI consistency comparison across three representations."""

    # Per-representation statistics
    native_mean: float | None = None
    native_std: float | None = None
    native_min: float | None = None
    native_max: float | None = None

    bicubic_mean: float | None = None
    bicubic_std: float | None = None
    bicubic_min: float | None = None
    bicubic_max: float | None = None

    sr_mean: float | None = None
    sr_std: float | None = None
    sr_min: float | None = None
    sr_max: float | None = None

    # Pairwise consistency (bicubic and SR vs native as reference baseline)
    bicubic_mae_vs_native: float | None = None
    bicubic_rmse_vs_native: float | None = None
    sr_mae_vs_native: float | None = None
    sr_rmse_vs_native: float | None = None

    # Improvement metrics (positive = SR is better than bicubic)
    mae_improvement_pct: float | None = None   # (bic_mae - sr_mae) / bic_mae * 100
    rmse_improvement_pct: float | None = None  # (bic_rmse - sr_rmse) / bic_rmse * 100

    # Distribution shift
    bicubic_distribution_shift: float | None = None  # |bicubic_mean - native_mean|
    sr_distribution_shift: float | None = None       # |sr_mean - native_mean|

    # Metadata
    red_band_idx: int = 2   # B04
    nir_band_idx: int = 3   # B08
    reference_is_genuine_hr: bool = False
    diagnostic_note: str = (
        "Native 10 m NDVI is used as the reference baseline for inter-method "
        "consistency comparisons.  These are NOT reconstruction accuracy metrics "
        "against a genuine HR reference.  SR improvements are relative to the "
        "native representation and may reflect spectral smoothing effects."
    )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_ndvi(
    image: np.ndarray,
    red_idx: int = 2,
    nir_idx: int = 3,
) -> np.ndarray:
    """Compute NDVI from a (C, H, W) float32 array.

    Parameters
    ----------
    image : np.ndarray, shape (C, H, W), values approximately in [0, 1]
    red_idx : int
        Index of B04 (Red).
    nir_idx : int
        Index of B08 (NIR).  Must be genuine Sentinel-2 NIR.

    Returns
    -------
    ndvi : np.ndarray, shape (H, W), values in [-1, 1]
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3 and image.shape[2] in (3, 4) and image.shape[0] not in (3, 4):
        image = np.transpose(image, (2, 0, 1))
    red = image[red_idx]
    nir = image[nir_idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(
            (nir + red) > 1e-8,
            (nir - red) / (nir + red),
            0.0,
        )
    return ndvi.astype(np.float32)


def _upscale_if_needed(
    arr: np.ndarray,
    target_shape: tuple[int, int],
    order: int = 3,
) -> np.ndarray:
    """Bicubic upscale to match target shape if shapes differ."""
    if arr.shape == target_shape:
        return arr
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy is required to upscale NDVI arrays for comparison.")
    scale_y = target_shape[0] / arr.shape[0]
    scale_x = target_shape[1] / arr.shape[1]
    return _zoom(arr, (scale_y, scale_x), order=order).astype(np.float32)


def _ndvi_stats(ndvi: np.ndarray) -> dict[str, float]:
    finite = ndvi[np.isfinite(ndvi)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ndvi_comparison(
    native: np.ndarray,
    sr: np.ndarray,
    bicubic: np.ndarray | None = None,
    red_idx: int = 2,
    nir_idx: int = 3,
    reference_is_genuine_hr: bool = False,
) -> NDVIResult:
    """Compare NDVI across native, SR and optionally bicubic representations.

    Parameters
    ----------
    native : np.ndarray, shape (C, H_n, W_n)
        Native 10 m input.  Used as the reference baseline.
    sr : np.ndarray, shape (C, H_s, W_s)
        PixelSight SR output (~2.5 m equivalent).
    bicubic : np.ndarray, shape (C, H_s, W_s), optional
        Bicubic interpolation to the same resolution as sr.
    red_idx, nir_idx : int
        Band indices for B04 and B08.
    reference_is_genuine_hr : bool
        True only if native is a genuine independent HR reference.
        For standard Sentinel-2 10 m input, leave False.

    Returns
    -------
    NDVIResult
    """
    result = NDVIResult(
        red_band_idx=red_idx,
        nir_band_idx=nir_idx,
        reference_is_genuine_hr=reference_is_genuine_hr,
    )

    ndvi_native = compute_ndvi(native, red_idx, nir_idx)
    ndvi_sr = compute_ndvi(sr, red_idx, nir_idx)

    stats_native = _ndvi_stats(ndvi_native)
    result.native_mean = stats_native["mean"]
    result.native_std = stats_native["std"]
    result.native_min = stats_native["min"]
    result.native_max = stats_native["max"]

    stats_sr = _ndvi_stats(ndvi_sr)
    result.sr_mean = stats_sr["mean"]
    result.sr_std = stats_sr["std"]
    result.sr_min = stats_sr["min"]
    result.sr_max = stats_sr["max"]

    # Upscale native NDVI to SR resolution for pixel-level comparison
    ndvi_native_up = _upscale_if_needed(ndvi_native, ndvi_sr.shape)

    result.sr_mae_vs_native = float(np.mean(np.abs(ndvi_sr - ndvi_native_up)))
    result.sr_rmse_vs_native = float(
        np.sqrt(np.mean((ndvi_sr - ndvi_native_up) ** 2))
    )
    result.sr_distribution_shift = abs(result.sr_mean - result.native_mean)  # type: ignore[operator]

    if bicubic is not None:
        ndvi_bicubic = compute_ndvi(bicubic, red_idx, nir_idx)
        stats_bic = _ndvi_stats(ndvi_bicubic)
        result.bicubic_mean = stats_bic["mean"]
        result.bicubic_std = stats_bic["std"]
        result.bicubic_min = stats_bic["min"]
        result.bicubic_max = stats_bic["max"]

        ndvi_bic_up = _upscale_if_needed(ndvi_bicubic, ndvi_sr.shape)
        ndvi_native_bic = _upscale_if_needed(ndvi_native, ndvi_bicubic.shape)

        result.bicubic_mae_vs_native = float(
            np.mean(np.abs(ndvi_bic_up - ndvi_native_up))
        )
        result.bicubic_rmse_vs_native = float(
            np.sqrt(np.mean((ndvi_bic_up - ndvi_native_up) ** 2))
        )
        result.bicubic_distribution_shift = abs(result.bicubic_mean - result.native_mean)  # type: ignore[operator]

        # Improvement metrics
        if result.bicubic_mae_vs_native and result.bicubic_mae_vs_native > 0:
            result.mae_improvement_pct = float(
                (result.bicubic_mae_vs_native - result.sr_mae_vs_native)  # type: ignore[operator]
                / result.bicubic_mae_vs_native * 100.0
            )
        if result.bicubic_rmse_vs_native and result.bicubic_rmse_vs_native > 0:
            result.rmse_improvement_pct = float(
                (result.bicubic_rmse_vs_native - result.sr_rmse_vs_native)  # type: ignore[operator]
                / result.bicubic_rmse_vs_native * 100.0
            )

    return result
