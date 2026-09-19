"""
Spectral fidelity metrics.

Scientific conventions
----------------------
* SAM, per-band MAE/RMSE and NDVI vs HR reference are ONLY computed when a
  genuine independent HR reference is supplied.
* NDVI comparisons between native, bicubic and SR are consistency diagnostics
  (native 10 m NDVI is used as the reference baseline) and are labeled as such.
* Band order follows the Sentinel-2 convention: B02, B03, B04, B08
  (indices 0, 1, 2, 3 in the 4-band stack).
* Do NOT compute NIR from RGB channels.
  The B08 band must come from genuine Sentinel-2 data.
"""

from __future__ import annotations

import numpy as np

from evaluation.image_metrics.results import SpectralMetrics

# Default Sentinel-2 band names for the 4-band stack
S2_BAND_NAMES = ["B02", "B03", "B04", "B08"]

# Band indices for NDVI in the 4-band stack
_RED_IDX = 2   # B04
_NIR_IDX = 3   # B08


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sam_degrees(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Spectral Angle Mapper in degrees (mean over valid pixels).

    Parameters
    ----------
    prediction, reference : np.ndarray, shape (C, H, W)
    """
    pred_flat = np.moveaxis(prediction, 0, -1).reshape(-1, prediction.shape[0])
    ref_flat = np.moveaxis(reference, 0, -1).reshape(-1, reference.shape[0])
    denom = np.linalg.norm(pred_flat, axis=1) * np.linalg.norm(ref_flat, axis=1)
    valid = denom > 1e-8
    if not np.any(valid):
        return float("nan")
    cosine = np.sum(pred_flat[valid] * ref_flat[valid], axis=1) / denom[valid]
    angles_deg = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(np.mean(angles_deg))


def _per_band_mae(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    mae = {}
    for i in range(prediction.shape[0]):
        name = S2_BAND_NAMES[i] if i < len(S2_BAND_NAMES) else f"band_{i}"
        mae[name] = float(np.mean(np.abs(prediction[i] - reference[i])))
    return mae


def _per_band_rmse(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    rmse = {}
    for i in range(prediction.shape[0]):
        name = S2_BAND_NAMES[i] if i < len(S2_BAND_NAMES) else f"band_{i}"
        rmse[name] = float(np.sqrt(np.mean((prediction[i] - reference[i]) ** 2)))
    return rmse


def _per_band_correlation(
    prediction: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    corr = {}
    for i in range(prediction.shape[0]):
        name = S2_BAND_NAMES[i] if i < len(S2_BAND_NAMES) else f"band_{i}"
        p = prediction[i].ravel()
        r = reference[i].ravel()
        if len(p) < 2:
            corr[name] = float("nan")
        else:
            corr[name] = float(np.corrcoef(p, r)[0, 1])
    return corr


def _per_band_psnr(
    prediction: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    psnr = {}
    for i in range(prediction.shape[0]):
        name = S2_BAND_NAMES[i] if i < len(S2_BAND_NAMES) else f"band_{i}"
        mse = float(np.mean((prediction[i] - reference[i]) ** 2))
        if mse == 0.0:
            psnr[name] = float("inf")
        else:
            psnr[name] = float(10.0 * np.log10(1.0 / mse))
    return psnr


# ---------------------------------------------------------------------------
# NDVI utilities
# ---------------------------------------------------------------------------

def compute_ndvi(image: np.ndarray, red_idx: int = _RED_IDX, nir_idx: int = _NIR_IDX) -> np.ndarray:
    """Compute NDVI from a (C, H, W) image.

    Parameters
    ----------
    image : np.ndarray, shape (C, H, W), values in [0, 1]
    red_idx : int
        Band index for B04 (red). Default = 2.
    nir_idx : int
        Band index for B08 (NIR). Default = 3.

    Returns
    -------
    ndvi : np.ndarray, shape (H, W), values in [-1, 1]

    Note
    ----
    B08 must come from genuine Sentinel-2 data.
    Do NOT synthesise NIR from RGB channels.
    """
    image = np.asarray(image, dtype=np.float32)
    red = image[red_idx]
    nir = image[nir_idx]
    denom = nir + red
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(denom > 1e-8, (nir - red) / denom, 0.0)
    return ndvi.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def spectral_fidelity(
    prediction: np.ndarray,
    native_array: np.ndarray | None = None,
    bicubic_array: np.ndarray | None = None,
    hr_reference: np.ndarray | None = None,
    hr_reference_available: bool = False,
    band_names: list[str] | None = None,
) -> SpectralMetrics:
    """Compute spectral fidelity metrics.

    Parameters
    ----------
    prediction : np.ndarray, shape (C, H, W)
        PixelSight SR output.
    native_array : np.ndarray, shape (C, H', W'), optional
        Native 10 m input.  Used for NDVI consistency baseline.
    bicubic_array : np.ndarray, shape (C, H, W), optional
        Bicubic interpolation to the same resolution as prediction.
    hr_reference : np.ndarray, shape (C, H, W), optional
        Genuine independent HR reference.
    hr_reference_available : bool
        True ONLY for genuine independent HR references.
    band_names : list[str], optional
        Override default S2 band names.

    Returns
    -------
    SpectralMetrics
    """
    global S2_BAND_NAMES
    if band_names:
        S2_BAND_NAMES = band_names

    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.ndim == 3 and prediction.shape[2] in (3, 4) and prediction.shape[0] not in (3, 4):
        prediction = np.transpose(prediction, (2, 0, 1))
    if native_array is not None:
        native_array = np.asarray(native_array, dtype=np.float32)
        if native_array.ndim == 3 and native_array.shape[2] in (3, 4) and native_array.shape[0] not in (3, 4):
            native_array = np.transpose(native_array, (2, 0, 1))
    if bicubic_array is not None:
        bicubic_array = np.asarray(bicubic_array, dtype=np.float32)
        if bicubic_array.ndim == 3 and bicubic_array.shape[2] in (3, 4) and bicubic_array.shape[0] not in (3, 4):
            bicubic_array = np.transpose(bicubic_array, (2, 0, 1))
    result = SpectralMetrics(hr_reference_available=hr_reference_available)

    # --- HR-reference spectral metrics ---
    if hr_reference_available and hr_reference is not None:
        hr_reference = np.asarray(hr_reference, dtype=np.float32)
        if prediction.shape != hr_reference.shape:
            result.sam_degrees = None
        else:
            result.sam_degrees = _sam_degrees(prediction, hr_reference)
            result.per_band_mae_hr = _per_band_mae(prediction, hr_reference)
            result.per_band_rmse_hr = _per_band_rmse(prediction, hr_reference)
            result.per_band_correlation_hr = _per_band_correlation(prediction, hr_reference)
            result.per_band_psnr_hr = _per_band_psnr(prediction, hr_reference)

            # NDVI vs genuine/synthetic HR reference
            ndvi_sr = compute_ndvi(prediction)
            ndvi_hr = compute_ndvi(hr_reference)
            result.ndvi_mae_vs_hr = float(np.mean(np.abs(ndvi_sr - ndvi_hr)))
            result.ndvi_rmse_vs_hr = float(np.sqrt(np.mean((ndvi_sr - ndvi_hr) ** 2)))
            result.ndvi_distribution_shift_hr = float(abs(np.mean(ndvi_sr) - np.mean(ndvi_hr)))
            result.ndvi_mae_vs_native = result.ndvi_mae_vs_hr
            result.ndvi_rmse_vs_native = result.ndvi_rmse_vs_hr

    # --- NDVI consistency diagnostics (native as baseline) ---
    ndvi_sr = compute_ndvi(prediction)
    result.ndvi_mean_sr = float(np.mean(ndvi_sr))
    result.ndvi_std_sr = float(np.std(ndvi_sr))

    if native_array is not None:
        native_array = np.asarray(native_array, dtype=np.float32)
        # Native NDVI — may be different resolution, compute separately
        ndvi_native = compute_ndvi(native_array)
        result.ndvi_mean_native = float(np.mean(ndvi_native))
        result.ndvi_std_native = float(np.std(ndvi_native))

        # Compare SR NDVI vs native NDVI (upscale native to match SR if needed)
        if ndvi_native.shape != ndvi_sr.shape:
            from scipy.ndimage import zoom
            scale = ndvi_sr.shape[0] / ndvi_native.shape[0]
            ndvi_native_up = zoom(ndvi_native, scale, order=3).astype(np.float32)
        else:
            ndvi_native_up = ndvi_native

        result.ndvi_mae_sr_vs_native = float(np.mean(np.abs(ndvi_sr - ndvi_native_up)))
        result.ndvi_rmse_sr_vs_native = float(
            np.sqrt(np.mean((ndvi_sr - ndvi_native_up) ** 2))
        )

    if bicubic_array is not None:
        bicubic_array = np.asarray(bicubic_array, dtype=np.float32)
        ndvi_bicubic = compute_ndvi(bicubic_array)
        result.ndvi_mean_bicubic = float(np.mean(ndvi_bicubic))
        result.ndvi_std_bicubic = float(np.std(ndvi_bicubic))

        if native_array is not None and ndvi_bicubic.shape != ndvi_native_up.shape:  # type: ignore[name-defined]
            from scipy.ndimage import zoom
            scale = ndvi_bicubic.shape[0] / ndvi_native.shape[0]  # type: ignore[name-defined]
            ndvi_native_up_b = zoom(ndvi_native, scale, order=3).astype(np.float32)  # type: ignore[name-defined]
        elif native_array is not None:
            ndvi_native_up_b = ndvi_native_up  # type: ignore[name-defined]

        if native_array is not None:
            result.ndvi_mae_bicubic_vs_native = float(
                np.mean(np.abs(ndvi_bicubic - ndvi_native_up_b))
            )
            result.ndvi_rmse_bicubic_vs_native = float(
                np.sqrt(np.mean((ndvi_bicubic - ndvi_native_up_b) ** 2))
            )

    return result
