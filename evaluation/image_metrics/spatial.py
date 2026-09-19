"""
Spatial fidelity metrics.

Scientific conventions
----------------------
* PSNR, SSIM, MAE and RMSE against an HR reference are ONLY computed when a
  genuine high-resolution reference array is supplied.  Passing an upsampled
  LR proxy as the reference is explicitly rejected.
* Gradient energy is a self-referential consistency diagnostic that does NOT
  require an HR reference and measures spatial detail richness.
* All arrays are expected as float32 numpy arrays in (C, H, W) format with
  values in [0, 1] unless otherwise noted.
"""

from __future__ import annotations

import numpy as np

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False

from evaluation.image_metrics.results import SpatialMetrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gradient_energy(array: np.ndarray) -> float:
    """Mean squared gradient magnitude over all bands.

    Parameters
    ----------
    array : np.ndarray, shape (C, H, W)
    """
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3 and array.shape[2] in (3, 4) and array.shape[0] not in (3, 4):
        array = np.transpose(array, (2, 0, 1))
    if array.ndim == 2:
        array = array[np.newaxis]
    total = 0.0
    for band in array:
        gy = np.diff(band, axis=0)
        gx = np.diff(band, axis=1)
        # Align shapes for element-wise operation
        gy_clip = gy[:, :gx.shape[1]]
        gx_clip = gx[:gy.shape[0], :]
        grad_sq = gy_clip ** 2 + gx_clip ** 2
        total += float(np.mean(grad_sq))
    return total / array.shape[0]


def _validate_reference(prediction: np.ndarray, reference: np.ndarray) -> None:
    """Raise ValueError if shapes are incompatible."""
    if prediction.shape != reference.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} != reference shape {reference.shape}. "
            "HR reference must match the SR output exactly.  "
            "Do NOT pass an upsampled LR image as a reference."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_psnr(prediction: np.ndarray, reference: np.ndarray) -> float:
    """PSNR in dB.  Requires genuine HR reference."""
    if not _SKIMAGE_AVAILABLE:
        mse = float(np.mean((prediction - reference) ** 2))
        if mse == 0.0:
            return float("inf")
        return float(10.0 * np.log10(1.0 / mse))
    return float(
        peak_signal_noise_ratio(reference, prediction, data_range=1.0)
    )


def compute_ssim(prediction: np.ndarray, reference: np.ndarray) -> float:
    """SSIM.  Requires genuine HR reference."""
    if not _SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image is required for SSIM computation.")
    # skimage expects (H, W, C) for multichannel
    pred_hwc = np.moveaxis(prediction, 0, -1)
    ref_hwc = np.moveaxis(reference, 0, -1)
    return float(
        structural_similarity(
            ref_hwc,
            pred_hwc,
            channel_axis=2,
            data_range=1.0,
        )
    )


def compute_mae(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Mean absolute error.  Requires genuine HR reference."""
    return float(np.mean(np.abs(prediction - reference)))


def compute_rmse(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Root-mean-squared error.  Requires genuine HR reference."""
    return float(np.sqrt(np.mean((prediction - reference) ** 2)))


def spatial_fidelity(
    prediction: np.ndarray,
    input_array: np.ndarray | None = None,
    bicubic_array: np.ndarray | None = None,
    hr_reference: np.ndarray | None = None,
    hr_reference_available: bool = False,
) -> SpatialMetrics:
    """Compute spatial fidelity metrics.

    Parameters
    ----------
    prediction : np.ndarray, shape (C, H, W)
        PixelSight SR output (~2.5 m-equivalent super-resolved representation).
    input_array : np.ndarray, shape (C, H, W), optional
        Native 10 m input (for gradient energy consistency diagnostic).
    bicubic_array : np.ndarray, shape (C, H, W), optional
        Bicubic interpolation to the same spatial resolution as prediction.
    hr_reference : np.ndarray, shape (C, H, W), optional
        Genuine high-resolution reference.  If None, HR-reference metrics are
        set to None and the result is labelled accordingly.
    hr_reference_available : bool
        Must be True only when hr_reference is a genuine independent HR source
        (e.g. SEN2NAIP).  Setting this to True with an LR upscale proxy is
        scientifically incorrect.

    Returns
    -------
    SpatialMetrics
    """
    prediction = np.asarray(prediction, dtype=np.float32)
    result = SpatialMetrics(hr_reference_available=hr_reference_available)

    # --- HR-reference metrics ---
    if hr_reference_available and hr_reference is not None:
        hr_reference = np.asarray(hr_reference, dtype=np.float32)
        if hr_reference.ndim == 3 and hr_reference.shape[2] in (3, 4) and hr_reference.shape[0] not in (3, 4):
            hr_reference = np.transpose(hr_reference, (2, 0, 1))
        _validate_reference(prediction, hr_reference)
        result.psnr = compute_psnr(prediction, hr_reference)
        result.ssim = compute_ssim(prediction, hr_reference)
        result.mae_hr = compute_mae(prediction, hr_reference)
        result.rmse_hr = compute_rmse(prediction, hr_reference)

        # Spatial correlation with reference
        p_flat = prediction.ravel()
        r_flat = hr_reference.ravel()
        valid = np.isfinite(p_flat) & np.isfinite(r_flat)
        if np.any(valid):
            corr = np.corrcoef(p_flat[valid], r_flat[valid])[0, 1]
            result.spatial_correlation_sr_vs_ref = float(corr) if np.isfinite(corr) else None

        # Edge preservation index (correlation between gradient magnitudes)
        sr_grad = _gradient_magnitude_map(prediction).ravel()
        ref_grad = _gradient_magnitude_map(hr_reference).ravel()
        g_valid = np.isfinite(sr_grad) & np.isfinite(ref_grad)
        if np.any(g_valid):
            g_corr = np.corrcoef(sr_grad[g_valid], ref_grad[g_valid])[0, 1]
            result.edge_preservation_index = float(g_corr) if np.isfinite(g_corr) else None
    else:
        result.psnr = None
        result.ssim = None
        result.mae_hr = None
        result.rmse_hr = None
        result.spatial_correlation_sr_vs_ref = None
        result.edge_preservation_index = None

    # --- Consistency diagnostics (no HR reference required) ---
    result.gradient_energy_sr = _gradient_energy(prediction)
    result.sharpness_score = result.gradient_energy_sr

    if input_array is not None:
        input_array = np.asarray(input_array, dtype=np.float32)
        if input_array.ndim == 3 and input_array.shape[2] in (3, 4) and input_array.shape[0] not in (3, 4):
            input_array = np.transpose(input_array, (2, 0, 1))
        result.gradient_energy_input = _gradient_energy(input_array)

    if bicubic_array is not None:
        bicubic_array = np.asarray(bicubic_array, dtype=np.float32)
        if bicubic_array.ndim == 3 and bicubic_array.shape[2] in (3, 4) and bicubic_array.shape[0] not in (3, 4):
            bicubic_array = np.transpose(bicubic_array, (2, 0, 1))
        result.gradient_energy_bicubic = _gradient_energy(bicubic_array)
        if result.gradient_energy_bicubic and result.gradient_energy_bicubic > 0:
            result.gradient_energy_sr_vs_bicubic_ratio = (
                result.gradient_energy_sr / result.gradient_energy_bicubic
            )
            result.high_frequency_energy_ratio = result.gradient_energy_sr_vs_bicubic_ratio

        # Spatial correlation between SR and bicubic
        p_flat = prediction.ravel()
        b_flat = bicubic_array.ravel()
        valid = np.isfinite(p_flat) & np.isfinite(b_flat)
        if np.any(valid):
            b_corr = np.corrcoef(p_flat[valid], b_flat[valid])[0, 1]
            result.spatial_correlation_sr_vs_bicubic = float(b_corr) if np.isfinite(b_corr) else None

    return result


def _gradient_magnitude_map(array: np.ndarray) -> np.ndarray:
    """Per-pixel mean gradient magnitude across channels."""
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[np.newaxis]
    mags = []
    for band in array:
        gy = np.diff(band, axis=0)
        gx = np.diff(band, axis=1)
        gy_clip = gy[:, :gx.shape[1]]
        gx_clip = gx[:gy.shape[0], :]
        mags.append(np.sqrt(gy_clip**2 + gx_clip**2 + 1e-10))
    return np.mean(mags, axis=0)


def gradient_energy_only(array: np.ndarray) -> float:
    """Convenience wrapper: compute gradient energy for a single array."""
    return _gradient_energy(np.asarray(array, dtype=np.float32))
