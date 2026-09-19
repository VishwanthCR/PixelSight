"""
Frequency-Domain Resolution & Modulation Transfer Function (MTF) Analysis
========================================================================
Validates effective physical spatial resolving power and detects high-frequency
hallucinations in diffusion-based satellite super-resolution.

Scientific Foundation:
----------------------
In optical remote sensing, a 2.5 m pixel grid does not guarantee 2.5 m optical resolution.
This module performs rigorous Fourier-domain diagnostic analyses:
1. Radial Power Spectral Density (RPSD): Azimuthally integrated frequency power profile.
2. Modulation Transfer Function (MTF50): Spatial frequency where contrast drops to 50%.
3. Fourier Ring Correlation (FRC): Measures cross-spectral consistency across independent
   stochastic diffusion seeds, locating the true spatial frequency cutoff (f_cutoff).
4. Effective Ground Sampling Distance (GSD_eff): True physical resolving power in meters.
5. High-Frequency Hallucination Index (HFHI): Ratio of anomalous generative high frequencies
   exceeding the physical optical modulation transfer envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class FrequencyResolutionMetrics:
    """Frequency-domain physical resolution and MTF diagnostics."""

    # Nominal pixel ground sampling distance
    nominal_gsd_meters: float = 2.5

    # Estimated effective spatial resolution under experimental protocol (from FRC cutoff in meters)
    # Never equate 2.5 m output pixel grid spacing with 2.5 m physical resolving power.
    effective_gsd_meters: Optional[float] = None

    # MTF50: Spatial frequency (cycles/pixel) where relative MTF = 0.50
    mtf50_frequency: Optional[float] = None

    # Fourier Ring Correlation (FRC) cutoff spatial frequency (cycles/pixel) at threshold 0.143 (1/2-bit)
    frc_cutoff_frequency: Optional[float] = None

    # High-Frequency Energy Ratio / Discrepancy Index vs smoothed bicubic baseline
    high_frequency_hallucination_index: float = 0.0

    # RPSD slope (roll-off exponent in log-log space, natural satellite imagery typically ~ -2.5 to -3.0)
    spectral_rolloff_slope: Optional[float] = None

    # Nyquist frequency of the 2.5 m grid (0.50 cycles/pixel) and input 10 m grid (0.125 cycles/pixel)
    nyquist_sr: float = 0.50
    nyquist_native_equivalent: float = 0.125

    protocol_disclaimer: str = (
        "Effective resolution is estimated under the experimental protocol using FRC cross-correlation "
        "across stochastic diffusion realizations. It does not establish sensor optical physical resolution."
    )
    diagnostic_summary: str = ""


def compute_radial_power_spectrum(image_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 1D azimuthally averaged Radial Power Spectral Density (RPSD).

    Parameters
    ----------
    image_2d : np.ndarray, shape (H, W)

    Returns
    -------
    radii : np.ndarray
        Spatial frequencies in cycles/pixel, normalized to [0, 0.5].
    radial_psd : np.ndarray
        Azimuthally averaged power spectrum.
    """
    arr = np.asarray(image_2d, dtype=np.float32)
    H, W = arr.shape
    # Windowing with Hanning window to prevent Fourier spectral leakage
    win = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    arr_windowed = (arr - np.mean(arr)) * win

    # 2D FFT
    F = np.fft.fftshift(np.fft.fft2(arr_windowed))
    psd2d = np.abs(F) ** 2 / (H * W)

    # Compute distance from center
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[-cy : H - cy, -cx : W - cx]
    r = np.sqrt(x * x + y * y)

    # Bin radii
    r_int = r.astype(np.int32)
    max_r = min(cy, cx)
    tbin = np.bincount(r_int.ravel(), psd2d.ravel())
    nr = np.bincount(r_int.ravel())

    # Avoid divide by zero
    valid = (nr > 0) & (np.arange(len(nr)) <= max_r)
    radial_psd = tbin[valid] / nr[valid]

    # Convert radius to cycles per pixel (0.0 to 0.5 at Nyquist)
    spatial_freqs = (np.arange(len(radial_psd)) / (2.0 * max_r)).astype(np.float32)
    return spatial_freqs, radial_psd.astype(np.float32)


def compute_fourier_ring_correlation(
    img1: np.ndarray,
    img2: np.ndarray,
    threshold: float = 0.143,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute Fourier Ring Correlation (FRC) between two independent SR realizations.

    FRC measures correlation across frequency rings. The resolution cutoff is
    the spatial frequency where FRC drops below the threshold (standard 0.143 or 1/2-bit).

    Parameters
    ----------
    img1, img2 : np.ndarray, shape (H, W)
        Two independent stochastic diffusion realizations from different random seeds.
    threshold : float
        Correlation threshold (default 0.143 for 1/2 bit criterion).

    Returns
    -------
    freqs : np.ndarray
    frc_curve : np.ndarray
    cutoff_freq : float
    """
    arr1 = np.asarray(img1, dtype=np.float32)
    arr2 = np.asarray(img2, dtype=np.float32)
    H, W = arr1.shape

    win = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    F1 = np.fft.fftshift(np.fft.fft2((arr1 - np.mean(arr1)) * win))
    F2 = np.fft.fftshift(np.fft.fft2((arr2 - np.mean(arr2)) * win))

    cross = F1 * np.conj(F2)
    p1 = np.abs(F1) ** 2
    p2 = np.abs(F2) ** 2

    cy, cx = H // 2, W // 2
    y, x = np.ogrid[-cy : H - cy, -cx : W - cx]
    r_int = np.sqrt(x * x + y * y).astype(np.int32)
    max_r = min(cy, cx)

    # Ring sums
    ring_cross = np.bincount(r_int.ravel(), np.real(cross).ravel())
    ring_p1 = np.bincount(r_int.ravel(), p1.ravel())
    ring_p2 = np.bincount(r_int.ravel(), p2.ravel())

    valid = (np.arange(len(ring_cross)) <= max_r) & (ring_p1 > 0) & (ring_p2 > 0)
    denom = np.sqrt(ring_p1[valid] * ring_p2[valid]) + 1e-12
    frc_curve = np.real(ring_cross[valid]) / denom

    freqs = np.arange(len(frc_curve)) / (2.0 * max_r)

    # Find cutoff where FRC drops below threshold
    below_thresh = np.where(frc_curve < threshold)[0]
    if len(below_thresh) > 0 and below_thresh[0] > 1:
        cutoff_idx = below_thresh[0]
        cutoff_freq = float(freqs[cutoff_idx])
    else:
        cutoff_freq = float(freqs[-1]) if len(freqs) > 0 else 0.50

    return freqs.astype(np.float32), frc_curve.astype(np.float32), cutoff_freq


class FrequencyResolutionAnalyzer:
    """Evaluates spatial frequency spectra, MTF, and effective resolving power."""

    def __init__(self, nominal_gsd_meters: float = 2.5):
        self.nominal_gsd = nominal_gsd_meters

    def analyze(
        self,
        sr_mean: np.ndarray,
        sr_realization_b: Optional[np.ndarray] = None,
        bicubic_baseline: Optional[np.ndarray] = None,
    ) -> FrequencyResolutionMetrics:
        """Analyze frequency properties of super-resolved imagery.

        Parameters
        ----------
        sr_mean : np.ndarray, shape (C, H, W) or (H, W)
            Primary SR prediction.
        sr_realization_b : np.ndarray, optional
            Independent stochastic realization (e.g., seed 1) for FRC computation.
        bicubic_baseline : np.ndarray, optional
            Bicubic interpolation of native input.
        """
        arr = np.asarray(sr_mean, dtype=np.float32)
        if arr.ndim == 3:
            # Analyze luminance / average across visible bands
            arr_2d = np.mean(arr[:3], axis=0) if arr.shape[0] >= 3 else arr[0]
        else:
            arr_2d = arr

        freqs, psd = compute_radial_power_spectrum(arr_2d)

        metrics = FrequencyResolutionMetrics(nominal_gsd_meters=self.nominal_gsd)

        # 1. RPSD slope (log-log linear regression for mid-frequencies 0.05 to 0.35)
        mid_mask = (freqs >= 0.05) & (freqs <= 0.35) & (psd > 0)
        if np.sum(mid_mask) > 5:
            log_f = np.log10(freqs[mid_mask])
            log_p = np.log10(psd[mid_mask])
            slope, _ = np.polyfit(log_f, log_p, 1)
            metrics.spectral_rolloff_slope = float(slope)

        # 2. Estimate MTF50: normalized frequency where RPSD drops to 50% relative to low-frequency baseline
        if len(psd) > 10:
            low_freq_power = float(np.mean(psd[1:4]))
            if low_freq_power > 0:
                normalized_psd = psd / low_freq_power
                mtf50_indices = np.where(normalized_psd <= 0.50)[0]
                if len(mtf50_indices) > 0 and mtf50_indices[0] > 0:
                    metrics.mtf50_frequency = float(freqs[mtf50_indices[0]])
                else:
                    metrics.mtf50_frequency = float(freqs[len(freqs) // 2])

        # 3. Fourier Ring Correlation & Effective Resolving Power
        if sr_realization_b is not None:
            arr_b = np.asarray(sr_realization_b, dtype=np.float32)
            if arr_b.ndim == 3:
                arr_b_2d = np.mean(arr_b[:3], axis=0) if arr_b.shape[0] >= 3 else arr_b[0]
            else:
                arr_b_2d = arr_b

            _, _, cutoff = compute_fourier_ring_correlation(arr_2d, arr_b_2d)
            metrics.frc_cutoff_frequency = cutoff
            if cutoff > 0.01:
                # Effective GSD: in physical units
                # Maximum Nyquist frequency corresponds to 1 / (2 * nominal_gsd)
                # If cutoff is at f_cutoff in [0, 0.5], eff_gsd = nominal_gsd / (2 * f_cutoff)
                metrics.effective_gsd_meters = float(self.nominal_gsd * (0.50 / cutoff))
            else:
                metrics.effective_gsd_meters = 10.0
        else:
            # Fallback estimation based on MTF50
            if metrics.mtf50_frequency and metrics.mtf50_frequency > 0:
                metrics.effective_gsd_meters = float(self.nominal_gsd * (0.25 / metrics.mtf50_frequency))

        # 4. High-Frequency Hallucination Index (HFHI)
        # Power beyond 0.35 cycles/pixel (approaching Nyquist limit) compared to baseline bicubic
        high_freq_mask = freqs > 0.35
        if np.any(high_freq_mask):
            sr_high_power = float(np.sum(psd[high_freq_mask]))
            if bicubic_baseline is not None:
                b_arr = np.asarray(bicubic_baseline, dtype=np.float32)
                b_2d = np.mean(b_arr[:3], axis=0) if b_arr.ndim == 3 and b_arr.shape[0] >= 3 else b_arr
                _, b_psd = compute_radial_power_spectrum(b_2d)
                b_high_power = float(np.sum(b_psd[high_freq_mask])) if len(b_psd) >= len(psd) else 1e-6
                # Ratio of generative high frequencies over smoothed bicubic
                metrics.high_frequency_hallucination_index = float(
                    sr_high_power / (b_high_power + 1e-8)
                )
            else:
                metrics.high_frequency_hallucination_index = float(sr_high_power / (np.sum(psd) + 1e-8))

        eff_str = f"{metrics.effective_gsd_meters:.2f} m" if metrics.effective_gsd_meters else "N/A"
        slope_str = f"{metrics.spectral_rolloff_slope:.2f}" if metrics.spectral_rolloff_slope is not None else "N/A"
        metrics.diagnostic_summary = (
            f"Nominal grid: {self.nominal_gsd:.1f} m | Estimated effective resolution: {eff_str} (experimental protocol) | "
            f"RPSD slope: {slope_str} | "
            f"High-frequency ratio: {metrics.high_frequency_hallucination_index:.2f}"
        )
        return metrics
