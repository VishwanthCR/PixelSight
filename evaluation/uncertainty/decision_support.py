r"""
PixelSight Uncertainty-Gated Decision Support & Hybrid Fusion
============================================================
Translates stochastic diffusion uncertainty into actionable, risk-controlled
downstream geospatial workflows (precision agriculture, flood/damage assessment).

Core Concepts:
--------------
1. Pure generative SR introduces hallucination risk in low-confidence regions.
2. Direct native 10 m observation is radiometrically faithful but spatially coarse.
3. Uncertainty-Gated Hybrid Fusion blends the two representations:
   - High-Confidence (Low \sigma): Preserves 2.5 m super-resolved micro-boundaries.
   - Low-Confidence (High \sigma): Smooths back to the radiometrically conserved native observation.
4. Downstream Risk Reduction: Quantifies false-positive suppression in critical decision tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class DecisionSupportResult:
    """Quantitative performance of uncertainty-guided hybrid fusion."""

    # Percentage of pixels routed to high-resolution generative representation
    sr_retention_percentage: float = 0.0

    # Percentage of pixels fallback-routed to calibrated native spectrum
    native_fallback_percentage: float = 0.0

    # Gradient energy (spatial sharpness) of the hybrid image relative to raw SR
    sharpness_retention_ratio: float = 0.0

    # Spectral consistency error (MAE vs native) of hybrid vs raw SR (negative indicates improved fidelity)
    spectral_error_reduction_pct: float = 0.0

    # High-Uncertainty SR Suppression Rate in low-confidence regions
    high_uncertainty_suppression_rate: float = 0.0
    hallucination_suppression_rate: float = 0.0  # backward-compatible alias

    # Overall Decision Safety Score in [0, 1]
    decision_safety_score: float = 0.0

    diagnostic_note: str = (
        "Uncertainty-gated fusion replaces high-uncertainty super-resolved regions with "
        "radiometrically conserved native observations to avoid high-risk false-positive decisions."
    )


class UncertaintyGatedFusionEngine:
    """Fuses super-resolved detail with native spectral fidelity based on diffusion uncertainty."""

    def __init__(
        self,
        uncertainty_percentile_cutoff: float = 75.0,
        transition_smoothness: float = 0.15,
    ):
        self.cutoff_percentile = uncertainty_percentile_cutoff
        self.smoothness = transition_smoothness

    def fuse(
        self,
        sr_2p5m: np.ndarray,
        native_10m: np.ndarray,
        uncertainty_map: np.ndarray,
    ) -> Tuple[np.ndarray, DecisionSupportResult]:
        """Perform risk-controlled hybrid fusion.

        Parameters
        ----------
        sr_2p5m : np.ndarray, shape (C, H, W)
            LDSR-S2 2.5 m prediction.
        native_10m : np.ndarray, shape (C, H_n, W_n)
            Native 10 m input.
        uncertainty_map : np.ndarray, shape (H, W)
            Stochastic diffusion standard deviation across N=5 seeds.

        Returns
        -------
        fused_image : np.ndarray, shape (C, H, W)
        result : DecisionSupportResult
        """
        sr = np.asarray(sr_2p5m, dtype=np.float32)
        nat = np.asarray(native_10m, dtype=np.float32)
        u_map = np.squeeze(np.asarray(uncertainty_map, dtype=np.float32))

        # 1. Upscale native to SR resolution using bicubic interpolation for seamless blending
        if nat.shape[-2:] != sr.shape[-2:]:
            from scipy.ndimage import zoom
            scale = (sr.shape[-2] / nat.shape[-2], sr.shape[-1] / nat.shape[-1])
            if nat.ndim == 3:
                up_nat = np.stack([zoom(nat[c], scale, order=3) for c in range(nat.shape[0])], axis=0)
            else:
                up_nat = zoom(nat, scale, order=3)
        else:
            up_nat = nat

        # 2. Determine confidence threshold tau from uncertainty distribution
        valid_u = u_map[np.isfinite(u_map)]
        tau = float(np.percentile(valid_u, self.cutoff_percentile)) if len(valid_u) > 0 else 0.005
        spread = max(float(np.std(valid_u)), 1e-5) * self.smoothness

        # Weight map: w = 1.0 (pure SR) when u << tau; w = 0.0 (pure native) when u >> tau
        # Sigmoidal transition with exponent clipping
        scaled_diff = np.clip((u_map - tau) / spread, -50.0, 50.0)
        w = 1.0 / (1.0 + np.exp(scaled_diff))
        w = np.clip(w, 0.0, 1.0).astype(np.float32)

        # 3. Fuse representations
        if sr.ndim == 3:
            w_expanded = w[np.newaxis, :, :]
            fused = w_expanded * sr + (1.0 - w_expanded) * up_nat
        else:
            fused = w * sr + (1.0 - w) * up_nat

        # 4. Compute decision support metrics
        high_conf_mask = w >= 0.50
        sr_pct = float(np.mean(high_conf_mask)) * 100.0
        fallback_pct = 100.0 - sr_pct

        # Sharpness retention: gradient energy of fused vs raw SR
        def _grad_energy(arr: np.ndarray) -> float:
            if arr.ndim == 3:
                arr = arr[0]
            gy = np.diff(arr, axis=0)
            gx = np.diff(arr, axis=1)
            return float(np.mean(gy[:, :gx.shape[1]] ** 2 + gx[:gy.shape[0], :] ** 2))

        sr_sharp = _grad_energy(sr)
        fused_sharp = _grad_energy(fused)
        sharp_ratio = float(fused_sharp / (sr_sharp + 1e-8))

        # Spectral error vs native upsampled baseline
        raw_sr_err = float(np.mean(np.abs(sr - up_nat)))
        fused_err = float(np.mean(np.abs(fused - up_nat)))
        err_reduction = float((raw_sr_err - fused_err) / (raw_sr_err + 1e-8) * 100.0)

        # Hallucination suppression rate: fraction of weight damped in top uncertainty quartile
        q75 = np.percentile(valid_u, 75.0) if len(valid_u) > 0 else 0.005
        top_u_mask = u_map >= q75
        suppression = float(np.mean(1.0 - w[top_u_mask])) if np.any(top_u_mask) else 0.0

        # Composite safety score in [0, 1]
        safety = float(np.clip(0.4 * (sr_pct / 100.0) + 0.3 * (err_reduction / 100.0) + 0.3 * suppression, 0.0, 1.0))

        result = DecisionSupportResult(
            sr_retention_percentage=round(sr_pct, 2),
            native_fallback_percentage=round(fallback_pct, 2),
            sharpness_retention_ratio=round(sharp_ratio, 3),
            spectral_error_reduction_pct=round(err_reduction, 2),
            high_uncertainty_suppression_rate=round(suppression * 100.0, 2),
            hallucination_suppression_rate=round(suppression * 100.0, 2),
            decision_safety_score=round(safety, 3),
        )

        return fused, result
