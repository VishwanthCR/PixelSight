"""
Uncertainty analysis for PixelSight LDSR-S2.

Mechanism
---------
The PixelSight Plan 3 uncertainty is produced by running the LDSR-S2 diffusion
model N times (default 5) with different random seeds and computing the
pixel-wise standard deviation of the resulting SR outputs.

This is STOCHASTIC DIFFUSION VARIATION uncertainty.

It is NOT:
  - MC-dropout uncertainty
  - Bayesian posterior uncertainty
  - Calibrated predictive uncertainty

The standard deviation quantifies how much the reconstructed fine-scale
detail changes across different diffusion trajectories for the same input.
Regions with high std are more sensitive to the random noise schedule and
should be interpreted with caution.

Scientific conventions
----------------------
* Uncertainty is always described as "stochastic diffusion variation" or
  "reconstruction uncertainty", never as "predictive uncertainty" in the
  Bayesian sense unless the model actually uses Bayesian inference.
* When an HR reference exists, the correlation between uncertainty and
  reconstruction error is measured.  Causality is NOT claimed.
* Negative or null correlation results are preserved and reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyStatistics:
    """Descriptive statistics for the per-pixel uncertainty map."""

    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    p10: float | None = None
    p25: float | None = None
    p75: float | None = None
    p90: float | None = None
    p99: float | None = None
    total_pixels: int | None = None

    # Region-level fractions (low/medium/high based on thresholds)
    low_uncertainty_fraction: float | None = None
    medium_uncertainty_fraction: float | None = None
    high_uncertainty_fraction: float | None = None
    low_threshold: float | None = None
    high_threshold: float | None = None

    # Mechanism description — always populated
    mechanism: str = (
        "Stochastic diffusion variation: pixel-wise standard deviation across "
        "N independent LDSR-S2 runs with different random seeds (100 diffusion "
        "sampling steps each).  This is NOT MC-dropout uncertainty."
    )
    n_samples: int | None = None
    sampling_steps: int = 100
    scale: int = 4


@dataclass
class UncertaintyErrorRelation:
    """Relationship between uncertainty and reconstruction/downstream error."""

    pearson_correlation: float | None = None
    spearman_correlation: float | None = None

    # Region-level error rates (error = misclassification or reconstruction MAE)
    low_uncertainty_region_error: float | None = None
    medium_uncertainty_region_error: float | None = None
    high_uncertainty_region_error: float | None = None

    low_threshold: float | None = None
    high_threshold: float | None = None

    # Quintile-based analysis (lowest 20%, middle 60%, highest 20%)
    lowest_20pct_error: float | None = None
    middle_60pct_error: float | None = None
    highest_20pct_error: float | None = None

    # Difference statistics
    high_vs_low_error_diff_pct_points: float | None = None
    high_vs_low_relative_increase_pct: float | None = None

    error_type: str = ""  # "downstream_segmentation_error_rate" | "reconstruction_mae"
    hr_reference_used: bool = False
    n_pixels: int | None = None

    interpretation: str = (
        "A higher error rate in high-uncertainty regions indicates that "
        "uncertainty carries information about potentially unreliable outputs. "
        "This is a statistical association — causality is NOT established."
    )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def uncertainty_statistics(
    uncertainty_map: np.ndarray,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    n_samples: int | None = None,
    sampling_steps: int = 100,
) -> UncertaintyStatistics:
    """Compute descriptive statistics for an uncertainty map.

    Parameters
    ----------
    uncertainty_map : np.ndarray
        Per-pixel uncertainty (std across N diffusion runs).
        May be shape (H, W) or (C, H, W).  If (C, H, W), the channel-mean
        is used.
    low_threshold : float, optional
        Below this value = low uncertainty.  Defaults to p33.
    high_threshold : float, optional
        Above this value = high uncertainty.  Defaults to p67.
    n_samples : int, optional
        Number of stochastic samples used (for provenance).
    sampling_steps : int
        Diffusion sampling steps per run (default 100).

    Returns
    -------
    UncertaintyStatistics
    """
    umap = np.asarray(uncertainty_map, dtype=np.float32)
    if umap.ndim == 3:
        umap = umap.mean(axis=0)  # (C, H, W) -> (H, W)

    finite = umap[np.isfinite(umap)]

    result = UncertaintyStatistics(
        n_samples=n_samples,
        sampling_steps=sampling_steps,
    )

    if finite.size == 0:
        return result

    result.total_pixels = int(finite.size)
    result.mean = float(np.mean(finite))
    result.median = float(np.median(finite))
    result.std = float(np.std(finite))
    result.min = float(np.min(finite))
    result.max = float(np.max(finite))
    result.p10 = float(np.percentile(finite, 10))
    result.p25 = float(np.percentile(finite, 25))
    result.p75 = float(np.percentile(finite, 75))
    result.p90 = float(np.percentile(finite, 90))
    result.p99 = float(np.percentile(finite, 99))

    # Derive thresholds from percentiles if not provided
    lo = low_threshold if low_threshold is not None else float(np.percentile(finite, 33))
    hi = high_threshold if high_threshold is not None else float(np.percentile(finite, 67))
    result.low_threshold = lo
    result.high_threshold = hi

    total = float(finite.size)
    result.low_uncertainty_fraction = float((finite < lo).sum()) / total
    result.medium_uncertainty_fraction = float(
        ((finite >= lo) & (finite <= hi)).sum()
    ) / total
    result.high_uncertainty_fraction = float((finite > hi).sum()) / total

    return result


def uncertainty_vs_error(
    uncertainty_map: np.ndarray,
    error_map: np.ndarray,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    error_type: str = "reconstruction_mae",
    hr_reference_used: bool = False,
) -> UncertaintyErrorRelation:
    """Measure the statistical relationship between uncertainty and error.

    Parameters
    ----------
    uncertainty_map : np.ndarray, shape (H, W) or (C, H, W)
        Per-pixel uncertainty.
    error_map : np.ndarray, shape (H, W)
        Per-pixel error (e.g. segmentation error indicator or reconstruction MAE).
        For segmentation: 1 where prediction != label, 0 where correct.
        For reconstruction: absolute difference vs HR reference.
    low_threshold, high_threshold : float, optional
        Region boundaries.  Default: p33 / p67 of uncertainty.
    error_type : str
        Description of error metric.
    hr_reference_used : bool
        True if error_map is computed against a genuine HR reference.

    Returns
    -------
    UncertaintyErrorRelation
    """
    umap = np.asarray(uncertainty_map, dtype=np.float32)
    if umap.ndim == 3:
        umap = umap.mean(axis=0)

    emap = np.asarray(error_map, dtype=np.float32)

    # Align shapes
    if umap.shape != emap.shape:
        from scipy.ndimage import zoom
        scale_y = umap.shape[0] / emap.shape[0]
        scale_x = umap.shape[1] / emap.shape[1]
        if scale_y != 1.0 or scale_x != 1.0:
            emap = zoom(emap, (scale_y, scale_x), order=1).astype(np.float32)

    u_flat = umap.ravel()
    e_flat = emap.ravel()

    valid = np.isfinite(u_flat) & np.isfinite(e_flat)
    u_v = u_flat[valid]
    e_v = e_flat[valid]

    result = UncertaintyErrorRelation(
        error_type=error_type,
        hr_reference_used=hr_reference_used,
        n_pixels=int(u_v.size),
    )

    if u_v.size < 2:
        return result

    # Pearson correlation
    corr_matrix = np.corrcoef(u_v, e_v)
    result.pearson_correlation = float(corr_matrix[0, 1])

    # Spearman correlation (rank-based)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(u_v, e_v)
    result.spearman_correlation = float(rho)

    # Region-level error rates
    lo = low_threshold if low_threshold is not None else float(np.percentile(u_v, 33))
    hi = high_threshold if high_threshold is not None else float(np.percentile(u_v, 67))
    result.low_threshold = lo
    result.high_threshold = hi

    low_mask = u_v < lo
    med_mask = (u_v >= lo) & (u_v <= hi)
    high_mask = u_v > hi

    result.low_uncertainty_region_error = (
        float(e_v[low_mask].mean()) if low_mask.any() else None
    )
    result.medium_uncertainty_region_error = (
        float(e_v[med_mask].mean()) if med_mask.any() else None
    )
    result.high_uncertainty_region_error = (
        float(e_v[high_mask].mean()) if high_mask.any() else None
    )

    # Quintiles: lowest 20% (p20), middle 60% (p20-p80), highest 20% (p80)
    p20 = float(np.percentile(u_v, 20))
    p80 = float(np.percentile(u_v, 80))
    q_low = u_v <= p20
    q_mid = (u_v > p20) & (u_v <= p80)
    q_high = u_v > p80
    result.lowest_20pct_error = float(e_v[q_low].mean()) if q_low.any() else None
    result.middle_60pct_error = float(e_v[q_mid].mean()) if q_mid.any() else None
    result.highest_20pct_error = float(e_v[q_high].mean()) if q_high.any() else None

    if (result.low_uncertainty_region_error is not None
            and result.high_uncertainty_region_error is not None
            and result.low_uncertainty_region_error > 0):
        result.high_vs_low_error_diff_pct_points = float(
            (result.high_uncertainty_region_error - result.low_uncertainty_region_error) * 100.0
        )
        result.high_vs_low_relative_increase_pct = float(
            (result.high_uncertainty_region_error / result.low_uncertainty_region_error - 1.0) * 100.0
        )

    return result


def export_uncertainty_error_analysis(
    relation: UncertaintyErrorRelation,
    output_path: str | Path,
) -> dict[str, Any]:
    """Save uncertainty_error_analysis.json report."""
    import json
    from dataclasses import asdict
    d = asdict(relation)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(d, f, indent=2)
    return d


def plot_uncertainty_vs_error(
    uncertainty_map: np.ndarray,
    error_map: np.ndarray,
    output_png_path: str | Path,
    title: str = "Uncertainty vs. Reconstruction Error",
) -> Path:
    """Generate uncertainty_vs_error.png calibration diagnostic curve."""
    import matplotlib.pyplot as plt
    out_path = Path(output_png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    u = np.asarray(uncertainty_map).ravel()
    e = np.asarray(error_map).ravel()
    valid = np.isfinite(u) & np.isfinite(e)
    u_v = u[valid]
    e_v = e[valid]

    # Subsample if large for responsive plotting
    if len(u_v) > 20000:
        idx = np.random.choice(len(u_v), 20000, replace=False)
        u_sub = u_v[idx]
        e_sub = e_v[idx]
    else:
        u_sub = u_v
        e_sub = e_v

    # Bin into deciles
    bins = np.linspace(np.percentile(u_v, 1), np.percentile(u_v, 99), 11)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_errors = []
    for i in range(len(bins) - 1):
        mask = (u_v >= bins[i]) & (u_v < bins[i+1])
        bin_errors.append(float(e_v[mask].mean()) if mask.any() else np.nan)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.scatter(u_sub, e_sub, alpha=0.08, s=2, color="#4A90E2", label="Pixels")
    ax.plot(bin_centers, bin_errors, color="#D0021B", linewidth=2.5, marker="o", label="Binned Mean Error")
    ax.set_xlabel(r"Reconstruction Uncertainty (Diffusion $\sigma$)")
    ax.set_ylabel("Reconstruction Absolute Error")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class UncertaintyAnalysis:
    """Load and analyse Plan 3 uncertainty results.

    Parameters
    ----------
    uncertainty_map_path : str | Path
        Path to the uncertainty_map.npy (or std_sr.npy) file.
    mean_sr_path : str | Path, optional
        Path to the mean SR array (mean_sr.npy).
    samples_dir : str | Path, optional
        Directory containing individual stochastic samples.
    n_samples : int
        Number of stochastic runs that produced the uncertainty map.
    """

    def __init__(
        self,
        uncertainty_map_path: str | Path,
        mean_sr_path: str | Path | None = None,
        samples_dir: str | Path | None = None,
        n_samples: int = 5,
        sampling_steps: int = 100,
    ) -> None:
        self.uncertainty_map_path = Path(uncertainty_map_path)
        self.mean_sr_path = Path(mean_sr_path) if mean_sr_path else None
        self.samples_dir = Path(samples_dir) if samples_dir else None
        self.n_samples = n_samples
        self.sampling_steps = sampling_steps

        self._uncertainty_map: np.ndarray | None = None
        self._mean_sr: np.ndarray | None = None

    @property
    def uncertainty_map(self) -> np.ndarray:
        if self._uncertainty_map is None:
            if not self.uncertainty_map_path.exists():
                raise FileNotFoundError(
                    f"Uncertainty map not found: {self.uncertainty_map_path}"
                )
            self._uncertainty_map = np.load(self.uncertainty_map_path).astype(np.float32)
        return self._uncertainty_map

    @property
    def mean_sr(self) -> np.ndarray | None:
        if self._mean_sr is None and self.mean_sr_path and self.mean_sr_path.exists():
            self._mean_sr = np.load(self.mean_sr_path).astype(np.float32)
        return self._mean_sr

    def compute_statistics(
        self,
        low_threshold: float | None = None,
        high_threshold: float | None = None,
    ) -> UncertaintyStatistics:
        return uncertainty_statistics(
            self.uncertainty_map,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            n_samples=self.n_samples,
            sampling_steps=self.sampling_steps,
        )

    def compute_error_relation(
        self,
        error_map: np.ndarray,
        error_type: str = "reconstruction_mae",
        hr_reference_used: bool = False,
        low_threshold: float | None = None,
        high_threshold: float | None = None,
    ) -> UncertaintyErrorRelation:
        return uncertainty_vs_error(
            self.uncertainty_map,
            error_map,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            error_type=error_type,
            hr_reference_used=hr_reference_used,
        )
