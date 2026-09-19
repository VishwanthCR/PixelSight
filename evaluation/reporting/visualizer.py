"""
PixelSight Scientific Visualizer
================================
Generates publication-quality multi-panel scientific figures for
remote sensing and super-resolution mapping evaluation.

Key outputs:
1. 10-panel Master Research Figure (Native, Bicubic, SR, Spectral NDVI,
   Diffusion Uncertainty, Categorical Reliability, Downstream Task, Error Calibration).
2. Uncertainty-Error Calibration Curves with error bars.
3. Spectral Signature and Radial Frequency Profile plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches


def _normalize_display_rgb(img_4band: np.ndarray) -> np.ndarray:
    """Normalize 4-band (B02, B03, B04, B08) or 3-band array to [0, 1] RGB for display.
    
    Assumes bands: 0=Blue, 1=Green, 2=Red, 3=NIR.
    RGB display: (Red, Green, Blue) -> bands (2, 1, 0).
    """
    arr = np.asarray(img_4band, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        if arr.shape[0] >= 3:
            rgb = np.stack([arr[2], arr[1], arr[0]], axis=-1)
        else:
            rgb = np.repeat(arr[0:1], 3, axis=0)
            rgb = np.moveaxis(rgb, 0, -1)
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        rgb = np.stack([arr[:, :, 2], arr[:, :, 1], arr[:, :, 0]], axis=-1)
    elif arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=-1)
    else:
        rgb = np.zeros((arr.shape[-2], arr.shape[-1], 3), dtype=np.float32)

    # 2% - 98% percentile stretch for satellite imagery
    p2, p98 = np.percentile(rgb[np.isfinite(rgb)], (2, 98))
    if p98 > p2:
        rgb = np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)
    else:
        rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def _compute_ndvi_for_plot(img_4band: np.ndarray) -> np.ndarray:
    """Extract NDVI from (C, H, W) or (H, W, C) array with Red=2, NIR=3."""
    arr = np.asarray(img_4band, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        red = arr[2]
        nir = arr[3] if arr.shape[0] > 3 else arr[0]
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        red = arr[:, :, 2]
        nir = arr[:, :, 3] if arr.shape[2] > 3 else arr[:, :, 0]
    else:
        return np.zeros((arr.shape[-2], arr.shape[-1]), dtype=np.float32)
    
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = nir + red
        ndvi = np.where(denom > 1e-8, (nir - red) / denom, 0.0)
    return np.clip(ndvi, -1.0, 1.0).astype(np.float32)


def plot_master_ten_panel(
    native_10m: np.ndarray,
    bicubic_2p5m: np.ndarray,
    sr_2p5m: np.ndarray,
    uncertainty_map: np.ndarray,
    reliability_map: np.ndarray | None = None,
    downstream_pred: np.ndarray | None = None,
    error_map: np.ndarray | None = None,
    output_path: str | Path = "results/figures/master_research_figure.png",
    dpi: int = 200,
    title: str = "PixelSight: Multi-Scale Spectral, Spatial & Uncertainty Diagnostics",
) -> Path:
    """Generate the flagship 10-panel scientific research figure.

    Panels:
    1. Native 10m RGB (Observed)
    2. Native 10m NDVI (Observed)
    3. Bicubic 2.5m RGB (Interpolated Baseline)
    4. Bicubic 2.5m NDVI (Interpolated Baseline)
    5. LDSR-S2 2.5m RGB (Super-Resolved Representation)
    6. LDSR-S2 2.5m NDVI (Super-Resolved Representation)
    7. Reconstruction Uncertainty Map (Stochastic Diffusion σ across 5 seeds)
    8. Spatial Reliability Classification (High / Medium / Low)
    9. Downstream Task Map / Proxy Segmentation
    10. Uncertainty vs Error Calibration Diagnostic
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare visual arrays
    rgb_native = _normalize_display_rgb(native_10m)
    ndvi_native = _compute_ndvi_for_plot(native_10m)

    rgb_bicubic = _normalize_display_rgb(bicubic_2p5m)
    ndvi_bicubic = _compute_ndvi_for_plot(bicubic_2p5m)

    rgb_sr = _normalize_display_rgb(sr_2p5m)
    ndvi_sr = _compute_ndvi_for_plot(sr_2p5m)

    u_map = np.squeeze(np.asarray(uncertainty_map, dtype=np.float32))

    fig = plt.figure(figsize=(20, 9), dpi=dpi)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    # 2 rows x 5 columns layout
    gs = fig.add_gridspec(2, 5, hspace=0.28, wspace=0.25, left=0.04, right=0.96, top=0.91, bottom=0.06)

    # Panel 1: Native RGB
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb_native)
    ax1.set_title("(a) Native Sentinel-2 (10 m)\nObserved RGB", fontsize=10, fontweight="bold")
    ax1.axis("off")

    # Panel 2: Native NDVI
    ax2 = fig.add_subplot(gs[1, 0])
    im_n_ndvi = ax2.imshow(ndvi_native, cmap="YlGn", vmin=-0.2, vmax=0.8)
    ax2.set_title("(b) Native 10 m NDVI\nObserved Spectral Index", fontsize=10, fontweight="bold")
    ax2.axis("off")
    plt.colorbar(im_n_ndvi, ax=ax2, orientation="horizontal", fraction=0.046, pad=0.04, label="NDVI")

    # Panel 3: Bicubic RGB
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.imshow(rgb_bicubic)
    ax3.set_title("(c) Bicubic Baseline (2.5 m)\nInterpolated RGB", fontsize=10, fontweight="bold")
    ax3.axis("off")

    # Panel 4: Bicubic NDVI
    ax4 = fig.add_subplot(gs[1, 1])
    im_b_ndvi = ax4.imshow(ndvi_bicubic, cmap="YlGn", vmin=-0.2, vmax=0.8)
    ax4.set_title("(d) Bicubic 2.5 m NDVI\nSmoothed Index", fontsize=10, fontweight="bold")
    ax4.axis("off")
    plt.colorbar(im_b_ndvi, ax=ax4, orientation="horizontal", fraction=0.046, pad=0.04, label="NDVI")

    # Panel 5: LDSR-S2 RGB
    ax5 = fig.add_subplot(gs[0, 2])
    ax5.imshow(rgb_sr)
    ax5.set_title("(e) PixelSight LDSR-S2 (~2.5 m)\nSuper-Resolved RGB", fontsize=10, fontweight="bold")
    ax5.axis("off")

    # Panel 6: LDSR-S2 NDVI
    ax6 = fig.add_subplot(gs[1, 2])
    im_s_ndvi = ax6.imshow(ndvi_sr, cmap="YlGn", vmin=-0.2, vmax=0.8)
    ax6.set_title("(f) LDSR-S2 2.5 m NDVI\nSub-Pixel Spectral Index", fontsize=10, fontweight="bold")
    ax6.axis("off")
    plt.colorbar(im_s_ndvi, ax=ax6, orientation="horizontal", fraction=0.046, pad=0.04, label="NDVI")

    # Panel 7: Uncertainty Map
    ax7 = fig.add_subplot(gs[0, 3])
    p95_u = float(np.percentile(u_map, 98)) if u_map.size > 0 else 0.01
    im_u = ax7.imshow(u_map, cmap="inferno", vmin=0.0, vmax=max(p95_u, 1e-4))
    ax7.set_title(r"(g) Diffusion Uncertainty" + "\n" + r"Stochastic $\sigma$ (N=5 seeds)", fontsize=10, fontweight="bold")
    ax7.axis("off")
    plt.colorbar(im_u, ax=ax7, orientation="horizontal", fraction=0.046, pad=0.04, label=r"Std Dev ($\sigma$)")

    # Panel 8: Reliability Map
    ax8 = fig.add_subplot(gs[1, 3])
    if reliability_map is not None:
        rel = np.squeeze(np.asarray(reliability_map, dtype=np.int32))
    else:
        # Construct categorical from quartiles
        q33, q66 = np.percentile(u_map, [33, 66])
        rel = np.where(u_map <= q33, 0, np.where(u_map <= q66, 1, 2))

    cmap_rel = ListedColormap(["#2ECC71", "#F39C12", "#E74C3C"])  # High=Green, Med=Orange, Low=Red
    norm_rel = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_rel.N)
    ax8.imshow(rel, cmap=cmap_rel, norm=norm_rel)
    ax8.set_title("(h) Spatial Reliability Map\nConfidence Tiers", fontsize=10, fontweight="bold")
    ax8.axis("off")
    patches = [
        mpatches.Patch(color="#2ECC71", label="High Reliability"),
        mpatches.Patch(color="#F39C12", label="Medium"),
        mpatches.Patch(color="#E74C3C", label="Low (High Unc.)"),
    ]
    ax8.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=1, fontsize=8, frameon=False)

    # Panel 9: Downstream Map (or Error Map if provided)
    ax9 = fig.add_subplot(gs[0, 4])
    if downstream_pred is not None:
        down_arr = np.squeeze(np.asarray(downstream_pred))
        im_down = ax9.imshow(down_arr, cmap="tab10")
        ax9.set_title("(i) Downstream Task\nWorldCover Proxy / Classes", fontsize=10, fontweight="bold")
        ax9.axis("off")
    elif error_map is not None:
        err_arr = np.squeeze(np.asarray(error_map, dtype=np.float32))
        im_err = ax9.imshow(err_arr, cmap="magma")
        ax9.set_title("(i) Downstream Error Map\nAbsolute Discrepancy", fontsize=10, fontweight="bold")
        ax9.axis("off")
        plt.colorbar(im_err, ax=ax9, orientation="horizontal", fraction=0.046, pad=0.04, label="Error")
    else:
        # High-frequency edge map
        edge = np.abs(ndvi_sr - ndvi_bicubic)
        im_edge = ax9.imshow(edge, cmap="plasma")
        ax9.set_title("(i) High-Frequency Residual\n|SR - Bicubic|", fontsize=10, fontweight="bold")
        ax9.axis("off")
        plt.colorbar(im_edge, ax=ax9, orientation="horizontal", fraction=0.046, pad=0.04, label="Residual")

    # Panel 10: Uncertainty vs Error Diagnostic / Scatter Curve
    ax10 = fig.add_subplot(gs[1, 4])
    u_flat = u_map.ravel()
    if error_map is not None:
        e_flat = np.asarray(error_map, dtype=np.float32).ravel()
    else:
        # Consistency error vs native upsampled baseline
        up_native = ndvi_native
        if up_native.shape != ndvi_sr.shape:
            from scipy.ndimage import zoom
            scale = (ndvi_sr.shape[0] / up_native.shape[0], ndvi_sr.shape[1] / up_native.shape[1])
            up_native = zoom(up_native, scale, order=1)
        e_flat = np.abs(ndvi_sr - up_native).ravel()

    valid = np.isfinite(u_flat) & np.isfinite(e_flat)
    u_val = u_flat[valid]
    e_val = e_flat[valid]

    if len(u_val) > 20:
        # Binned quintile curve
        n_bins = 6
        bins = np.quantile(u_val, np.linspace(0, 1, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) > 2:
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bin_means = []
            bin_errs = []
            for b_low, b_high in zip(bins[:-1], bins[1:]):
                mask = (u_val >= b_low) & (u_val <= b_high)
                if np.any(mask):
                    bin_means.append(float(np.mean(e_val[mask])))
                    bin_errs.append(float(np.std(e_val[mask]) / np.sqrt(np.sum(mask))))
                else:
                    bin_means.append(float("nan"))
                    bin_errs.append(0.0)

            # Subsample points for scatter
            idx_sub = np.random.choice(len(u_val), size=min(1500, len(u_val)), replace=False)
            ax10.scatter(u_val[idx_sub], e_val[idx_sub], alpha=0.15, s=4, color="#3498DB", label="Sample Pixels")
            ax10.errorbar(bin_centers, bin_means, yerr=bin_errs, fmt="-o", color="#E74C3C", linewidth=2, capsize=3, label="Quantile Mean")
            ax10.set_xlabel(r"Diffusion $\sigma$", fontsize=8)
            ax10.set_ylabel("Error Metric", fontsize=8)
            ax10.set_title(r"(j) Uncertainty Calibration" + "\n" + r"Error vs Stochastic $\sigma$", fontsize=10, fontweight="bold")
            ax10.tick_params(labelsize=7)
            ax10.legend(fontsize=7, loc="upper left")
            ax10.grid(True, linestyle="--", alpha=0.4)
        else:
            ax10.text(0.5, 0.5, "Uniform Uncertainty", ha="center", va="center", transform=ax10.transAxes)
    else:
        ax10.text(0.5, 0.5, "Diagnostic Unavailable", ha="center", va="center", transform=ax10.transAxes)

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_quintile_error_curve(
    uncertainty_map: np.ndarray,
    error_map: np.ndarray,
    output_path: str | Path = "results/figures/quintile_calibration.png",
    dpi: int = 150,
    title: str = "Uncertainty Quintile Error Curve",
) -> Path:
    """Generate publication-ready quintile calibration curve with Pearson r."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    u_flat = np.asarray(uncertainty_map, dtype=np.float32).ravel()
    e_flat = np.asarray(error_map, dtype=np.float32).ravel()
    valid = np.isfinite(u_flat) & np.isfinite(e_flat)
    u_val = u_flat[valid]
    e_val = e_flat[valid]

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=dpi)
    quantiles = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_edges = np.quantile(u_val, quantiles)
    bin_edges = np.unique(bin_edges)

    q_labels = ["Q1 (Lowest)", "Q2", "Q3", "Q4", "Q5 (Highest)"][:len(bin_edges) - 1]
    q_errors = []
    q_stds = []

    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (u_val >= low) & (u_val <= high)
        if np.any(mask):
            q_errors.append(float(np.mean(e_val[mask])))
            q_stds.append(float(np.std(e_val[mask])))
        else:
            q_errors.append(0.0)
            q_stds.append(0.0)

    bars = ax.bar(q_labels, q_errors, yerr=q_stds, capsize=4, color="#3498DB", edgecolor="#2980B9", alpha=0.85)
    ax.set_ylabel("Mean Reconstruction Error", fontsize=10, fontweight="bold")
    ax.set_xlabel("Uncertainty Quintile", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate correlation
    if len(u_val) > 2:
        r = float(np.corrcoef(u_val, e_val)[0, 1])
        ax.text(
            0.05, 0.92, f"Pearson r = {r:.4f}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#BDC3C7"),
            fontsize=9,
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
