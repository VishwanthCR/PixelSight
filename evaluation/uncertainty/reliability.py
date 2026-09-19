"""
SR Reliability / Reconstruction-Risk Map.

Scientific conventions
----------------------
* Reliability categories (HIGH / MEDIUM / LOW) are based on reconstruction
  uncertainty level.  LOW reliability means "exercise caution", NOT that
  pixels are hallucinated or fabricated.
* Use scientifically cautious terminology:
    - "reconstruction uncertainty"
    - "low reliability region"
    - "reconstruction-risk region"
  NOT:
    - "hallucinated pixels"
    - "fabricated details"
* When a genuine HR reference is available, the reliability map is validated
  against the actual reconstruction error to assess whether the reliability
  categories are meaningful predictors of error.
* Causality is NOT claimed even when the correlation is strong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

_CATEGORY_HIGH = "HIGH_RELIABILITY"
_CATEGORY_MEDIUM = "MEDIUM_RELIABILITY"
_CATEGORY_LOW = "LOW_RELIABILITY"


@dataclass
class ReliabilityResult:
    """Spatial reliability assessment summary."""

    high_reliability_fraction: float | None = None
    medium_reliability_fraction: float | None = None
    low_reliability_fraction: float | None = None

    # Pixel counts
    high_reliability_pixels: int | None = None
    medium_reliability_pixels: int | None = None
    low_reliability_pixels: int | None = None
    total_pixels: int | None = None

    # Thresholds
    uncertainty_threshold_low: float | None = None   # below this = HIGH reliability
    uncertainty_threshold_high: float | None = None  # above this = LOW reliability

    # Validation against HR reference (when available)
    validated_against_hr: bool = False
    high_reliability_mean_error: float | None = None
    medium_reliability_mean_error: float | None = None
    low_reliability_mean_error: float | None = None
    reliability_error_correlation: float | None = None

    # Scientific documentation
    limitation_note: str = (
        "Reliability categories reflect reconstruction uncertainty, "
        "not confirmed error.  LOW reliability means the reconstruction "
        "is more sensitive to the diffusion random seed and should be "
        "interpreted with caution.  This does NOT imply pixels are hallucinated."
    )
    validated_note: str = ""


# ---------------------------------------------------------------------------
# Reliability map array (spatial output)
# ---------------------------------------------------------------------------

class ReliabilityMap:
    """Spatial reconstruction reliability map.

    Attributes
    ----------
    array : np.ndarray, shape (H, W), dtype uint8
        0 = HIGH_RELIABILITY, 1 = MEDIUM_RELIABILITY, 2 = LOW_RELIABILITY
    uncertainty_map : np.ndarray, shape (H, W)
        The underlying uncertainty map.
    thresholds : tuple[float, float]
        (low_threshold, high_threshold)
    """

    LABELS = {0: _CATEGORY_HIGH, 1: _CATEGORY_MEDIUM, 2: _CATEGORY_LOW}
    COLORS = {
        0: (0, 180, 0),    # GREEN — high reliability
        1: (255, 165, 0),  # ORANGE — medium reliability
        2: (200, 0, 0),    # RED — low reliability
    }

    def __init__(
        self,
        uncertainty_map: np.ndarray,
        low_threshold: float | None = None,
        high_threshold: float | None = None,
    ) -> None:
        umap = np.asarray(uncertainty_map, dtype=np.float32)
        if umap.ndim == 3:
            umap = umap.mean(axis=0)
        self.uncertainty_map = umap

        finite = umap[np.isfinite(umap)]
        self.low_threshold = (
            low_threshold if low_threshold is not None
            else float(np.percentile(finite, 33))
        )
        self.high_threshold = (
            high_threshold if high_threshold is not None
            else float(np.percentile(finite, 67))
        )
        self.thresholds = (self.low_threshold, self.high_threshold)

        # Build the reliability array
        arr = np.ones(umap.shape, dtype=np.uint8)  # default: MEDIUM (1)
        arr[umap < self.low_threshold] = 0          # HIGH reliability
        arr[umap > self.high_threshold] = 2         # LOW reliability
        self.array = arr

    @property
    def high_mask(self) -> np.ndarray:
        return self.array == 0

    @property
    def medium_mask(self) -> np.ndarray:
        return self.array == 1

    @property
    def low_mask(self) -> np.ndarray:
        return self.array == 2

    def to_result(self, error_map: np.ndarray | None = None) -> ReliabilityResult:
        """Compute summary statistics, optionally validated against an error map."""
        total = float(self.array.size)
        result = ReliabilityResult(
            uncertainty_threshold_low=self.low_threshold,
            uncertainty_threshold_high=self.high_threshold,
            total_pixels=int(total),
        )

        result.high_reliability_pixels = int(self.high_mask.sum())
        result.medium_reliability_pixels = int(self.medium_mask.sum())
        result.low_reliability_pixels = int(self.low_mask.sum())

        result.high_reliability_fraction = result.high_reliability_pixels / total
        result.medium_reliability_fraction = result.medium_reliability_pixels / total
        result.low_reliability_fraction = result.low_reliability_pixels / total

        if error_map is not None:
            error_map = np.asarray(error_map, dtype=np.float32)
            if error_map.shape != self.array.shape:
                from scipy.ndimage import zoom
                scale_y = self.array.shape[0] / error_map.shape[0]
                scale_x = self.array.shape[1] / error_map.shape[1]
                error_map = zoom(error_map, (scale_y, scale_x), order=1).astype(np.float32)

            result.validated_against_hr = True
            result.high_reliability_mean_error = (
                float(error_map[self.high_mask].mean()) if self.high_mask.any() else None
            )
            result.medium_reliability_mean_error = (
                float(error_map[self.medium_mask].mean()) if self.medium_mask.any() else None
            )
            result.low_reliability_mean_error = (
                float(error_map[self.low_mask].mean()) if self.low_mask.any() else None
            )

            # Correlation between uncertainty (channel mean) and error
            u_flat = self.uncertainty_map.ravel()
            e_flat = error_map.ravel()
            valid = np.isfinite(u_flat) & np.isfinite(e_flat)
            if valid.sum() > 1:
                result.reliability_error_correlation = float(
                    np.corrcoef(u_flat[valid], e_flat[valid])[0, 1]
                )

            result.validated_note = (
                "Reliability categories have been validated against an actual "
                "reconstruction error map.  Higher mean error in low-reliability "
                "regions supports the utility of the uncertainty-based reliability "
                "map as a spatial indicator of reconstruction quality."
            )

        return result

    def save_npy(self, path: str | Path) -> None:
        """Save the reliability map array as .npy."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.array)

    def save_png(self, path: str | Path) -> None:
        """Save the reliability map as an RGB PNG."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            from matplotlib.patches import Patch
        except ImportError:
            raise ImportError("matplotlib is required to save reliability PNG.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        cmap = ListedColormap(
            [
                [c / 255.0 for c in self.COLORS[0]],  # HIGH
                [c / 255.0 for c in self.COLORS[1]],  # MEDIUM
                [c / 255.0 for c in self.COLORS[2]],  # LOW
            ]
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(self.array, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(
            "PixelSight — SR Reconstruction Reliability Map\n"
            "(based on stochastic diffusion uncertainty)",
            fontsize=12,
        )
        ax.axis("off")

        legend_elements = [
            Patch(facecolor=[c / 255.0 for c in self.COLORS[0]], label="HIGH reliability"),
            Patch(facecolor=[c / 255.0 for c in self.COLORS[1]], label="MEDIUM reliability"),
            Patch(facecolor=[c / 255.0 for c in self.COLORS[2]], label="LOW reliability\n(reconstruction-risk region)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def save_geotiff(self, path: str | Path, meta: dict | None = None) -> Path:
        """Save the reliability map as a uint8 GeoTIFF."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import rasterio
            if meta is not None:
                out_meta = meta.copy()
                out_meta.update({
                    "dtype": "uint8",
                    "count": 1,
                    "height": self.array.shape[0],
                    "width": self.array.shape[1],
                })
                with rasterio.open(path, "w", **out_meta) as dst:
                    dst.write(self.array[np.newaxis])
            else:
                with rasterio.open(
                    path,
                    "w",
                    driver="GTiff",
                    height=self.array.shape[0],
                    width=self.array.shape[1],
                    count=1,
                    dtype="uint8",
                ) as dst:
                    dst.write(self.array[np.newaxis])
        except ImportError:
            # Fallback to npy if rasterio not installed
            np.save(path.with_suffix(".npy"), self.array)
        return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_reliability_map(
    uncertainty_map: np.ndarray,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    error_map: np.ndarray | None = None,
) -> tuple[ReliabilityMap, ReliabilityResult]:
    """Compute the reliability map and result summary.

    Parameters
    ----------
    uncertainty_map : np.ndarray
        Per-pixel reconstruction uncertainty (std across N diffusion runs).
    low_threshold : float, optional
        Below this value = HIGH reliability.  Defaults to p33.
    high_threshold : float, optional
        Above this value = LOW reliability.  Defaults to p67.
    error_map : np.ndarray, optional
        Per-pixel reconstruction error.  When provided, the reliability
        categories are validated against actual reconstruction error.

    Returns
    -------
    (ReliabilityMap, ReliabilityResult)
    """
    rmap = ReliabilityMap(
        uncertainty_map,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    result = rmap.to_result(error_map=error_map)
    return rmap, result
