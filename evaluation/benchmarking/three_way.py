"""
Controlled three-way benchmark: Native vs Bicubic vs PixelSight.

Design principles
-----------------
* The benchmark uses the same spatial region and evaluation data for all three
  representations wherever possible.
* The three representations are clearly distinguished:
    - NATIVE_10M      : Native Sentinel-2 10 m input — observed data.
    - BICUBIC_2P5M    : Bicubic interpolation — interpolated, no new information.
    - PIXELSIGHT_2P5M : PixelSight LDSR-S2 4× SR — generatively reconstructed.
* Bicubic is only a baseline.  It is NEVER used as an HR ground truth.
* Do NOT assume PixelSight must improve all metrics.
* Negative results (SR worse than bicubic or native) are preserved and reported.
* Full experiment provenance is recorded for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.image_metrics.spatial import spatial_fidelity, gradient_energy_only
from evaluation.image_metrics.spectral import spectral_fidelity
from evaluation.downstream.ndvi import ndvi_comparison, NDVIResult
from evaluation.downstream.urban import urban_spatial_analysis, UrbanSpatialResult
from evaluation.downstream.segmentation import SegmentationResult


# ---------------------------------------------------------------------------
# Representation labels
# ---------------------------------------------------------------------------

NATIVE = "native_10m"
BICUBIC = "bicubic_2p5m"
PIXELSIGHT = "pixelsight_ldsr_s2_2p5m"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkProvenance:
    """Reproducibility metadata for the benchmark."""

    model: str = "LDSR-S2"
    checkpoint: str = "opensr-ldsrs2_v1_0_0.ckpt"
    sampling_steps: int = 100
    scale: int = 4
    bands: list[str] = field(default_factory=lambda: ["B02", "B03", "B04", "B08"])
    input_scene: str = ""
    crs: str = ""
    input_resolution_m: float = 10.0
    output_resolution_m: float = 2.5
    n_uncertainty_samples: int = 5
    random_seed: str = "0,1,2,3,4"
    hr_reference_available: bool = False
    hr_reference_source: str | None = None
    data_sources: dict[str, str] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Benchmark result containers
# ---------------------------------------------------------------------------

@dataclass
class RepresentationMetrics:
    """Metrics for a single representation."""

    label: str = ""
    # Spatial
    psnr: float | None = None          # None when no HR reference
    ssim: float | None = None          # None when no HR reference
    mae_hr: float | None = None        # None when no HR reference
    rmse_hr: float | None = None       # None when no HR reference
    gradient_energy: float | None = None

    # Spectral
    sam_degrees: float | None = None   # None when no HR reference
    ndvi_mean: float | None = None
    ndvi_std: float | None = None
    ndvi_mae_vs_native: float | None = None
    ndvi_rmse_vs_native: float | None = None

    # Segmentation
    pixel_accuracy: float | None = None
    mean_iou: float | None = None
    mean_dice: float | None = None
    mean_precision: float | None = None
    mean_recall: float | None = None
    segmentation_label_is_proxy: bool = True
    segmentation_methodological_note: str = ""


@dataclass
class BenchmarkResult:
    """Complete three-way benchmark result."""

    provenance: BenchmarkProvenance = field(default_factory=BenchmarkProvenance)

    native: RepresentationMetrics = field(
        default_factory=lambda: RepresentationMetrics(label=NATIVE)
    )
    bicubic: RepresentationMetrics = field(
        default_factory=lambda: RepresentationMetrics(label=BICUBIC)
    )
    pixelsight: RepresentationMetrics = field(
        default_factory=lambda: RepresentationMetrics(label=PIXELSIGHT)
    )

    ndvi: NDVIResult | None = None
    urban: UrbanSpatialResult | None = None

    # Scientific documentation
    hr_reference_note: str = "High-resolution reference unavailable."
    negative_findings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                self.to_dict(), fh, indent=2,
                default=lambda x: None if (isinstance(x, float) and not np.isfinite(x)) else x
            )


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class ThreeWayBenchmark:
    """Run the controlled Native vs Bicubic vs PixelSight benchmark.

    This class loads pre-computed arrays and runs all evaluation metrics
    in a controlled, reproducible way.

    Parameters
    ----------
    native_array : np.ndarray, shape (C, H_n, W_n)
        Native 10 m Sentinel-2 imagery (B02/B03/B04/B08).
    sr_array : np.ndarray, shape (C, H_s, W_s)
        PixelSight LDSR-S2 4× SR output.
    bicubic_array : np.ndarray, shape (C, H_s, W_s), optional
        Bicubic interpolation.  If None, computed from native_array.
    hr_reference : np.ndarray, shape (C, H_s, W_s), optional
        Genuine independent HR reference.  Do NOT pass LR upscale proxies.
    provenance : BenchmarkProvenance, optional
        Experiment metadata.
    """

    def __init__(
        self,
        native_array: np.ndarray,
        sr_array: np.ndarray,
        bicubic_array: np.ndarray | None = None,
        hr_reference: np.ndarray | None = None,
        hr_reference_available: bool = False,
        provenance: BenchmarkProvenance | None = None,
    ) -> None:
        self.native = np.asarray(native_array, dtype=np.float32)
        if self.native.ndim == 3 and self.native.shape[2] in (3, 4) and self.native.shape[0] not in (3, 4):
            self.native = np.transpose(self.native, (2, 0, 1))
        self.sr = np.asarray(sr_array, dtype=np.float32)
        if self.sr.ndim == 3 and self.sr.shape[2] in (3, 4) and self.sr.shape[0] not in (3, 4):
            self.sr = np.transpose(self.sr, (2, 0, 1))
        self.hr_reference = hr_reference
        self.hr_reference_available = hr_reference_available
        self.provenance = provenance or BenchmarkProvenance()

        if bicubic_array is not None:
            self.bicubic = np.asarray(bicubic_array, dtype=np.float32)
            if self.bicubic.ndim == 3 and self.bicubic.shape[2] in (3, 4) and self.bicubic.shape[0] not in (3, 4):
                self.bicubic = np.transpose(self.bicubic, (2, 0, 1))
        else:
            self.bicubic = self._compute_bicubic()

    def _compute_bicubic(self) -> np.ndarray:
        """Bicubic interpolation of native to SR resolution."""
        from scipy.ndimage import zoom
        scale = self.sr.shape[-1] / self.native.shape[-1]
        result = np.zeros_like(self.sr)
        for b in range(self.native.shape[0]):
            result[b] = zoom(self.native[b], scale, order=3)
        return np.clip(result, 0.0, 1.0)

    def run(
        self,
        native_segmentation: SegmentationResult | None = None,
        bicubic_segmentation: SegmentationResult | None = None,
        sr_segmentation: SegmentationResult | None = None,
    ) -> BenchmarkResult:
        """Execute the full benchmark and return a BenchmarkResult.

        Parameters
        ----------
        native_segmentation, bicubic_segmentation, sr_segmentation :
            Pre-computed segmentation results.  These are optional and used
            to populate the benchmark result tables.
        """
        result = BenchmarkResult(provenance=self.provenance)

        if not self.hr_reference_available:
            result.hr_reference_note = (
                "High-resolution reference unavailable.  "
                "HR-reference metrics (PSNR, SSIM, SAM, MAE, RMSE vs reference) "
                "cannot be computed.  Reporting null for these fields."
            )

        # --- Spatial metrics ---
        spatial_sr = spatial_fidelity(
            self.sr,
            input_array=self.native,
            bicubic_array=self.bicubic,
            hr_reference=self.hr_reference if self.hr_reference_available else None,
            hr_reference_available=self.hr_reference_available,
        )
        result.pixelsight.psnr = spatial_sr.psnr
        result.pixelsight.ssim = spatial_sr.ssim
        result.pixelsight.mae_hr = spatial_sr.mae_hr
        result.pixelsight.rmse_hr = spatial_sr.rmse_hr
        result.pixelsight.gradient_energy = spatial_sr.gradient_energy_sr

        result.native.gradient_energy = gradient_energy_only(self.native)
        result.bicubic.gradient_energy = gradient_energy_only(self.bicubic)

        if self.hr_reference_available and self.hr_reference is not None:
            spatial_bic = spatial_fidelity(
                self.bicubic,
                hr_reference=self.hr_reference,
                hr_reference_available=True,
            )
            result.bicubic.psnr = spatial_bic.psnr
            result.bicubic.ssim = spatial_bic.ssim
            result.bicubic.mae_hr = spatial_bic.mae_hr
            result.bicubic.rmse_hr = spatial_bic.rmse_hr

        # --- NDVI ---
        ndvi_result = ndvi_comparison(
            self.native,
            self.sr,
            bicubic=self.bicubic,
            reference_is_genuine_hr=self.hr_reference_available,
        )
        result.ndvi = ndvi_result

        result.native.ndvi_mean = ndvi_result.native_mean
        result.native.ndvi_std = ndvi_result.native_std
        result.bicubic.ndvi_mean = ndvi_result.bicubic_mean
        result.bicubic.ndvi_std = ndvi_result.bicubic_std
        result.bicubic.ndvi_mae_vs_native = ndvi_result.bicubic_mae_vs_native
        result.bicubic.ndvi_rmse_vs_native = ndvi_result.bicubic_rmse_vs_native
        result.pixelsight.ndvi_mean = ndvi_result.sr_mean
        result.pixelsight.ndvi_std = ndvi_result.sr_std
        result.pixelsight.ndvi_mae_vs_native = ndvi_result.sr_mae_vs_native
        result.pixelsight.ndvi_rmse_vs_native = ndvi_result.sr_rmse_vs_native

        # --- Urban spatial ---
        urban_result = urban_spatial_analysis(self.native, self.sr, self.bicubic)
        result.urban = urban_result

        # --- Segmentation (from pre-computed results) ---
        for seg, rep_result in [
            (native_segmentation, result.native),
            (bicubic_segmentation, result.bicubic),
            (sr_segmentation, result.pixelsight),
        ]:
            if seg is not None:
                rep_result.pixel_accuracy = seg.pixel_accuracy
                rep_result.mean_iou = seg.mean_iou
                rep_result.mean_dice = seg.mean_dice
                rep_result.mean_precision = seg.mean_precision
                rep_result.mean_recall = seg.mean_recall
                rep_result.segmentation_label_is_proxy = seg.label_is_proxy
                rep_result.segmentation_methodological_note = seg.methodological_note

        # --- Negative findings ---
        result.negative_findings = _identify_negative_findings(result)
        result.limitations = self.provenance.limitations

        return result


def _identify_negative_findings(result: BenchmarkResult) -> list[str]:
    """Identify and document any negative findings automatically."""
    findings = []

    # Segmentation: check if SR is worse than native
    if (result.pixelsight.mean_iou is not None
            and result.native.mean_iou is not None
            and result.pixelsight.mean_iou < result.native.mean_iou):
        findings.append(
            f"PixelSight SR did not improve downstream segmentation mIoU "
            f"(native: {result.native.mean_iou:.4f}, "
            f"PixelSight: {result.pixelsight.mean_iou:.4f}).  "
            f"This is reported as a valid research finding."
        )

    # Segmentation: check if SR is worse than bicubic
    if (result.pixelsight.mean_iou is not None
            and result.bicubic.mean_iou is not None
            and result.pixelsight.mean_iou < result.bicubic.mean_iou):
        findings.append(
            f"PixelSight SR did not improve downstream segmentation mIoU "
            f"compared to bicubic baseline "
            f"(bicubic: {result.bicubic.mean_iou:.4f}, "
            f"PixelSight: {result.pixelsight.mean_iou:.4f})."
        )

    return findings
