"""
Structured evaluation result containers.

Design principles
-----------------
* All three fidelity dimensions (spatial, spectral, geospatial) are recorded
  together so that evaluations are always complete and self-documenting.
* When a genuine HR reference is absent, HR-reference metrics are stored as
  None (serialised as null in JSON) and the report labels them explicitly as
  unavailable.  They are NEVER computed against an upsampled LR proxy.
* Consistency diagnostics (e.g. input-to-SR statistics) are stored separately
  from reference-based metrics and labelled accordingly.
* Every result carries provenance metadata so that experiments are reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

@dataclass
class ExperimentProvenance:
    """Metadata needed to reproduce an evaluation."""

    model: str = "LDSR-S2"
    checkpoint: str = "opensr-ldsrs2_v1_0_0.ckpt"
    sampling_steps: int = 100
    scale: int = 4
    bands: list[str] = field(default_factory=lambda: ["B02", "B03", "B04", "B08"])
    input_scene: str = ""
    crs: str = ""
    input_resolution_m: float = 10.0
    output_resolution_m: float = 2.5
    random_seed: str | None = None
    experiment_config: str = ""
    hr_reference_available: bool = False
    hr_reference_source: str | None = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-dimension fidelity results
# ---------------------------------------------------------------------------

@dataclass
class SpatialMetrics:
    """Spatial fidelity metrics.

    HR-reference metrics (psnr, ssim, mae_hr, rmse_hr) are None when no
    genuine high-resolution reference exists.
    Gradient energy is a scene-level consistency diagnostic available without
    a reference.
    """

    # Reference-based (None when HR reference unavailable)
    psnr: float | None = None
    ssim: float | None = None
    mae_hr: float | None = None
    rmse_hr: float | None = None
    spatial_correlation_sr_vs_ref: float | None = None
    edge_preservation_index: float | None = None

    # Consistency diagnostics (do not require HR reference)
    gradient_energy_input: float | None = None
    gradient_energy_sr: float | None = None
    gradient_energy_bicubic: float | None = None
    gradient_energy_sr_vs_bicubic_ratio: float | None = None
    high_frequency_energy_ratio: float | None = None
    spatial_correlation_sr_vs_bicubic: float | None = None
    sharpness_score: float | None = None

    hr_reference_available: bool = False
    diagnostic_note: str = (
        "Spatial metrics computed without a genuine HR reference are "
        "input-output consistency diagnostics, NOT reconstruction accuracy."
    )


@dataclass
class SpectralMetrics:
    """Spectral fidelity metrics.

    SAM, per-band MAE/RMSE, and NDVI MAE/RMSE vs HR reference are None when
    no genuine HR reference exists.
    NDVI statistics (mean, std) and inter-method comparisons are always
    available as consistency diagnostics.
    """

    # Reference-based (None when HR reference unavailable)
    sam_degrees: float | None = None
    per_band_mae_hr: dict[str, float] | None = None
    per_band_rmse_hr: dict[str, float] | None = None
    per_band_psnr_hr: dict[str, float] | None = None
    per_band_correlation_hr: dict[str, float] | None = None
    ndvi_mae_vs_native: float | None = None
    ndvi_rmse_vs_native: float | None = None
    ndvi_mae_vs_hr: float | None = None
    ndvi_rmse_vs_hr: float | None = None
    ndvi_distribution_shift_hr: float | None = None

    # Consistency diagnostics
    ndvi_mean_native: float | None = None
    ndvi_std_native: float | None = None
    ndvi_mean_bicubic: float | None = None
    ndvi_std_bicubic: float | None = None
    ndvi_mean_sr: float | None = None
    ndvi_std_sr: float | None = None
    ndvi_mae_bicubic_vs_native: float | None = None
    ndvi_rmse_bicubic_vs_native: float | None = None
    ndvi_mae_sr_vs_native: float | None = None
    ndvi_rmse_sr_vs_native: float | None = None

    hr_reference_available: bool = False
    diagnostic_note: str = (
        "Spectral metrics computed without a genuine HR reference are "
        "input-output consistency diagnostics, NOT reconstruction accuracy."
    )


@dataclass
class GeospatialMetrics:
    """Geospatial integrity checks.

    These do not require an HR reference — they verify that the SR output
    preserves the input CRS, affine transform, spatial extent, and band order.
    """

    crs_preserved: bool | None = None
    crs_input: str | None = None
    crs_output: str | None = None
    affine_correct: bool | None = None
    spatial_extent_match: bool | None = None
    extent_delta_m: float | None = None
    pixel_alignment: bool | None = None
    resolution_metadata_correct: bool | None = None
    output_resolution_m: float | None = None
    band_count_correct: bool | None = None
    band_count_output: int | None = None
    nodata_handled: bool | None = None
    orientation_correct: bool | None = None
    all_checks_passed: bool = False
    failed_checks: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Downstream task results
# ---------------------------------------------------------------------------

@dataclass
class SegmentationResult:
    """Per-class and aggregate segmentation metrics."""

    # Aggregate
    pixel_accuracy: float | None = None
    mean_iou: float | None = None
    mean_dice: float | None = None
    mean_precision: float | None = None
    mean_recall: float | None = None

    # Per-class
    per_class_iou: dict[str, float] = field(default_factory=dict)
    per_class_dice: dict[str, float] = field(default_factory=dict)
    per_class_precision: dict[str, float] = field(default_factory=dict)
    per_class_recall: dict[str, float] = field(default_factory=dict)

    # Provenance
    representation: str = ""       # "native_10m" | "bicubic_2p5m" | "pixelsight_2p5m"
    label_type: str = ""            # "worldcover_10m_proxy" | "genuine_2p5m"
    label_is_proxy: bool = True
    methodological_note: str = ""


@dataclass
class NDVIResult:
    """NDVI consistency comparison across three representations."""

    # Per-representation statistics
    native_mean: float | None = None
    native_std: float | None = None
    bicubic_mean: float | None = None
    bicubic_std: float | None = None
    sr_mean: float | None = None
    sr_std: float | None = None

    # Pairwise consistency (bicubic and SR vs native as reference)
    bicubic_mae_vs_native: float | None = None
    bicubic_rmse_vs_native: float | None = None
    sr_mae_vs_native: float | None = None
    sr_rmse_vs_native: float | None = None

    # Improvement (negative means SR is worse)
    mae_improvement_pct: float | None = None
    rmse_improvement_pct: float | None = None

    diagnostic_note: str = (
        "Native 10 m NDVI is used as the reference baseline. "
        "These comparisons measure spectral consistency, not reconstruction "
        "accuracy against a genuine HR reference."
    )


@dataclass
class UrbanSpatialResult:
    """Urban spatial analysis across three representations."""

    gradient_energy_native: float | None = None
    gradient_energy_bicubic: float | None = None
    gradient_energy_sr: float | None = None
    sr_vs_bicubic_ratio: float | None = None
    sr_vs_bicubic_pct: float | None = None
    sr_vs_native_ratio: float | None = None

    diagnostic_note: str = (
        "Gradient energy measures spatial detail richness. "
        "Higher SR gradient energy than bicubic indicates more reconstructed "
        "fine-scale structure.  It does NOT establish improved detection accuracy."
    )


# ---------------------------------------------------------------------------
# Uncertainty results
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyStatistics:
    """Descriptive statistics for the uncertainty map."""

    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    p10: float | None = None
    p25: float | None = None
    p75: float | None = None
    p90: float | None = None
    total_pixels: int | None = None

    mechanism: str = (
        "Stochastic diffusion variation: pixel-wise standard deviation across "
        "N independent LDSR-S2 runs with different random seeds. "
        "This is NOT MC-dropout uncertainty."
    )


@dataclass
class UncertaintyErrorRelation:
    """Relationship between uncertainty and reconstruction/downstream error."""

    pearson_correlation: float | None = None
    high_uncertainty_error_rate: float | None = None
    low_uncertainty_error_rate: float | None = None
    difference_pct_points: float | None = None
    relative_increase_pct: float | None = None
    error_type: str = ""  # "downstream_segmentation" | "reconstruction_mae"
    hr_reference_used: bool = False
    interpretation: str = (
        "A higher error rate in high-uncertainty regions is a useful indicator "
        "that uncertainty carries information about potentially unreliable outputs. "
        "Causality is NOT established by this correlation alone."
    )


# ---------------------------------------------------------------------------
# Reliability map result
# ---------------------------------------------------------------------------

@dataclass
class ReliabilityResult:
    """Spatial reliability assessment."""

    high_reliability_fraction: float | None = None
    medium_reliability_fraction: float | None = None
    low_reliability_fraction: float | None = None
    uncertainty_threshold_low: float | None = None
    uncertainty_threshold_high: float | None = None
    validated_against_hr: bool = False
    validation_note: str = (
        "Reliability categories are based on reconstruction uncertainty level. "
        "Low reliability means 'exercise caution' — NOT that pixels are hallucinated."
    )


# ---------------------------------------------------------------------------
# Top-level evaluation result
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Complete evaluation result for one representation/experiment.

    Every field that requires a genuine HR reference is stored as None when
    that reference is unavailable.  Consumers MUST check has_hr_reference
    before interpreting those fields.
    """

    representation: str = ""  # "native_10m" | "bicubic_2p5m" | "pixelsight_2p5m"
    provenance: ExperimentProvenance = field(default_factory=ExperimentProvenance)
    has_hr_reference: bool = False

    # Fidelity dimensions
    spatial: SpatialMetrics = field(default_factory=SpatialMetrics)
    spectral: SpectralMetrics = field(default_factory=SpectralMetrics)
    geospatial: GeospatialMetrics = field(default_factory=GeospatialMetrics)

    # Downstream tasks
    segmentation: SegmentationResult | None = None
    ndvi: NDVIResult | None = None
    urban: UrbanSpatialResult | None = None

    # Uncertainty
    uncertainty_statistics: UncertaintyStatistics | None = None
    uncertainty_error_relation: UncertaintyErrorRelation | None = None
    reliability: ReliabilityResult | None = None

    # Limitation registry
    limitations: list[str] = field(default_factory=list)
    negative_findings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write result to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=_json_default)

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationResult":
        """Load result from a JSON file."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return _from_dict(cls, data)


def _json_default(obj: Any) -> Any:
    """JSON serialisation fallback for float NaN/Inf."""
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        if obj == float("inf") or obj == float("-inf"):
            return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _from_dict(cls, data: dict) -> Any:
    """Shallow reconstruction from dict (handles nested dataclasses)."""
    import dataclasses
    if not dataclasses.is_dataclass(cls):
        return data
    kwargs = {}
    for f in dataclasses.fields(cls):
        val = data.get(f.name, None)
        origin = getattr(f.type, "__origin__", None)
        # Recurse into known dataclass fields
        if isinstance(val, dict) and dataclasses.is_dataclass(f.type):
            val = _from_dict(f.type, val)
        kwargs[f.name] = val
    return cls(**kwargs)
