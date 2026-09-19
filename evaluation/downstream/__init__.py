"""Downstream evaluation sub-package."""
from evaluation.downstream.segmentation import (
    segmentation_metrics,
    SegmentationResult,
    aggregate_segmentation,
)
from evaluation.downstream.ndvi import (
    compute_ndvi,
    ndvi_comparison,
    NDVIResult,
)
from evaluation.downstream.urban import (
    urban_spatial_analysis,
    UrbanSpatialResult,
    gradient_energy,
)
from evaluation.downstream.crop import (
    evaluate_crop_monitoring,
    CropAnalysisResult,
)
from evaluation.downstream.disaster import (
    evaluate_bitemporal_change,
    DisasterAnalysisResult,
)
from evaluation.downstream.srm_conservation import (
    SubPixelMappingValidator,
    SubPixelConservationResult,
)

__all__ = [
    "segmentation_metrics",
    "SegmentationResult",
    "aggregate_segmentation",
    "compute_ndvi",
    "ndvi_comparison",
    "NDVIResult",
    "urban_spatial_analysis",
    "UrbanSpatialResult",
    "gradient_energy",
    "evaluate_crop_monitoring",
    "CropAnalysisResult",
    "evaluate_bitemporal_change",
    "DisasterAnalysisResult",
    "SubPixelMappingValidator",
    "SubPixelConservationResult",
]


