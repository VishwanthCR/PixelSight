"""
PixelSight Scientific Evaluation Package
=========================================

Uncertainty-Aware, Spectral-Spatial-Geospatial and Downstream-Task-Aware
Satellite Super-Resolution Evaluation Framework.

Sub-packages
------------
image_metrics   : Spatial, spectral, and geospatial fidelity metrics.
downstream      : Downstream task evaluation (segmentation, NDVI, urban).
uncertainty     : Uncertainty analysis and reliability mapping.
benchmarking    : Three-way benchmark and scientific result table generation.

Scientific Conventions
----------------------
* HR-reference metrics (PSNR, SSIM, SAM vs reference) are ONLY reported when a
  genuine high-resolution reference exists.  When unavailable the value is None
  and the report field states "High-resolution reference unavailable."
* Input-to-SR consistency diagnostics are labeled explicitly as consistency
  diagnostics, NOT as reconstruction accuracy.
* The LDSR-S2 uncertainty mechanism is stochastic diffusion variation across
  independent random seeds, NOT MC-dropout.
* The SR output is described as a "4× super-resolved representation (~2.5 m
  equivalent)", never as "true observed 2.5 m imagery".
* Negative findings are valid research findings and must be reported as-is.
"""

from evaluation.image_metrics.results import EvaluationResult
from evaluation.image_metrics.spatial import spatial_fidelity
from evaluation.image_metrics.spectral import spectral_fidelity
from evaluation.image_metrics.geospatial import geospatial_fidelity

__all__ = [
    "EvaluationResult",
    "spatial_fidelity",
    "spectral_fidelity",
    "geospatial_fidelity",
]
