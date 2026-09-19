"""
PixelSight Reference Subsystem
==============================
Handles discovery, spatial/CRS alignment, and synthetic paired benchmark generation.

Hierarchy:
1. REAL_HR_REFERENCE ("real_hr"): Independently observed high-resolution satellite imagery.
2. EXTERNAL_HR_REFERENCE ("external_hr"): External regional HR imagery needing reprojection/alignment.
3. SYNTHETIC_HR_REFERENCE ("synthetic_hr"): Controlled degradation benchmark with known target.
"""

from evaluation.reference.discovery import ReferenceDiscovery, DiscoveredReference
from evaluation.reference.alignment import ReferenceAligner, AlignmentReport
from evaluation.reference.synthetic import SyntheticBenchmarkGenerator, DegradationConfig

__all__ = [
    "ReferenceDiscovery",
    "DiscoveredReference",
    "ReferenceAligner",
    "AlignmentReport",
    "SyntheticBenchmarkGenerator",
    "DegradationConfig",
]
