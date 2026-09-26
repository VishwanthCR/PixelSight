"""
PixelSight Reference Package
============================
Automatic discovery, matching, download caching, alignment, and evaluation
for external high-resolution satellite imagery.
"""

from .registry import ReferenceRegistry, ReferenceSource, ReferenceTile
from .spatial_match import SpatialMatcher, SpatialMatchResult
from .temporal_match import TemporalMatcher, TemporalMatchResult
from .spectral_match import SpectralMatcher, SpectralMatchResult
from .downloader import ReferenceDownloader
from .alignment import ReferenceAligner, AlignmentReport
from .evaluator import ReferenceEvaluator
from .discovery import ReferenceDiscoveryService, ReferenceDiscoveryResult

__all__ = [
    "ReferenceRegistry",
    "ReferenceSource",
    "ReferenceTile",
    "SpatialMatcher",
    "SpatialMatchResult",
    "TemporalMatcher",
    "TemporalMatchResult",
    "SpectralMatcher",
    "SpectralMatchResult",
    "ReferenceDownloader",
    "ReferenceAligner",
    "AlignmentReport",
    "ReferenceEvaluator",
    "ReferenceDiscoveryService",
    "ReferenceDiscoveryResult",
]
