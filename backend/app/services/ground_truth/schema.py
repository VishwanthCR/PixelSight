"""
Ground Truth Schemas and Data Models
=====================================
Defines formal schemas for expert vector annotations, review workflow,
common evaluation grids, and scientific label status types.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LabelStatus(str, Enum):
    """Scientific label source provenance categories."""
    GROUND_TRUTH = "GROUND_TRUTH"         # Independently validated expert ground truth
    REFERENCE_LABELS = "REFERENCE_LABELS" # External land-cover labels (unverified)
    PROXY_LABELS = "PROXY_LABELS"         # Evaluation proxy (e.g., ESA WorldCover)
    NONE = "NONE"                         # No labels available


class ValidationStatus(str, Enum):
    """Editorial lifecycle of vector ground-truth annotations."""
    DRAFT = "draft"
    REVIEW = "review"
    VALIDATED = "validated"


# Canonical 7 Land-Cover Classes + Ignore (matching ESA WorldCover palette)
GT_CLASSES = [
    {"id": 0, "name": "Tree", "color": "#28b45a", "desc": "Trees & closed forest canopy"},
    {"id": 1, "name": "Shrubland", "color": "#78aa50", "desc": "Shrub and bush formations"},
    {"id": 2, "name": "Grassland", "color": "#aad264", "desc": "Natural herbaceous vegetation"},
    {"id": 3, "name": "Cropland", "color": "#dcbe46", "desc": "Cultivated agricultural land"},
    {"id": 4, "name": "Built-up", "color": "#d25a37", "desc": "Impervious structures & building clusters"},
    {"id": 5, "name": "Bare", "color": "#96876e", "desc": "Bare soil, sand, and rock"},
    {"id": 6, "name": "Water", "color": "#327dd2", "desc": "Permanent & seasonal water bodies"},
]
IGNORE_CLASS = {"id": 255, "name": "Ignore", "color": "#1e293b", "desc": "No-data / unclassified background"}

CLASS_BY_ID = {c["id"]: c for c in GT_CLASSES}
CLASS_BY_ID[255] = IGNORE_CLASS
CLASS_BY_NAME = {c["name"].lower(): c for c in GT_CLASSES}
CLASS_BY_NAME["ignore"] = IGNORE_CLASS


class AnnotationFeatureProperties(BaseModel):
    class_id: int
    class_name: str
    annotator: Optional[str] = "Expert Annotator"
    created_at: Optional[str] = None
    notes: Optional[str] = None


class AnnotationFeature(BaseModel):
    type: str = "Feature"
    geometry: Dict[str, Any]
    properties: AnnotationFeatureProperties


class AnnotationCollection(BaseModel):
    type: str = "FeatureCollection"
    features: List[AnnotationFeature] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    type: str  # "error" | "warning" | "info"
    code: str
    message: str
    feature_index: Optional[int] = None
    class_name: Optional[str] = None


class ValidationReport(BaseModel):
    is_valid: bool
    status: ValidationStatus = ValidationStatus.DRAFT
    total_features: int = 0
    covered_classes: List[str] = Field(default_factory=list)
    missing_classes: List[str] = Field(default_factory=list)
    errors: List[ValidationIssue] = Field(default_factory=list)
    warnings: List[ValidationIssue] = Field(default_factory=list)


class GroundTruthMetadata(BaseModel):
    source: str = "Expert Vector Annotation"
    annotation_method: str = "Polygon Vector Digitize"
    annotator: str = "Lead Remote Sensing Specialist"
    reviewer: Optional[str] = None
    created_at: str
    updated_at: str
    reference_imagery: str = "Aligned 2.5m HR Reference"
    reference_acquisition_date: Optional[str] = None
    aoi: Optional[List[float]] = None
    crs: str = "EPSG:4326"
    evaluation_grid: str = "2.5m"
    resampling_method: str = "nearest"
    alignment_method: str = "affine_reproject"
    class_schema: str = "ESA WorldCover 7-class + Ignore"
    coverage_percentage: float = 0.0
    ignored_percentage: float = 0.0
    class_area_statistics: Dict[str, Any] = Field(default_factory=dict)
    validation_status: ValidationStatus = ValidationStatus.DRAFT
    label_status: LabelStatus = LabelStatus.NONE
    validation_report: Optional[Dict[str, Any]] = None
