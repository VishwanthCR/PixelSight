from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class InspectionResponse(BaseModel):
    valid: bool
    compatible: bool
    filename: str
    format: str | None = None
    width: int | None = None
    height: int | None = None
    bands: int | None = None
    band_names: list[str | None] = Field(default_factory=list)
    dtype: str | None = None
    crs: str | None = None
    transform: list[float] | None = None
    resolution: list[float] | None = None
    bounds: list[float] | None = None
    nodata: list[float | None] = Field(default_factory=list)
    band_metadata: list[dict[str, Any]] = Field(default_factory=list)
    requires_preprocessing: bool = False
    estimated_output_dimensions: list[int] | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PreprocessResponse(BaseModel):
    job_id: str
    inspection: InspectionResponse
    operations: list[dict[str, Any]]
    output_file: str


class JobResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    progress: float
    error: str | None = None
    outputs: dict[str, Any] = Field(default_factory=dict)
    application: str = "research"
    use_case: str = "research"
    core_engine: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    report: dict[str, Any] | None = None


class ResultResponse(BaseModel):
    job_id: str
    status: str
    outputs: dict[str, Any]
    application: str = "research"
    use_case: str = "research"
    core_engine: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    report: dict[str, Any] | None = None


class ApplicationDefinition(BaseModel):
    id: str
    name: str
    description: str
    scientific_status: str
    bands: dict[str, Any] = Field(default_factory=dict)
    mode: str = "single_image"


class CapabilitiesResponse(BaseModel):
    version: str
    model: str
    scale: int
    sampling_steps: int
    tile_size: int
    supported_bands: list[str]
    applications: list[ApplicationDefinition]


class CopernicusCapabilitiesResponse(BaseModel):
    configured: bool
    collections: list[str]
    bands: list[str]
    native_resolution_m: float
    sr_scale_factor: int
    sr_resolution_label: str
    aoi_limits: dict[str, float]
    supported_country: str = "INDIA"


class CopernicusSearchRequest(BaseModel):
    aoi: Any
    start_date: str
    end_date: str
    max_cloud_cover: float = 20.0
    limit: int = 20


class CopernicusSearchResponse(BaseModel):
    scenes: list[dict[str, Any]]
    total_found: int
    best_scene: dict[str, Any] | None = None
    aoi_summary: dict[str, Any]


class CopernicusEstimateRequest(BaseModel):
    aoi: Any


class CopernicusEstimateResponse(BaseModel):
    valid: bool
    inside_india: bool = True
    supported_country: str = "INDIA"
    bbox: list[float]
    geojson: dict[str, Any]
    dimensions_km: dict[str, float]
    area_sqkm: float
    category: str
    native_10m: dict[str, Any]
    super_resolution_2_5m: dict[str, Any]
    tiles: dict[str, Any]


class CopernicusAcquireRequest(BaseModel):
    aoi: Any
    scene_id: str | None = None
    date: str | None = None
    date_range: tuple[str, str] | None = None


class CopernicusApplicationJobRequest(BaseModel):
    aoi: Any
    scene_id: str | None = None
    date: str | None = None
    # Temporal disaster fields
    pre_scene_id: str | None = None
    pre_date: str | None = None
    post_scene_id: str | None = None
    post_date: str | None = None


class ReferenceDiscoveryRequest(BaseModel):
    aoi: Any | None = None
    geometry: dict[str, Any] | None = None
    date: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_preference: str | None = None


class ReferenceDiscoveryResponse(BaseModel):
    available: bool
    source: str | None = None
    reference_id: str | None = None
    reference_path: str | None = None
    resolution_m: float | None = None
    spatial_overlap: float = 0.0
    temporal_difference_days: int | None = None
    temporal_match_status: str = "UNKNOWN"
    spectral_compatibility: str = "INCOMPATIBLE"
    match_status: str = "UNAVAILABLE"
    selection_reason: str
    limitations: list[str] = Field(default_factory=list)
    eligible_metrics: list[str] = Field(default_factory=list)
    ineligible_metrics: dict[str, str] = Field(default_factory=dict)
    band_mapping: dict[str, int | None] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    candidates_searched: int = 0
    candidate_summary: list[dict[str, Any]] = Field(default_factory=list)


class GroundTruthCreateRequest(BaseModel):
    job_id: str
    aoi: list[float] | None = None
    annotator: str | None = "Expert Annotator"
    notes: str | None = None


class GroundTruthAnnotationRequest(BaseModel):
    geojson: dict[str, Any]
    annotator: str | None = "Expert Annotator"
    notes: str | None = None


class GroundTruthReviewRequest(BaseModel):
    status: str  # "draft" | "review" | "validated"
    reviewer: str | None = None


class GroundTruthRasterizeRequest(BaseModel):
    evaluation_grid: str = "2.5m"
    auto_validate: bool = False

