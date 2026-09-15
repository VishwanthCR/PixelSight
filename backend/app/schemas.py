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


class ResultResponse(BaseModel):
    job_id: str
    status: str
    outputs: dict[str, Any]
