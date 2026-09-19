import io
import json
from pathlib import Path
import tempfile
import time
from typing import Annotated

import numpy as np
import rasterio
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import FileResponse
from PIL import Image
from rasterio.transform import from_origin
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from backend.app.config import (
    ALLOWED_SUFFIXES,
    AOI_INTERACTIVE_MAX_AREA_SQKM,
    AOI_MAX_AREA_SQKM,
    MAX_UPLOAD_BYTES,
    RESULTS_ROOT,
)
from backend.app.schemas import (
    CapabilitiesResponse,
    CopernicusAcquireRequest,
    CopernicusApplicationJobRequest,
    CopernicusCapabilitiesResponse,
    CopernicusEstimateRequest,
    CopernicusEstimateResponse,
    CopernicusSearchRequest,
    CopernicusSearchResponse,
    HealthResponse,
    InspectionResponse,
    JobResponse,
    PreprocessResponse,
    ResultResponse,
)
from backend.app.services.aoi import AOIValidationError, validate_and_estimate_aoi
from backend.app.services.app_registry import registry
from backend.app.services.copernicus_auth import CopernicusAuthError, copernicus_auth
from backend.app.services.copernicus_catalog import CopernicusCatalogError, copernicus_catalog
from backend.app.services.copernicus_processing import CopernicusProcessingError, copernicus_processing
from backend.app.services.core_engine import PixelSightEngine
from backend.app.services.crop import run_crop_analysis
from backend.app.services.disaster import align_temporal_pair, run_disaster_analysis
from backend.app.services.geocoding import geocode_location
from backend.app.services.jobs import JobStore
from backend.app.services.raster import inspect_raster, preprocess_raster
from backend.app.services.reporting import write_manifest, write_report
from backend.app.services.uncertainty import generate_uncertainty_map
from backend.app.services.urban import compare_urban_analysis, run_urban_analysis, run_urban_pipeline
from backend.app.services.visualization import save_rgb_preview


app = FastAPI(title="PixelSight API", version="1.0.0")
jobs = JobStore(RESULTS_ROOT)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail="Only GeoTIFF/TIFF and common image files (.tif, .tiff, .png, .jpg, .jpeg, .bmp) are supported.",
        )
    return suffix


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    _safe_suffix(upload.filename)
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
        raise HTTPException(
            status_code=422,
            detail="RGB input is not compatible with the current Sentinel-2 LDSR-S2 model. Please provide compatible Sentinel-2 multispectral data.",
        )

    total = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Uploaded file exceeds the size limit.")
            output.write(chunk)


def _job_response(job: dict) -> JobResponse:
    fields = {key: job[key] for key in ("job_id", "status", "stage", "progress", "error", "outputs") if key in job}
    fields["application"] = job.get("application", "research")
    return JobResponse(**fields)


def _calculate_sam(prediction: np.ndarray, target: np.ndarray) -> float | None:
    pred = np.asarray(prediction, dtype=np.float32).reshape(-1, prediction.shape[-1])
    true = np.asarray(target, dtype=np.float32).reshape(-1, target.shape[-1])
    pred_norm = np.linalg.norm(pred, axis=1)
    true_norm = np.linalg.norm(true, axis=1)
    valid = (pred_norm > 1e-8) & (true_norm > 1e-8)
    if not np.any(valid):
        return None
    pred = pred[valid]
    true = true[valid]
    pred_norm = np.linalg.norm(pred, axis=1)
    true_norm = np.linalg.norm(true, axis=1)
    cosine = np.sum(pred * true, axis=1) / np.maximum(pred_norm * true_norm, 1e-12)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.mean(np.arccos(cosine))))


def _resample_reference_to_output(reference_path: Path, output_shape: tuple[int, int]) -> np.ndarray:
    with rasterio.open(reference_path) as src:
        reference = src.read().astype(np.float32)
    if reference.size == 0:
        raise ValueError("Reference image contains no pixel values.")
    if np.nanmax(reference) > 1.5:
        reference = reference / 10000.0
    reference = np.clip(reference, 0.0, 1.0)
    if reference.shape[0] > 4:
        reference = reference[:4]
    bands = []
    for band in reference:
        image = Image.fromarray(np.clip(band * 255.0, 0, 255).astype(np.uint8), mode="L")
        resized = image.resize((output_shape[1], output_shape[0]), Image.BICUBIC)
        bands.append(np.asarray(resized, dtype=np.float32) / 255.0)
    return np.stack(bands, axis=-1)


def _evaluate_reference(sr_path: Path, reference_path: Path) -> dict:
    try:
        with rasterio.open(sr_path) as sr_src:
            sr = np.moveaxis(sr_src.read(), 0, -1).astype(np.float32)
        if np.nanmax(sr) > 1.5:
            sr = sr / 10000.0
        sr = np.clip(sr, 0.0, 1.0)
        ref = _resample_reference_to_output(reference_path, sr.shape[:2])
        if not np.isfinite(sr).all() or not np.isfinite(ref).all():
            return {
                "status": "reference_unavailable",
                "psnr": None,
                "ssim": None,
                "sam": None,
                "reason": "Input image contains non-finite values after preprocessing.",
            }
        psnr = float(peak_signal_noise_ratio(ref, sr, data_range=1.0))
        ssim = float(structural_similarity(ref, sr, channel_axis=-1, data_range=1.0))
        sam = _calculate_sam(sr, ref)
        return {
            "status": "reference_available",
            "psnr": psnr,
            "ssim": ssim,
            "sam": sam,
            "reason": "Computed by comparing the original uploaded input against the generated 4x output after resampling the input to the SR grid.",
        }
    except Exception as exc:  # pragma: no cover - defensive branch
        return {
            "status": "reference_unavailable",
            "psnr": None,
            "ssim": None,
            "sam": None,
            "reason": f"Input-to-output evaluation failed: {exc}",
        }


# ===========================================================================
# Core API Routes
# ===========================================================================

@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="pixelsight")


@app.get("/api/v1/applications")
def get_applications() -> list[dict]:
    return registry.list_applications()


@app.get("/api/v1/capabilities", response_model=CapabilitiesResponse)
def get_capabilities() -> CapabilitiesResponse:
    return CapabilitiesResponse(**registry.get_capabilities())


@app.post("/api/v1/inspect", response_model=InspectionResponse)
async def inspect(upload: Annotated[UploadFile, File(...)]) -> InspectionResponse:
    suffix = _safe_suffix(upload.filename)
    with tempfile.TemporaryDirectory(prefix="pixelsight-inspect-") as directory:
        path = Path(directory) / f"input{suffix}"
        await _save_upload(upload, path)
        return inspect_raster(path, upload.filename)


@app.post("/api/v1/preprocess", response_model=PreprocessResponse)
async def preprocess(upload: Annotated[UploadFile, File(...)]) -> PreprocessResponse:
    suffix = _safe_suffix(upload.filename)
    job_id, job_dir = jobs.create()
    input_path = job_dir / "input" / f"original{suffix}"
    await _save_upload(upload, input_path)
    output_path = job_dir / "preprocessing" / "normalized.tif"
    try:
        inspection, operations = preprocess_raster(input_path, output_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PreprocessResponse(
        job_id=job_id,
        inspection=inspection,
        operations=operations,
        output_file="preprocessing/normalized.tif",
    )


# ===========================================================================
# Research Pipeline (Default)
# ===========================================================================

@app.post("/api/v1/process", response_model=JobResponse, status_code=202)
async def process(upload: Annotated[UploadFile, File(...)]) -> JobResponse:
    suffix = _safe_suffix(upload.filename)
    job_id, job_dir = jobs.create(application="research")
    input_path = job_dir / "input" / f"original{suffix}"
    await _save_upload(upload, input_path)

    def run() -> None:
        started = time.perf_counter()
        jobs.update(job_id, stage="inspecting", progress=5.0)
        inspection = PixelSightEngine.inspect(input_path)
        if not inspection.compatible:
            raise ValueError("Input is incompatible: " + "; ".join(inspection.errors))

        jobs.update(job_id, stage="preprocessing", progress=15.0)
        normalized_path = job_dir / "preprocessing" / "normalized.tif"
        _, preprocessing_operations = PixelSightEngine.preprocess(input_path, normalized_path)

        jobs.update(job_id, stage="super_resolution", progress=20.0)
        output_files, device = _run_super_resolution(job_id, job_dir, normalized_path)

        jobs.update(job_id, stage="uncertainty_analysis", progress=88.0)
        sr_path = job_dir / output_files["super_resolution"]
        uncertainty = PixelSightEngine.uncertainty(
            normalized_path,
            sr_path,
            job_dir / "uncertainty" / "uncertainty_map.tif",
            job_dir / "uncertainty" / "uncertainty_map.png",
        )

        urban_analysis = run_urban_analysis(
            sr_path,
            job_dir / "analysis" / "urban_planning_map.png",
            classified_output=job_dir / "analysis" / "urban_classes.tif",
            checkpoint_path=PROJECT_ROOT / "checkpoints" / "segmentation" / "unet_worldcover_best.pth",
            device=device,
        )
        input_urban_analysis = run_urban_analysis(
            normalized_path,
            job_dir / "analysis" / "input_urban_planning_map.png",
            classified_output=job_dir / "analysis" / "input_urban_classes.tif",
            checkpoint_path=PROJECT_ROOT / "checkpoints" / "segmentation" / "unet_worldcover_best.pth",
            device=device,
        )
        urban_analysis["comparison"] = compare_urban_analysis(input_urban_analysis, urban_analysis)
        jobs.update(job_id, stage="urban_analysis", progress=92.0)
        jobs.update(job_id, stage="reporting", progress=95.0)

        report_path = job_dir / "report" / "report.json"
        output_files["report_markdown"] = "report/report.md"
        output_files["report_html"] = "report/report.html"
        output_files["original"] = f"input/{input_path.name}"
        output_files["urban_planning_map"] = "analysis/urban_planning_map.png"
        output_files["urban_classification"] = "analysis/urban_classes.tif"
        output_files["uncertainty_map"] = "uncertainty/uncertainty_map.png"
        output_files["uncertainty_geotiff"] = "uncertainty/uncertainty_map.tif"

        evaluation = _evaluate_reference(sr_path, normalized_path)
        output_files["input_reference"] = "preprocessing/normalized.tif"

        report = write_report(
            report_path,
            job_id=job_id,
            input_metadata=inspection.model_dump(),
            preprocessing=preprocessing_operations,
            runtime_seconds=time.perf_counter() - started,
            device=device,
            output_files=output_files,
            evaluation=evaluation,
            urban_analysis=urban_analysis,
            uncertainty=uncertainty,
            application="research",
        )
        output_files["report"] = "report/report.json"
        write_manifest(job_dir, job_id, report)
        output_files["manifest"] = "manifest.json"
        reference_available = evaluation.get("status") == "reference_available"

        jobs.update(
            job_id,
            status="completed",
            stage="completed",
            progress=100.0,
            outputs={
                "super_resolution": True,
                "original_preview": "input/original_preview.png",
                "super_resolution_preview": "super_resolution/sr_preview.png",
                "report": "report/report.json",
                "uncertainty": uncertainty,
                "uncertainty_map": "uncertainty/uncertainty_map.png",
                "uncertainty_geotiff": "uncertainty/uncertainty_map.tif",
                "segmentation": True,
                "urban_analysis": urban_analysis,
                "urban_classification": output_files["urban_classification"],
                "urban_planning_map": output_files["urban_planning_map"],
                "input_urban_classification": "analysis/input_urban_classes.tif",
                "evaluation": reference_available,
                "runtime_seconds": report["runtime"]["seconds"],
                "limitations": report["scientific_limitations"],
                "input_reference": output_files.get("input_reference"),
                "manifest": "manifest.json",
                "report_html": "report/report.html",
                "report_markdown": "report/report.md",
            },
        )

    jobs.submit(job_id, run)
    return _job_response(jobs.get(job_id))


# ===========================================================================
# Application Pipelines: Crop Monitoring
# ===========================================================================

@app.post("/api/v1/applications/crop", response_model=JobResponse, status_code=202)
async def process_crop(upload: Annotated[UploadFile, File(...)]) -> JobResponse:
    suffix = _safe_suffix(upload.filename)
    job_id, job_dir = jobs.create(application="crop")
    input_path = job_dir / "input" / f"original{suffix}"
    await _save_upload(upload, input_path)

    def run() -> None:
        started = time.perf_counter()
        jobs.update(job_id, stage="inspecting", progress=5.0)
        inspection = PixelSightEngine.inspect(input_path)
        if not inspection.compatible:
            raise ValueError("Input is incompatible: " + "; ".join(inspection.errors))

        jobs.update(job_id, stage="preprocessing", progress=15.0)
        normalized_path = job_dir / "preprocessing" / "normalized.tif"
        _, preprocessing_operations = PixelSightEngine.preprocess(input_path, normalized_path)

        jobs.update(job_id, stage="super_resolution", progress=25.0)
        output_files, device = _run_super_resolution(job_id, job_dir, normalized_path)

        jobs.update(job_id, stage="uncertainty_analysis", progress=70.0)
        sr_path = job_dir / output_files["super_resolution"]
        uncertainty = PixelSightEngine.uncertainty(
            normalized_path,
            sr_path,
            job_dir / "uncertainty" / "uncertainty_map.tif",
            job_dir / "uncertainty" / "uncertainty_map.png",
        )

        jobs.update(job_id, stage="crop_analysis", progress=85.0)
        crop_results = run_crop_analysis(normalized_path, sr_path, job_dir)

        jobs.update(job_id, stage="reporting", progress=95.0)
        report_path = job_dir / "report" / "report.json"
        output_files.update(crop_results["outputs"])
        output_files["report_markdown"] = "report/report.md"
        output_files["report_html"] = "report/report.html"
        output_files["original"] = f"input/{input_path.name}"
        output_files["uncertainty_map"] = "uncertainty/uncertainty_map.png"
        output_files["uncertainty_geotiff"] = "uncertainty/uncertainty_map.tif"

        evaluation = _evaluate_reference(sr_path, normalized_path)
        report = write_report(
            report_path,
            job_id=job_id,
            input_metadata=inspection.model_dump(),
            preprocessing=preprocessing_operations,
            runtime_seconds=time.perf_counter() - started,
            device=device,
            output_files=output_files,
            evaluation=evaluation,
            uncertainty=uncertainty,
            application="crop",
            application_data=crop_results,
        )
        output_files["report"] = "report/report.json"
        write_manifest(job_dir, job_id, report)
        output_files["manifest"] = "manifest.json"

        jobs.update(
            job_id,
            status="completed",
            stage="completed",
            progress=100.0,
            outputs={
                "application": "crop",
                "super_resolution": True,
                "original_preview": "input/original_preview.png",
                "super_resolution_preview": "super_resolution/sr_preview.png",
                "crop": crop_results,
                "uncertainty": uncertainty,
                "uncertainty_map": "uncertainty/uncertainty_map.png",
                "uncertainty_geotiff": "uncertainty/uncertainty_map.tif",
                "manifest": "manifest.json",
                "report": "report/report.json",
                "report_html": "report/report.html",
                "report_markdown": "report/report.md",
                "runtime_seconds": report["runtime"]["seconds"],
                "limitations": report["scientific_limitations"],
            },
        )

    jobs.submit(job_id, run)
    return _job_response(jobs.get(job_id))


# ===========================================================================
# Application Pipelines: Urban Analysis
# ===========================================================================

@app.post("/api/v1/applications/urban", response_model=JobResponse, status_code=202)
async def process_urban(upload: Annotated[UploadFile, File(...)]) -> JobResponse:
    suffix = _safe_suffix(upload.filename)
    job_id, job_dir = jobs.create(application="urban")
    input_path = job_dir / "input" / f"original{suffix}"
    await _save_upload(upload, input_path)

    def run() -> None:
        started = time.perf_counter()
        jobs.update(job_id, stage="inspecting", progress=5.0)
        inspection = PixelSightEngine.inspect(input_path)
        if not inspection.compatible:
            raise ValueError("Input is incompatible: " + "; ".join(inspection.errors))

        jobs.update(job_id, stage="preprocessing", progress=15.0)
        normalized_path = job_dir / "preprocessing" / "normalized.tif"
        _, preprocessing_operations = PixelSightEngine.preprocess(input_path, normalized_path)

        jobs.update(job_id, stage="super_resolution", progress=25.0)
        output_files, device = _run_super_resolution(job_id, job_dir, normalized_path)

        jobs.update(job_id, stage="uncertainty_analysis", progress=70.0)
        sr_path = job_dir / output_files["super_resolution"]
        uncertainty = PixelSightEngine.uncertainty(
            normalized_path,
            sr_path,
            job_dir / "uncertainty" / "uncertainty_map.tif",
            job_dir / "uncertainty" / "uncertainty_map.png",
        )

        jobs.update(job_id, stage="urban_analysis", progress=85.0)
        urban_results = run_urban_pipeline(
            normalized_path,
            sr_path,
            job_dir,
            device=device,
            checkpoint_path=PROJECT_ROOT / "checkpoints" / "segmentation" / "unet_worldcover_best.pth",
        )

        jobs.update(job_id, stage="reporting", progress=95.0)
        report_path = job_dir / "report" / "report.json"
        output_files.update(urban_results["outputs"])
        output_files["report_markdown"] = "report/report.md"
        output_files["report_html"] = "report/report.html"
        output_files["original"] = f"input/{input_path.name}"
        output_files["uncertainty_map"] = "uncertainty/uncertainty_map.png"
        output_files["uncertainty_geotiff"] = "uncertainty/uncertainty_map.tif"

        evaluation = _evaluate_reference(sr_path, normalized_path)
        report = write_report(
            report_path,
            job_id=job_id,
            input_metadata=inspection.model_dump(),
            preprocessing=preprocessing_operations,
            runtime_seconds=time.perf_counter() - started,
            device=device,
            output_files=output_files,
            evaluation=evaluation,
            uncertainty=uncertainty,
            application="urban",
            application_data=urban_results,
        )
        output_files["report"] = "report/report.json"
        write_manifest(job_dir, job_id, report)
        output_files["manifest"] = "manifest.json"

        jobs.update(
            job_id,
            status="completed",
            stage="completed",
            progress=100.0,
            outputs={
                "application": "urban",
                "super_resolution": True,
                "original_preview": "input/original_preview.png",
                "super_resolution_preview": "super_resolution/sr_preview.png",
                "urban": urban_results,
                "uncertainty": uncertainty,
                "uncertainty_map": "uncertainty/uncertainty_map.png",
                "uncertainty_geotiff": "uncertainty/uncertainty_map.tif",
                "manifest": "manifest.json",
                "report": "report/report.json",
                "report_html": "report/report.html",
                "report_markdown": "report/report.md",
                "runtime_seconds": report["runtime"]["seconds"],
                "limitations": report["scientific_limitations"],
            },
        )

    jobs.submit(job_id, run)
    return _job_response(jobs.get(job_id))


# ===========================================================================
# Application Pipelines: Disaster Management (Temporal)
# ===========================================================================

@app.post("/api/v1/applications/disaster", response_model=JobResponse, status_code=202)
async def process_disaster(
    pre_event: Annotated[UploadFile, File(...)],
    post_event: Annotated[UploadFile, File(...)],
) -> JobResponse:
    pre_suffix = _safe_suffix(pre_event.filename)
    post_suffix = _safe_suffix(post_event.filename)

    job_id, job_dir = jobs.create(application="disaster")
    pre_path = job_dir / "input" / f"pre_event{pre_suffix}"
    post_path = job_dir / "input" / f"post_event{post_suffix}"
    await _save_upload(pre_event, pre_path)
    await _save_upload(post_event, post_path)

    def run() -> None:
        started = time.perf_counter()
        jobs.update(job_id, stage="inspecting", progress=5.0)
        pre_insp = PixelSightEngine.inspect(pre_path)
        post_insp = PixelSightEngine.inspect(post_path)
        if not pre_insp.compatible:
            raise ValueError("Pre-event input is incompatible: " + "; ".join(pre_insp.errors))
        if not post_insp.compatible:
            raise ValueError("Post-event input is incompatible: " + "; ".join(post_insp.errors))

        jobs.update(job_id, stage="preprocessing", progress=15.0)
        pre_norm = job_dir / "preprocessing" / "pre_normalized.tif"
        post_norm = job_dir / "preprocessing" / "post_normalized.tif"
        _, pre_ops = PixelSightEngine.preprocess(pre_path, pre_norm)
        _, post_ops = PixelSightEngine.preprocess(post_path, post_norm)

        jobs.update(job_id, stage="alignment", progress=25.0)
        aligned_pre = job_dir / "preprocessing" / "aligned_pre.tif"
        aligned_post = job_dir / "preprocessing" / "aligned_post.tif"
        alignment_info = align_temporal_pair(pre_norm, post_norm, aligned_pre, aligned_post)

        jobs.update(job_id, stage="super_resolution", progress=35.0)
        disaster_dir = job_dir / "application" / "disaster"
        sr_pre_tif = disaster_dir / "pre_event" / "sr_pre.tif"
        sr_post_tif = disaster_dir / "post_event" / "sr_post.tif"

        _, dev1 = PixelSightEngine.enhance(aligned_pre, sr_pre_tif)
        _, dev2 = PixelSightEngine.enhance(aligned_post, sr_post_tif)
        device = dev1

        jobs.update(job_id, stage="uncertainty_analysis", progress=75.0)
        pre_unc_tif = disaster_dir / "pre_event" / "unc_pre.tif"
        pre_unc_png = disaster_dir / "pre_event" / "unc_pre.png"
        post_unc_tif = disaster_dir / "post_event" / "unc_post.tif"
        post_unc_png = disaster_dir / "post_event" / "unc_post.png"

        pre_unc = PixelSightEngine.uncertainty(aligned_pre, sr_pre_tif, pre_unc_tif, pre_unc_png)
        post_unc = PixelSightEngine.uncertainty(aligned_post, sr_post_tif, post_unc_tif, post_unc_png)

        jobs.update(job_id, stage="change_detection", progress=85.0)
        disaster_results = run_disaster_analysis(
            sr_pre_tif,
            sr_post_tif,
            pre_unc_tif,
            post_unc_tif,
            job_dir,
        )

        jobs.update(job_id, stage="reporting", progress=95.0)
        report_path = job_dir / "report" / "report.json"
        output_files = dict(disaster_results["outputs"])
        output_files["report_markdown"] = "report/report.md"
        output_files["report_html"] = "report/report.html"
        output_files["pre_event"] = f"input/{pre_path.name}"
        output_files["post_event"] = f"input/{post_path.name}"

        report = write_report(
            report_path,
            job_id=job_id,
            input_metadata=pre_insp.model_dump(),
            preprocessing=pre_ops + post_ops,
            runtime_seconds=time.perf_counter() - started,
            device=device,
            output_files=output_files,
            application="disaster",
            application_data=disaster_results,
            disaster_metadata=alignment_info,
            uncertainty={"pre": pre_unc, "post": post_unc},
        )
        output_files["report"] = "report/report.json"
        write_manifest(job_dir, job_id, report)
        output_files["manifest"] = "manifest.json"

        jobs.update(
            job_id,
            status="completed",
            stage="completed",
            progress=100.0,
            outputs={
                "application": "disaster",
                "super_resolution": True,
                "disaster": disaster_results,
                "alignment": alignment_info,
                "manifest": "manifest.json",
                "report": "report/report.json",
                "report_html": "report/report.html",
                "report_markdown": "report/report.md",
                "runtime_seconds": report["runtime"]["seconds"],
                "limitations": report["scientific_limitations"],
            },
        )

    jobs.submit(job_id, run)
    return _job_response(jobs.get(job_id))


# ===========================================================================
# Copernicus Data Space Ecosystem / Sentinel Hub Endpoints
# ===========================================================================

@app.get("/api/v1/copernicus/capabilities", response_model=CopernicusCapabilitiesResponse)
def get_copernicus_capabilities() -> CopernicusCapabilitiesResponse:
    """Check Copernicus configuration, available collections, bands, and AOI limits."""
    return CopernicusCapabilitiesResponse(
        configured=copernicus_auth.is_configured(),
        collections=["sentinel-2-l2a"],
        bands=["B02", "B03", "B04", "B08"],
        native_resolution_m=10.0,
        sr_scale_factor=4,
        sr_resolution_label="4× super-resolved representation (~2.5 m equivalent)",
        aoi_limits={
            "max_area_sqkm": AOI_MAX_AREA_SQKM,
            "interactive_max_sqkm": AOI_INTERACTIVE_MAX_AREA_SQKM,
        },
    )


@app.get("/api/v1/copernicus/geocode")
def geocode_place(q: str = Query(..., min_length=2, description="Place or city name to geocode")) -> dict:
    """Geocodes a place name to coordinates and default 1.28km AOI."""
    results = geocode_location(q)
    return {
        "query": q,
        "results": results,
        "total": len(results),
    }


@app.get("/api/v1/copernicus/india-boundary")
def get_india_boundary() -> dict:
    """Returns the official GeoJSON boundary for India for map visualization."""
    from backend.app.services.aoi import _INDIA_GEOJSON_PATH
    if _INDIA_GEOJSON_PATH.exists():
        return json.loads(_INDIA_GEOJSON_PATH.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="India boundary data not found.")


@app.post("/api/v1/copernicus/estimate", response_model=CopernicusEstimateResponse)
def estimate_copernicus_aoi(req: CopernicusEstimateRequest) -> CopernicusEstimateResponse:
    """Validates AOI coordinates, calculates dimensions, area, and estimated tile counts."""
    try:
        info = validate_and_estimate_aoi(req.aoi)
        return CopernicusEstimateResponse(**info)
    except AOIValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/copernicus/search", response_model=CopernicusSearchResponse)
def search_copernicus_scenes(req: CopernicusSearchRequest) -> CopernicusSearchResponse:
    """Searches Copernicus STAC/Catalog for Sentinel-2 L2A scenes matching AOI and date range."""
    try:
        aoi_info = validate_and_estimate_aoi(req.aoi)
    except AOIValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        scenes = copernicus_catalog.search_scenes(
            aoi=req.aoi,
            start_date=req.start_date,
            end_date=req.end_date,
            max_cloud_cover=req.max_cloud_cover,
            limit=req.limit,
        )
        best = copernicus_catalog.select_best_scene(scenes)
        return CopernicusSearchResponse(
            scenes=scenes,
            total_found=len(scenes),
            best_scene=best,
            aoi_summary=aoi_info,
        )
    except CopernicusAuthError as exc:
        raise HTTPException(
            status_code=401,
            detail=f"Copernicus authentication error: {exc}",
        ) from exc
    except CopernicusCatalogError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Copernicus Catalog error: {exc}",
        ) from exc


@app.post("/api/v1/copernicus/acquire")
def acquire_copernicus_data(req: CopernicusAcquireRequest) -> dict:
    """Directly acquires 4-band GeoTIFF from Copernicus Processing API."""
    try:
        target_path, meta = copernicus_processing.acquire_aoi(
            aoi=req.aoi,
            scene_id=req.scene_id,
            date=req.date,
            date_range=req.date_range,
        )
        inspection = PixelSightEngine.inspect(target_path)
        return {
            "path": str(target_path),
            "metadata": meta,
            "inspection": inspection.model_dump(),
        }
    except AOIValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (CopernicusAuthError, CopernicusProcessingError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/v1/applications/{application}/from-copernicus", response_model=JobResponse, status_code=202)
def process_from_copernicus(application: str, req: CopernicusApplicationJobRequest) -> JobResponse:
    """Launches an asynchronous job sourced directly from Copernicus Data Space Sentinel-2 data."""
    app_id = application.lower().strip()
    if app_id not in {"crop", "urban", "disaster", "research"}:
        raise HTTPException(status_code=400, detail=f"Unsupported application '{application}'.")

    # Validate AOI
    try:
        aoi_info = validate_and_estimate_aoi(req.aoi)
    except AOIValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    job_id, job_dir = jobs.create(application=app_id)

    # -----------------------------------------------------------------------
    # Async worker for Disaster Management (Dual-Scene Temporal Acquisition)
    # -----------------------------------------------------------------------
    if app_id == "disaster":
        if not req.pre_date and not req.pre_scene_id:
            raise HTTPException(status_code=422, detail="Disaster analysis requires pre-event date or scene.")
        if not req.post_date and not req.post_scene_id:
            raise HTTPException(status_code=422, detail="Disaster analysis requires post-event date or scene.")

        def run_disaster() -> None:
            started = time.perf_counter()
            try:
                # Stage 1: PRE_EVENT_ACQUISITION
                jobs.update(job_id, stage="PRE_EVENT_ACQUISITION", progress=10.0)
                pre_path = job_dir / "input" / "pre_event.tif"
                _, pre_meta = copernicus_processing.acquire_aoi(
                    aoi=req.aoi,
                    scene_id=req.pre_scene_id,
                    date=req.pre_date,
                    destination=pre_path,
                )

                # Stage 2: POST_EVENT_ACQUISITION
                jobs.update(job_id, stage="POST_EVENT_ACQUISITION", progress=25.0)
                post_path = job_dir / "input" / "post_event.tif"
                _, post_meta = copernicus_processing.acquire_aoi(
                    aoi=req.aoi,
                    scene_id=req.post_scene_id,
                    date=req.post_date,
                    destination=post_path,
                )

                # Save acquisition records
                (job_dir / "input" / "copernicus_request.json").write_text(
                    json.dumps({
                        "aoi": aoi_info["bbox"],
                        "pre_event": pre_meta,
                        "post_event": post_meta,
                    }, indent=2),
                    encoding="utf-8",
                )

                save_rgb_preview(pre_path, job_dir / "input" / "pre_event_preview.png")
                save_rgb_preview(post_path, job_dir / "input" / "post_event_preview.png")

                # Stage 3: Inspect & Preprocess
                jobs.update(job_id, stage="PREPROCESSING", progress=35.0)
                pre_insp = PixelSightEngine.inspect(pre_path)
                post_insp = PixelSightEngine.inspect(post_path)
                pre_norm = job_dir / "preprocessing" / "pre_normalized.tif"
                post_norm = job_dir / "preprocessing" / "post_normalized.tif"
                _, pre_ops = PixelSightEngine.preprocess(pre_path, pre_norm)
                _, post_ops = PixelSightEngine.preprocess(post_path, post_norm)

                # Stage 4: ALIGNMENT
                jobs.update(job_id, stage="ALIGNMENT", progress=45.0)
                aligned_pre = job_dir / "preprocessing" / "aligned_pre.tif"
                aligned_post = job_dir / "preprocessing" / "aligned_post.tif"
                alignment_info = align_temporal_pair(pre_norm, post_norm, aligned_pre, aligned_post)

                # Stage 5: SUPER_RESOLUTION_PRE
                jobs.update(job_id, stage="SUPER_RESOLUTION_PRE", progress=55.0)
                disaster_dir = job_dir / "application" / "disaster"
                sr_pre_tif = disaster_dir / "pre_event" / "sr_pre.tif"
                sr_post_tif = disaster_dir / "post_event" / "sr_post.tif"
                _, dev1 = PixelSightEngine.enhance(aligned_pre, sr_pre_tif)

                # Stage 6: SUPER_RESOLUTION_POST
                jobs.update(job_id, stage="SUPER_RESOLUTION_POST", progress=70.0)
                _, dev2 = PixelSightEngine.enhance(aligned_post, sr_post_tif)
                device = dev1

                # Stage 7: UNCERTAINTY
                jobs.update(job_id, stage="UNCERTAINTY", progress=80.0)
                pre_unc_tif = disaster_dir / "pre_event" / "unc_pre.tif"
                pre_unc_png = disaster_dir / "pre_event" / "unc_pre.png"
                post_unc_tif = disaster_dir / "post_event" / "unc_post.tif"
                post_unc_png = disaster_dir / "post_event" / "unc_post.png"
                pre_unc = PixelSightEngine.uncertainty(aligned_pre, sr_pre_tif, pre_unc_tif, pre_unc_png)
                post_unc = PixelSightEngine.uncertainty(aligned_post, sr_post_tif, post_unc_tif, post_unc_png)

                # Stage 8: CHANGE_ANALYSIS
                jobs.update(job_id, stage="CHANGE_ANALYSIS", progress=90.0)
                disaster_results = run_disaster_analysis(
                    sr_pre_tif,
                    sr_post_tif,
                    pre_unc_tif,
                    post_unc_tif,
                    job_dir,
                )

                # Stage 9: REPORTING
                jobs.update(job_id, stage="REPORTING", progress=95.0)
                report_path = job_dir / "report" / "report.json"
                output_files = dict(disaster_results["outputs"])
                output_files["report_markdown"] = "report/report.md"
                output_files["report_html"] = "report/report.html"
                output_files["pre_event"] = "input/pre_event.tif"
                output_files["post_event"] = "input/post_event.tif"
                output_files["pre_event_preview"] = "input/pre_event_preview.png"
                output_files["post_event_preview"] = "input/post_event_preview.png"

                report = write_report(
                    report_path,
                    job_id=job_id,
                    input_metadata={
                        **pre_insp.model_dump(),
                        "source": "Copernicus Data Space Ecosystem",
                        "collection": "Sentinel-2 L2A",
                        "pre_scene_id": req.pre_scene_id,
                        "post_scene_id": req.post_scene_id,
                        "aoi": aoi_info["bbox"],
                    },
                    preprocessing=pre_ops + post_ops,
                    runtime_seconds=time.perf_counter() - started,
                    device=device,
                    output_files=output_files,
                    application="disaster",
                    application_data=disaster_results,
                    disaster_metadata=alignment_info,
                    uncertainty={"pre": pre_unc, "post": post_unc},
                )
                output_files["report"] = "report/report.json"
                write_manifest(job_dir, job_id, report)
                output_files["manifest"] = "manifest.json"

                jobs.update(
                    job_id,
                    status="completed",
                    stage="completed",
                    progress=100.0,
                    outputs={
                        "application": "disaster",
                        "source": "Copernicus Data Space Ecosystem",
                        "super_resolution": True,
                        "disaster": disaster_results,
                        "alignment": alignment_info,
                        "pre_event_preview": "input/pre_event_preview.png",
                        "post_event_preview": "input/post_event_preview.png",
                        "manifest": "manifest.json",
                        "report": "report/report.json",
                        "report_html": "report/report.html",
                        "report_markdown": "report/report.md",
                        "runtime_seconds": report["runtime"]["seconds"],
                        "limitations": report["scientific_limitations"],
                    },
                )
            except Exception as exc:
                jobs.update(job_id, status="failed", stage="failed", error=str(exc))

        jobs.submit(job_id, run_disaster)
        return _job_response(jobs.get(job_id))

    # -----------------------------------------------------------------------
    # Async worker for Single-Scene Pipelines (Crop, Urban, Research)
    # -----------------------------------------------------------------------
    def run_single() -> None:
        started = time.perf_counter()
        try:
            # Stage 1: DOWNLOADING_DATA
            jobs.update(job_id, stage="DOWNLOADING_DATA", progress=15.0)
            source_path = job_dir / "input" / "source.tif"
            _, copernicus_meta = copernicus_processing.acquire_aoi(
                aoi=req.aoi,
                scene_id=req.scene_id,
                date=req.date,
                destination=source_path,
            )

            (job_dir / "input" / "copernicus_request.json").write_text(
                json.dumps(copernicus_meta, indent=2), encoding="utf-8"
            )
            (job_dir / "input" / "source_metadata.json").write_text(
                json.dumps(copernicus_meta, indent=2), encoding="utf-8"
            )

            original_preview = job_dir / "input" / "original_preview.png"
            save_rgb_preview(source_path, original_preview)

            # Stage 2: PREPROCESSING
            jobs.update(job_id, stage="PREPROCESSING", progress=25.0)
            inspection = PixelSightEngine.inspect(source_path)
            if not inspection.compatible:
                raise ValueError("Acquired Copernicus raster is incompatible: " + "; ".join(inspection.errors))

            normalized_path = job_dir / "preprocessing" / "normalized.tif"
            _, preprocessing_operations = PixelSightEngine.preprocess(source_path, normalized_path)

            # Stage 3: SUPER_RESOLUTION
            jobs.update(job_id, stage="SUPER_RESOLUTION", progress=40.0)
            output_files, device = _run_super_resolution(job_id, job_dir, normalized_path)
            sr_path = job_dir / output_files["super_resolution"]

            # Stage 4: UNCERTAINTY
            jobs.update(job_id, stage="UNCERTAINTY", progress=70.0)
            uncertainty = PixelSightEngine.uncertainty(
                normalized_path,
                sr_path,
                job_dir / "uncertainty" / "uncertainty_map.tif",
                job_dir / "uncertainty" / "uncertainty_map.png",
            )

            # Stage 5: APPLICATION_ANALYSIS
            jobs.update(job_id, stage="APPLICATION_ANALYSIS", progress=85.0)
            application_data = None

            if app_id == "crop":
                crop_results = run_crop_analysis(normalized_path, sr_path, job_dir)
                application_data = crop_results
                output_files.update(crop_results["outputs"])
            elif app_id == "urban":
                urban_results = run_urban_pipeline(normalized_path, sr_path, job_dir)
                application_data = urban_results
                output_files.update(urban_results["outputs"])

            # Stage 6: REPORTING
            jobs.update(job_id, stage="REPORTING", progress=95.0)
            report_path = job_dir / "report" / "report.json"
            output_files["report_markdown"] = "report/report.md"
            output_files["report_html"] = "report/report.html"
            output_files["original"] = "input/source.tif"
            output_files["uncertainty_map"] = "uncertainty/uncertainty_map.png"
            output_files["uncertainty_geotiff"] = "uncertainty/uncertainty_map.tif"

            evaluation = _evaluate_reference(sr_path, normalized_path)
            report = write_report(
                report_path,
                job_id=job_id,
                input_metadata={
                    **inspection.model_dump(),
                    "source": "Copernicus Data Space Ecosystem",
                    "collection": "Sentinel-2 L2A",
                    "scene_id": req.scene_id,
                    "acquisition_date": req.date,
                    "aoi": aoi_info["bbox"],
                },
                preprocessing=preprocessing_operations,
                runtime_seconds=time.perf_counter() - started,
                device=device,
                output_files=output_files,
                evaluation=evaluation,
                uncertainty=uncertainty,
                application=app_id,
                application_data=application_data,
            )
            output_files["report"] = "report/report.json"
            write_manifest(job_dir, job_id, report)
            output_files["manifest"] = "manifest.json"

            final_outputs = {
                "application": app_id,
                "source": "Copernicus Data Space Ecosystem",
                "super_resolution": True,
                "original_preview": "input/original_preview.png",
                "super_resolution_preview": "super_resolution/sr_preview.png",
                "uncertainty": uncertainty,
                "uncertainty_map": "uncertainty/uncertainty_map.png",
                "uncertainty_geotiff": "uncertainty/uncertainty_map.tif",
                "manifest": "manifest.json",
                "report": "report/report.json",
                "report_html": "report/report.html",
                "report_markdown": "report/report.md",
                "runtime_seconds": report["runtime"]["seconds"],
                "limitations": report["scientific_limitations"],
            }
            if app_id == "crop" and application_data:
                final_outputs["crop"] = application_data
            elif app_id == "urban" and application_data:
                final_outputs["urban"] = application_data

            jobs.update(
                job_id,
                status="completed",
                stage="completed",
                progress=100.0,
                outputs=final_outputs,
            )
        except Exception as exc:
            jobs.update(job_id, status="failed", stage="failed", error=str(exc))

    jobs.submit(job_id, run_single)
    return _job_response(jobs.get(job_id))


# ===========================================================================
# Application Job Status & Query Routes
# ===========================================================================

@app.get("/api/v1/applications/{job_id}", response_model=JobResponse)
def get_application_job(job_id: str) -> JobResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_response(job)


@app.get("/api/v1/applications/{job_id}/results", response_model=ResultResponse)
def get_application_results(job_id: str) -> ResultResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ResultResponse(job_id=job_id, status=job["status"], outputs=job["outputs"], application=job.get("application", "research"))


@app.get("/api/v1/applications/{job_id}/report")
def get_application_report(job_id: str) -> dict:
    return get_report(job_id)


# ===========================================================================
# Core Job and Result Endpoints
# ===========================================================================

def _run_super_resolution(job_id: str, job_dir: Path, input_path: Path) -> tuple[dict[str, str], str]:
    output_path = job_dir / "super_resolution" / "sr.tif"
    sr_preview = job_dir / "super_resolution" / "sr_preview.png"
    original_preview = job_dir / "input" / "original_preview.png"

    save_rgb_preview(input_path, original_preview)
    meta, device = PixelSightEngine.enhance(
        normalized_path=input_path,
        output_sr_path=output_path,
        preview_path=sr_preview,
    )

    jobs.update(
        job_id,
        progress=90.0,
        outputs={
            "super_resolution_file": "super_resolution/sr.tif",
            "original_preview": "input/original_preview.png",
            "super_resolution_preview": "super_resolution/sr_preview.png",
        },
    )
    return {
        "original": f"input/{input_path.name}",
        "normalized_input": "preprocessing/normalized.tif",
        "super_resolution": "super_resolution/sr.tif",
        "original_preview": "input/original_preview.png",
        "super_resolution_preview": "super_resolution/sr_preview.png",
        "urban_planning_map": "analysis/urban_planning_map.png",
    }, device


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_response(job)


@app.get("/api/v1/results/{job_id}", response_model=ResultResponse)
def get_results(job_id: str) -> ResultResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return ResultResponse(job_id=job_id, status=job["status"], outputs=job["outputs"], application=job.get("application", "research"))


@app.get("/api/v1/results/{job_id}/files/{relative_path:path}")
def download_result(job_id: str, relative_path: str) -> FileResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    job_dir = RESULTS_ROOT / job_id
    requested = (job_dir / relative_path).resolve()
    if job_dir.resolve() not in requested.parents or not requested.is_file():
        raise HTTPException(status_code=404, detail="Result file not found.")
    return FileResponse(requested)


@app.get("/api/v1/reports/{job_id}")
def get_report(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    report_path = RESULTS_ROOT / job_id / "report" / "report.json"
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="Report is not available yet.")
    return json.loads(report_path.read_text(encoding="utf-8"))


@app.get("/api/v1/manifest/{job_id}")
def get_manifest(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    manifest_path = RESULTS_ROOT / job_id / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="Manifest is not available yet.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _unavailable_module(module: str) -> dict:
    raise HTTPException(
        status_code=501,
        detail=f"Module '{module}' is not integrated into the application pipeline yet.",
    )


@app.post("/api/v1/batch/process", status_code=501)
def batch_process() -> dict:
    return _unavailable_module("batch/process")


@app.post("/api/v1/super-resolution", status_code=501)
def super_resolution() -> dict:
    return _unavailable_module("super-resolution")


@app.post("/api/v1/uncertainty", status_code=501)
def uncertainty() -> dict:
    return _unavailable_module("uncertainty")


@app.post("/api/v1/segmentation", status_code=501)
def segmentation() -> dict:
    return _unavailable_module("segmentation")


@app.post("/api/v1/analysis/urban", status_code=501)
def urban_analysis() -> dict:
    return _unavailable_module("analysis/urban")


@app.post("/api/v1/analysis/crop", status_code=501)
def crop_analysis() -> dict:
    return _unavailable_module("analysis/crop")


@app.post("/api/v1/analysis/disaster", status_code=501)
def disaster_analysis() -> dict:
    return _unavailable_module("analysis/disaster")
