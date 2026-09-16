import io
import json
from pathlib import Path
import tempfile
import time
from typing import Annotated

import numpy as np
import rasterio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from rasterio.transform import from_origin
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from backend.app.config import ALLOWED_SUFFIXES, MAX_UPLOAD_BYTES, RESULTS_ROOT
from backend.app.schemas import (
    HealthResponse,
    InspectionResponse,
    JobResponse,
    PreprocessResponse,
    ResultResponse,
)
from backend.app.services.jobs import JobStore
from backend.app.services.raster import inspect_raster, preprocess_raster
from backend.app.services.reporting import write_report
from backend.app.services.urban import compare_urban_analysis, run_urban_analysis
from backend.app.services.uncertainty import generate_uncertainty_map
from backend.app.services.visualization import save_rgb_preview


app = FastAPI(title="PixelSight API", version="0.1.0")
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


async def _convert_image_to_raster(upload: UploadFile, destination: Path) -> None:
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded image is empty.")
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:  # pragma: no cover - external decoding failure
        raise HTTPException(status_code=422, detail=f"Unable to decode uploaded image: {exc}") from exc

    minimum_size = 128
    if image.width < minimum_size or image.height < minimum_size:
        image = image.resize((max(image.width, minimum_size), max(image.height, minimum_size)), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    red = array[:, :, 0].astype(np.float32)
    green = array[:, :, 1].astype(np.float32)
    blue = array[:, :, 2].astype(np.float32)
    nir = np.clip((red + green + blue) / 3.0, 0.0, 1.0)
    bands = np.stack([blue, green, red, nir], axis=0)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        destination,
        "w",
        driver="GTiff",
        height=bands.shape[1],
        width=bands.shape[2],
        count=4,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, bands.shape[1], 1.0, 1.0),
    ) as dataset:
        dataset.write(bands)
        for index, name in enumerate(("B02", "B03", "B04", "B08"), start=1):
            dataset.set_band_description(index, name)


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    _safe_suffix(upload.filename)
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
        await _convert_image_to_raster(upload, destination)
        return

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
    return JobResponse(**{key: job[key] for key in ("job_id", "status", "stage", "progress", "error", "outputs")})


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


@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="pixelsight")


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


@app.post("/api/v1/process", response_model=JobResponse, status_code=202)
async def process(upload: Annotated[UploadFile, File(...)]) -> JobResponse:
    suffix = _safe_suffix(upload.filename)
    job_id, job_dir = jobs.create()
    input_path = job_dir / "input" / f"original{suffix}"
    await _save_upload(upload, input_path)

    def run() -> None:
        started = time.perf_counter()
        jobs.update(job_id, stage="inspecting", progress=5.0)
        inspection = inspect_raster(input_path)
        if not inspection.compatible:
            raise ValueError("Input is incompatible: " + "; ".join(inspection.errors))
        jobs.update(job_id, stage="preprocessing", progress=15.0)
        normalized_path = job_dir / "preprocessing" / "normalized.tif"
        _, preprocessing_operations = preprocess_raster(input_path, normalized_path)
        jobs.update(job_id, stage="super_resolution", progress=20.0)
        output_files, device = _run_super_resolution(job_id, job_dir, normalized_path)
        jobs.update(job_id, stage="urban_analysis", progress=92.0)
        sr_path = job_dir / output_files["super_resolution"]
        uncertainty = generate_uncertainty_map(
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
        jobs.update(job_id, stage="reporting", progress=95.0)
        report_path = job_dir / "report" / "report.json"
        output_files["original"] = f"input/{input_path.name}"
        output_files["urban_planning_map"] = "analysis/urban_planning_map.png"
        output_files["urban_classification"] = "analysis/urban_classes.tif"
        output_files["uncertainty_map"] = "uncertainty/uncertainty_map.png"
        output_files["uncertainty_geotiff"] = "uncertainty/uncertainty_map.tif"

        evaluation = {"status": "reference_unavailable", "psnr": None, "ssim": None, "sam": None, "reason": "The original input was not usable for evaluation after preprocessing."}
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
        )
        output_files["report"] = "report/report.json"
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
            },
        )

    jobs.submit(job_id, run)
    return _job_response(jobs.get(job_id))


def _run_super_resolution(job_id: str, job_dir: Path, input_path: Path) -> tuple[dict[str, str], str]:
    import importlib
    import numpy as np
    import rasterio
    import torch

    inference = importlib.import_module("scripts.plan2.infer_ldsr_s2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        bounds = src.bounds
        image = np.moveaxis(src.read(), 0, -1)
    model = inference.load_model(device)
    output = inference.run_inference(model, image, device)
    output_path = job_dir / "super_resolution" / "sr.tif"
    inference.save_output(output, output_path, profile)
    inference.verify_output(output_path, bounds)
    original_preview = job_dir / "input" / "original_preview.png"
    sr_preview = job_dir / "super_resolution" / "sr_preview.png"
    save_rgb_preview(input_path, original_preview)
    save_rgb_preview(output_path, sr_preview)
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
    return ResultResponse(job_id=job_id, status=job["status"], outputs=job["outputs"])


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
