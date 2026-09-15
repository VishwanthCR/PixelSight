from pathlib import Path
import tempfile
import time
import json
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

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
from backend.app.services.visualization import save_rgb_preview


app = FastAPI(title="PixelSight API", version="0.1.0")
jobs = JobStore(RESULTS_ROOT)


def _safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Only .tif and .tiff files are supported.")
    return suffix


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    _safe_suffix(upload.filename)
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
        jobs.update(job_id, stage="reporting", progress=95.0)
        report_path = job_dir / "report" / "report.json"
        output_files["original"] = f"input/{input_path.name}"
        report = write_report(
            report_path,
            job_id=job_id,
            input_metadata=inspection.model_dump(),
            preprocessing=preprocessing_operations,
            runtime_seconds=time.perf_counter() - started,
            device=device,
            output_files=output_files,
        )
        output_files["report"] = "report/report.json"
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
                "uncertainty": False,
                "segmentation": False,
                "evaluation": False,
                "runtime_seconds": report["runtime"]["seconds"],
                "limitations": [
                    "Uncertainty and segmentation services are not yet wired into the application job pipeline.",
                    "Reference-dependent evaluation is unavailable without a valid high-resolution reference.",
                ],
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
