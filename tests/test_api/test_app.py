from io import BytesIO
from pathlib import Path

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.transform import from_origin

from backend.app.main import app
from backend.app.services.urban import compare_urban_analysis


client = TestClient(app)


def make_raster(path: Path, descriptions: tuple[str, ...]) -> None:
    profile = {
        "driver": "GTiff",
        "height": 128,
        "width": 128,
        "count": 4,
        "dtype": "uint16",
        "crs": "EPSG:32643",
        "transform": from_origin(500000, 2000000, 10, 10),
    }
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(np.full((4, 128, 128), 5000, dtype=np.uint16))
        for index, description in enumerate(descriptions, start=1):
            dataset.set_band_description(index, description)


def test_health() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_inspect_rejects_unidentified_bands(tmp_path: Path) -> None:
    path = tmp_path / "rgb.tif"
    make_raster(path, ("red", "green", "blue", "nir"))

    with path.open("rb") as source:
        response = client.post(
            "/api/v1/inspect",
            files={"upload": (path.name, source, "image/tiff")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert body["compatible"] is False


def test_preprocess_normalizes_approved_bands(tmp_path: Path) -> None:
    path = tmp_path / "scene.tif"
    make_raster(path, ("B02", "B03", "B04", "B08"))

    with path.open("rb") as source:
        response = client.post(
            "/api/v1/preprocess",
            files={"upload": (path.name, source, "image/tiff")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["inspection"]["compatible"] is True
    assert {operation["operation"] for operation in body["operations"]} >= {
        "normalize_reflectance",
        "convert_dtype",
    }


def test_png_image_is_converted_to_compatible_raster() -> None:
    buffer = BytesIO()
    Image.new("RGB", (128, 128), color=(10, 20, 30)).save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/api/v1/inspect",
        files={"upload": ("scene.png", buffer, "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["compatible"] is True
    assert body["bands"] == 4
    assert body["band_names"] == ["B02", "B03", "B04", "B08"]
    assert body["dtype"] == "float32"
    assert body["requires_preprocessing"] is False


def test_non_raster_extension_is_rejected() -> None:
    response = client.post(
        "/api/v1/inspect",
        files={"upload": ("scene.exe", b"not-a-raster", "application/octet-stream")},
    )

    assert response.status_code == 415


def test_process_marks_job_processing_immediately(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input.tif"
    make_raster(input_path, ("B02", "B03", "B04", "B08"))

    def fake_run_super_resolution(job_id: str, job_dir: Path, input_path: Path):
        output_path = job_dir / "super_resolution" / "sr.tif"
        with rasterio.open(input_path) as src:
            data = src.read().astype(np.float32) / 10000.0
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=data.shape[1],
            width=data.shape[2],
            count=4,
            dtype="float32",
            crs="EPSG:32643",
            transform=from_origin(500000, 2000000, 10, 10),
        ) as dst:
            dst.write(data)
        original_preview = job_dir / "input" / "original_preview.png"
        sr_preview = job_dir / "super_resolution" / "sr_preview.png"
        original_preview.write_bytes(b"png")
        sr_preview.write_bytes(b"png")
        return {
            "original": "input/original.tif",
            "normalized_input": "preprocessing/normalized.tif",
            "super_resolution": "super_resolution/sr.tif",
            "original_preview": "input/original_preview.png",
            "super_resolution_preview": "super_resolution/sr_preview.png",
        }, "cpu"

    monkeypatch.setattr("backend.app.main._run_super_resolution", fake_run_super_resolution)

    with input_path.open("rb") as input_file:
        response = client.post(
            "/api/v1/process",
            files={"upload": (input_path.name, input_file, "image/tiff")},
        )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "processing"
    job_id = body["job_id"]
    final = client.get(f"/api/v1/jobs/{job_id}")
    assert final.json()["status"] in {"processing", "completed"}


def test_process_uses_input_as_evaluation_reference(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input.tif"
    make_raster(input_path, ("B02", "B03", "B04", "B08"))

    def fake_run_super_resolution(job_id: str, job_dir: Path, input_path: Path):
        output_path = job_dir / "super_resolution" / "sr.tif"
        with rasterio.open(input_path) as src:
            data = src.read().astype(np.float32) / 10000.0
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=data.shape[1],
            width=data.shape[2],
            count=4,
            dtype="float32",
            crs="EPSG:32643",
            transform=from_origin(500000, 2000000, 10, 10),
        ) as dst:
            dst.write(data)
        original_preview = job_dir / "input" / "original_preview.png"
        sr_preview = job_dir / "super_resolution" / "sr_preview.png"
        original_preview.write_bytes(b"png")
        sr_preview.write_bytes(b"png")
        return {
            "original": "input/original.tif",
            "normalized_input": "preprocessing/normalized.tif",
            "super_resolution": "super_resolution/sr.tif",
            "original_preview": "input/original_preview.png",
            "super_resolution_preview": "super_resolution/sr_preview.png",
        }, "cpu"

    monkeypatch.setattr("backend.app.main._run_super_resolution", fake_run_super_resolution)
    monkeypatch.setattr(
        "backend.app.main.run_urban_analysis",
        lambda *args, **kwargs: {
            "status": "estimated",
            "map": "analysis/urban_planning_map.png",
            "tree_clusters": 2,
            "estimated_building_clusters": 3,
            "limitations": [],
        },
    )

    with input_path.open("rb") as input_file:
        response = client.post(
            "/api/v1/process",
            files={"upload": (input_path.name, input_file, "image/tiff")},
        )

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    for _ in range(50):
        job_response = client.get(f"/api/v1/jobs/{job_id}")
        if job_response.json()["status"] == "completed":
            break
    final = client.get(f"/api/v1/jobs/{job_id}")
    assert final.json()["status"] == "completed"
    assert final.json()["outputs"]["evaluation"] is True


def test_unintegrated_module_is_explicit() -> None:
    response = client.post("/api/v1/uncertainty")

    assert response.status_code == 501


def test_urban_comparison_reports_input_and_sr_deltas() -> None:
    input_analysis = {
        "object_counts": {
            "tree": {"label": "Trees", "object_count": 2, "area_percent": 10.0},
        }
    }
    sr_analysis = {
        "object_counts": {
            "tree": {"label": "Trees", "object_count": 5, "area_percent": 12.5},
        }
    }

    comparison = compare_urban_analysis(input_analysis, sr_analysis)

    assert comparison["classes"]["tree"]["input_object_count"] == 2
    assert comparison["classes"]["tree"]["sr_object_count"] == 5
    assert comparison["classes"]["tree"]["object_count_change"] == 3
    assert comparison["classes"]["tree"]["area_change_percent_points"] == 2.5