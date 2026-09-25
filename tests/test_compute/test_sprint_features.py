"""
Regression and integration tests for sprint objectives:
- Compute execution planner endpoint (GET/POST /api/v1/compute/plan)
- Job cancellation and retry lifecycle (POST /api/v1/jobs/{job_id}/cancel and retry)
- Land cover classification/segmentation endpoint and status
- Master evaluation report endpoint (GET /api/v1/evaluation/report)
"""
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from fastapi.testclient import TestClient

from backend.app.main import app, jobs


@pytest.fixture
def client():
    return TestClient(app)


def make_valid_geotiff(path: Path, width: int = 128, height: int = 128) -> None:
    """Create a synthetically valid 4-band Sentinel-2 GeoTIFF (B02, B03, B04, B08)."""
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 4,
        "dtype": "uint16",
        "crs": "EPSG:32643",
        "transform": from_origin(500000, 2000000, 10, 10),
    }
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(np.full((4, height, width), 2500, dtype=np.uint16))
        for index, description in enumerate(("B02", "B03", "B04", "B08"), start=1):
            dataset.set_band_description(index, description)


# ---------------------------------------------------------------------------
# 1. Compute Plan Tests
# ---------------------------------------------------------------------------

def test_compute_plan_get_endpoint(client):
    response = client.get("/api/v1/compute/plan?width=128&height=128&batch_size=1&n_images=1")
    assert response.status_code == 200
    data = response.json()
    assert "backend" in data
    assert "workers" in data
    assert "gpu_devices" in data
    assert "tile_batch_size" in data
    assert "parallelism" in data
    assert "reason" in data
    assert "estimated_tiles_per_scene" in data
    assert data["estimated_tiles_per_scene"] == 1
    assert data["diffusion_steps"] == 100
    assert data["scale_factor"] == 4


def test_compute_plan_scales_with_image_size(client):
    small_resp = client.get("/api/v1/compute/plan?width=128&height=128")
    large_resp = client.get("/api/v1/compute/plan?width=512&height=512")
    assert small_resp.status_code == 200
    assert large_resp.status_code == 200
    small_tiles = small_resp.json()["estimated_tiles_per_scene"]
    large_tiles = large_resp.json()["estimated_tiles_per_scene"]
    assert large_tiles > small_tiles
    assert large_tiles == 25  # 5x5 grid of 128x128 tiles with 12px overlap covering 512x512


# ---------------------------------------------------------------------------
# 2. Job Cancel & Retry Tests
# ---------------------------------------------------------------------------

def test_job_cancel_endpoint(client):
    job_id, _ = jobs.create(application="research")
    response = client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "cancelled"
    assert data["job_id"] == job_id

    # Verify status in job store
    job = jobs.get(job_id)
    assert job["status"] == "cancelled"


def test_job_cancel_nonexistent_returns_404(client):
    response = client.post("/api/v1/jobs/ps_nonexistent_xyz/cancel")
    assert response.status_code == 404


def test_job_retry_nonexistent_returns_404(client):
    response = client.post("/api/v1/jobs/ps_nonexistent_xyz/retry")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# 3. Segmentation / Classification Subsystem Tests
# ---------------------------------------------------------------------------

def test_classification_status_endpoint(client):
    response = client.get("/api/v1/classification/status")
    assert response.status_code == 200
    data = response.json()
    assert "available" in data
    assert "model" in data
    assert "device" in data
    assert "classes" in data
    assert len(data["classes"]) == 8  # 7 ESA classes + Ignore (255)
    class_ids = [c["id"] for c in data["classes"]]
    assert set(class_ids) == {0, 1, 2, 3, 4, 5, 6, 255}


def test_segmentation_status_alias(client):
    """GET /api/v1/segmentation/status behaves identically to classification/status."""
    response = client.get("/api/v1/segmentation/status")
    assert response.status_code == 200
    assert "available" in response.json()


def test_classification_rejects_non_tiff(client, tmp_path):
    png_path = tmp_path / "test.png"
    png_path.write_bytes(b"NOT_A_TIFF")
    with open(png_path, "rb") as f:
        response = client.post(
            "/api/v1/classification",
            files={"upload": ("test.png", f, "image/png")},
        )
    assert response.status_code == 415


def test_classification_missing_checkpoint_truthful_error(client, tmp_path):
    tif_path = tmp_path / "valid.tif"
    make_valid_geotiff(tif_path)

    # Mock checkpoint non-existence to verify truthful error
    with patch("pathlib.Path.exists", return_value=False):
        with open(tif_path, "rb") as f:
            response = client.post(
                "/api/v1/classification",
                files={"upload": ("valid.tif", f, "image/tiff")},
            )
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["status"] == "unavailable"
        assert "Classification unavailable" in detail["title"]
        assert "expected_resource" in detail
        assert "how_to_fix" in detail


def test_classification_rejects_incompatible_bands(client, tmp_path):
    """GeoTIFF with wrong band descriptions is rejected with 422."""
    tif_path = tmp_path / "rgb.tif"
    profile = {
        "driver": "GTiff",
        "height": 64,
        "width": 64,
        "count": 3,
        "dtype": "uint16",
        "crs": "EPSG:32643",
        "transform": from_origin(500000, 2000000, 10, 10),
    }
    with rasterio.open(tif_path, "w", **profile) as ds:
        ds.write(np.full((3, 64, 64), 2000, dtype=np.uint16))

    with open(tif_path, "rb") as f:
        response = client.post(
            "/api/v1/classification",
            files={"upload": ("rgb.tif", f, "image/tiff")},
        )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 4. Evaluation Report Endpoint Tests
# ---------------------------------------------------------------------------

def test_evaluation_report_endpoint(client):
    response = client.get("/api/v1/evaluation/report")
    assert response.status_code == 200
    data = response.json()
    assert "metadata" in data
    assert "image_metrics" in data
    assert "spectral_metrics" in data
    assert "uncertainty_metrics" in data
    assert "downstream_metrics" in data
    assert "methodological_limitations" in data
    assert isinstance(data["methodological_limitations"], list)
