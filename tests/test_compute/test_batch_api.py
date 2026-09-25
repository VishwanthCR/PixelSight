"""
Tests for the PixelSight batch processing API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Batch endpoint — validation
# ---------------------------------------------------------------------------

def test_batch_rejects_empty_file_list(client):
    """POST /api/v1/batch/process with no files should return 422."""
    response = client.post("/api/v1/batch/process", data={"application": "research"})
    # FastAPI returns 422 for missing required fields
    assert response.status_code in {422, 400}


def test_batch_rejects_invalid_application(client, tmp_path):
    """Unsupported application names should return 400 (or 422 if validation fires first)."""
    tif_path = tmp_path / "test.tif"
    tif_path.write_bytes(b"DUMMY_GEOTIFF_CONTENT")
    with open(tif_path, "rb") as f:
        response = client.post(
            "/api/v1/batch/process",
            files=[("uploads", ("test.tif", f, "image/tiff"))],
            data={"application": "disaster"},  # disaster not supported in batch
        )
    # FastAPI may return 422 for form validation or 400 for business logic
    assert response.status_code in {400, 422}


def test_batch_rejects_non_tiff(client, tmp_path):
    """Non-TIF files should be rejected by the safe-suffix check."""
    png_path = tmp_path / "image.png"
    png_path.write_bytes(b"FAKE_PNG")
    with open(png_path, "rb") as f:
        response = client.post(
            "/api/v1/batch/process",
            files=[("uploads", ("image.png", f, "image/png"))],
            data={"application": "research"},
        )
    # Expect 415 (unsupported media) or 422 from the safe-suffix guard
    assert response.status_code in {415, 422}


# ---------------------------------------------------------------------------
# Batch jobs listing
# ---------------------------------------------------------------------------

def test_list_batch_jobs_empty(client):
    """GET /api/v1/batch/jobs returns a list (may be empty on fresh server)."""
    response = client.get("/api/v1/batch/jobs")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "jobs" in data
    assert isinstance(data["jobs"], list)
    assert data["total"] == len(data["jobs"])


def test_list_batch_jobs_with_unknown_batch_filter(client):
    """Filtering by a nonexistent batch_id returns empty list."""
    response = client.get("/api/v1/batch/jobs?batch_id=nonexistent_batch_xyz")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["jobs"] == []


# ---------------------------------------------------------------------------
# Compute profile endpoint
# ---------------------------------------------------------------------------

def test_compute_profile_endpoint(client):
    """GET /api/v1/compute/profile returns hardware and execution_plan."""
    response = client.get("/api/v1/compute/profile")
    assert response.status_code == 200
    data = response.json()
    assert "hardware" in data
    assert "execution_plan" in data


def test_compute_profile_hardware_fields(client):
    response = client.get("/api/v1/compute/profile")
    hw = response.json()["hardware"]
    required = ["device_type", "environment", "cpu_logical_cores", "torch_version"]
    for field in required:
        assert field in hw, f"Missing hardware field: {field}"


def test_compute_profile_execution_plan_fields(client):
    response = client.get("/api/v1/compute/profile")
    ep = response.json()["execution_plan"]
    required = ["strategy", "ldsr_device", "tile_batch_size", "tile_workers", "fp16_inference"]
    for field in required:
        assert field in ep, f"Missing execution_plan field: {field}"


def test_compute_profile_device_type_valid(client):
    response = client.get("/api/v1/compute/profile")
    dt = response.json()["hardware"]["device_type"]
    assert dt in {"cpu", "cuda", "mps"}
