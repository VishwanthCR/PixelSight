from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from backend.app.main import app


client = TestClient(app)


def _create_test_geotiff(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((4, 128, 128), 0.2, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=128,
        width=128,
        count=4,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 128.0, 1.0, 1.0),
    ) as dst:
        dst.write(data)
        for idx, name in enumerate(("B02", "B03", "B04", "B08"), start=1):
            dst.set_band_description(idx, name)
    return path


def test_get_applications():
    response = client.get("/api/v1/applications")
    assert response.status_code == 200
    apps = response.json()
    assert isinstance(apps, list)
    ids = [a["id"] for a in apps]
    assert "research" in ids
    assert "crop" in ids
    assert "urban" in ids
    assert "disaster" in ids


def test_get_capabilities():
    response = client.get("/api/v1/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "LDSR-S2"
    assert data["scale"] == 4
    assert len(data["applications"]) >= 4


def test_crop_application_endpoint(tmp_path: Path):
    tif_path = _create_test_geotiff(tmp_path / "crop_input.tif")
    with open(tif_path, "rb") as f:
        response = client.post("/api/v1/applications/crop", files={"upload": ("crop_input.tif", f, "image/tiff")})
    assert response.status_code == 202
    data = response.json()
    assert data["application"] == "crop"
    assert "job_id" in data


def test_urban_application_endpoint(tmp_path: Path):
    tif_path = _create_test_geotiff(tmp_path / "urban_input.tif")
    with open(tif_path, "rb") as f:
        response = client.post("/api/v1/applications/urban", files={"upload": ("urban_input.tif", f, "image/tiff")})
    assert response.status_code == 202
    data = response.json()
    assert data["application"] == "urban"
    assert "job_id" in data


def test_disaster_application_endpoint(tmp_path: Path):
    pre_path = _create_test_geotiff(tmp_path / "pre_event.tif")
    post_path = _create_test_geotiff(tmp_path / "post_event.tif")
    with open(pre_path, "rb") as f1, open(post_path, "rb") as f2:
        response = client.post(
            "/api/v1/applications/disaster",
            files={
                "pre_event": ("pre_event.tif", f1, "image/tiff"),
                "post_event": ("post_event.tif", f2, "image/tiff"),
            },
        )
    assert response.status_code == 202
    data = response.json()
    assert data["application"] == "disaster"
    assert "job_id" in data
