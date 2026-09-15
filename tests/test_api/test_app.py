from pathlib import Path

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from backend.app.main import app


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


def test_non_raster_extension_is_rejected() -> None:
    response = client.post(
        "/api/v1/inspect",
        files={"upload": ("scene.png", b"not-a-raster", "image/png")},
    )

    assert response.status_code == 415


def test_unintegrated_module_is_explicit() -> None:
    response = client.post("/api/v1/uncertainty")

    assert response.status_code == 501