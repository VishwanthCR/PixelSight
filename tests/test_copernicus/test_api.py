from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import numpy as np

from backend.app.main import app
from .test_processing import _create_mock_tiff_bytes

client = TestClient(app)


def test_copernicus_capabilities_endpoint():
    response = client.get("/api/v1/copernicus/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "configured" in data
    assert data["bands"] == ["B02", "B03", "B04", "B08"]
    assert data["sr_scale_factor"] == 4
    assert "aoi_limits" in data
    assert data["supported_country"] == "INDIA"


def test_copernicus_geocode_endpoint():
    # Test Indian city
    resp = client.get("/api/v1/copernicus/geocode?q=Chennai")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) > 0
    assert data["results"][0]["inside_india"] is True

    # Test out-of-India city
    resp_ny = client.get("/api/v1/copernicus/geocode?q=New York")
    assert resp_ny.status_code == 200
    data_ny = resp_ny.json()
    assert len(data_ny["results"]) > 0
    assert data_ny["results"][0]["inside_india"] is False


def test_copernicus_india_boundary_endpoint():
    resp = client.get("/api/v1/copernicus/india-boundary")
    assert resp.status_code == 200
    data = resp.json()
    assert "geometry" in data or "coordinates" in data


def test_copernicus_estimate_endpoint():
    response = client.post(
        "/api/v1/copernicus/estimate",
        json={"aoi": [80.264798, 13.076941, 80.276602, 13.088459]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["inside_india"] is True
    assert data["supported_country"] == "INDIA"
    assert data["area_sqkm"] > 0
    assert "native_10m" in data
    assert "super_resolution_2_5m" in data


def test_copernicus_estimate_outside_india_rejected():
    response = client.post(
        "/api/v1/copernicus/estimate",
        json={"aoi": [12.48, 41.89, 12.50, 41.91]},
    )
    assert response.status_code == 422
    assert "outside India" in response.json()["detail"]


def test_copernicus_search_endpoint():
    mock_scenes = [
        {
            "id": "S2A_TEST_SCENE",
            "datetime": "2023-06-29T10:00:31Z",
            "cloud_cover": 3.2,
            "collection": "Sentinel-2 L2A",
            "bbox": [80.264798, 13.076941, 80.276602, 13.088459],
        }
    ]

    with patch("backend.app.main.copernicus_catalog.search_scenes", return_value=mock_scenes):
        response = client.post(
            "/api/v1/copernicus/search",
            json={
                "aoi": [80.264798, 13.076941, 80.276602, 13.088459],
                "start_date": "2023-06-01",
                "end_date": "2023-06-30",
                "max_cloud_cover": 20.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_found"] == 1
        assert data["scenes"][0]["id"] == "S2A_TEST_SCENE"
        assert data["best_scene"]["id"] == "S2A_TEST_SCENE"


def test_copernicus_application_job_creation(tmp_path):
    mock_tiff = _create_mock_tiff_bytes(128, 128)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = mock_tiff

    with patch("backend.app.services.copernicus_processing.copernicus_auth.get_access_token", return_value="mock_token"):
        with patch("requests.post", return_value=mock_resp):
            response = client.post(
                "/api/v1/applications/crop/from-copernicus",
                json={
                    "aoi": [80.264798, 13.076941, 80.276602, 13.088459],
                    "scene_id": "S2A_TEST_SCENE",
                    "date": "2023-06-29T10:00:31Z",
                },
            )
            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            assert data["application"] == "crop"
