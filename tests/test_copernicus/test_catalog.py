from unittest.mock import patch, MagicMock
import pytest

from backend.app.services.copernicus_catalog import CopernicusCatalogService, CopernicusCatalogError


def test_copernicus_catalog_search():
    service = CopernicusCatalogService()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "features": [
            {
                "id": "S2A_MSIL2A_20230629T100031",
                "bbox": [12.4, 41.8, 12.6, 42.0],
                "properties": {
                    "datetime": "2023-06-29T10:00:31Z",
                    "eo:cloud_cover": 4.5,
                    "platform": "Sentinel-2A",
                },
            },
            {
                "id": "S2B_MSIL2A_20230625T100031",
                "bbox": [12.4, 41.8, 12.6, 42.0],
                "properties": {
                    "datetime": "2023-06-25T10:00:31Z",
                    "eo:cloud_cover": 18.2,
                    "platform": "Sentinel-2B",
                },
            },
            {
                "id": "S2A_MSIL2A_20230620T100031",
                "bbox": [12.4, 41.8, 12.6, 42.0],
                "properties": {
                    "datetime": "2023-06-20T10:00:31Z",
                    "eo:cloud_cover": 75.0,  # exceeds max_cloud_cover
                    "platform": "Sentinel-2A",
                },
            },
        ]
    }

    with patch("backend.app.services.copernicus_catalog.copernicus_auth.get_access_token", return_value="mock_token"):
        with patch("requests.post", return_value=mock_resp):
            scenes = service.search_scenes(
                aoi=[12.4, 41.8, 12.6, 42.0],
                start_date="2023-06-01",
                end_date="2023-06-30",
                max_cloud_cover=20.0,
            )

    assert len(scenes) == 2
    assert scenes[0]["id"] == "S2A_MSIL2A_20230629T100031"
    assert scenes[0]["cloud_cover"] == 4.5

    best = service.select_best_scene(scenes)
    assert best is not None
    assert best["id"] == "S2A_MSIL2A_20230629T100031"
    assert "Lowest cloud coverage" in best["selection_criterion"]
