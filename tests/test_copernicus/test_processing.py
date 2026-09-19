import io
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import rasterio
from rasterio.transform import from_origin

from backend.app.services.copernicus_processing import CopernicusProcessingService, CopernicusProcessingError
from backend.app.services.raster import inspect_raster


def _create_mock_tiff_bytes(width=64, height=64) -> bytes:
    mem = io.BytesIO()
    profile = {
        "driver": "GTiff",
        "count": 4,
        "dtype": "float32",
        "width": width,
        "height": height,
        "crs": "EPSG:4326",
        "transform": from_origin(12.48, 41.91, 0.0001, 0.0001),
    }
    with rasterio.open(mem, "w", **profile) as dst:
        data = np.random.uniform(0.05, 0.6, (4, height, width)).astype(np.float32)
        dst.write(data)
    return mem.getvalue()


def test_copernicus_processing_acquire(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    service = CopernicusProcessingService(cache_dir=cache_dir)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = _create_mock_tiff_bytes()

    out_file = tmp_path / "acquired.tif"

    with patch("backend.app.services.copernicus_processing.copernicus_auth.get_access_token", return_value="mock_token"):
        with patch("requests.post", return_value=mock_resp):
            target_path, meta = service.acquire_aoi(
                aoi=[80.264798, 13.076941, 80.276602, 13.088459],
                scene_id="TEST_SCENE_123",
                date="2023-06-29T10:00:31Z",
                destination=out_file,
            )

    assert target_path.exists()
    assert meta["scene_id"] == "TEST_SCENE_123"

    # Verify GeoTIFF inspection
    inspection = inspect_raster(target_path)
    assert inspection.valid is True
    assert inspection.compatible is True
    assert inspection.bands == 4
    assert inspection.band_names == ["B02", "B03", "B04", "B08"]
    assert inspection.crs is not None
