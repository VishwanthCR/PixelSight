import os
from pathlib import Path
import pytest

from backend.app.services.aoi import validate_and_estimate_aoi
from backend.app.services.copernicus_auth import copernicus_auth
from backend.app.services.copernicus_catalog import copernicus_catalog
from backend.app.services.copernicus_processing import copernicus_processing
from backend.app.services.core_engine import PixelSightEngine
from backend.app.services.crop import run_crop_analysis
from backend.app.services.raster import inspect_raster
from backend.app.services.reporting import write_manifest, write_report


@pytest.mark.integration
def test_real_copernicus_pipeline_e2e(tmp_path: Path):
    """
    End-to-end integration test against live Copernicus Data Space Ecosystem / Sentinel Hub APIs.
    Validates:
    1. OAuth authentication
    2. Catalog STAC scene search
    3. AOI acquisition of 4-band Sentinel-2 L2A (B02, B03, B04, B08) GeoTIFF
    4. GeoTIFF geospatial metadata integrity (CRS, transform, resolution, bands)
    5. Core preprocessing & super-resolution
    6. Uncertainty generation
    7. Downstream crop NDVI analysis
    8. Report and manifest generation
    """
    if not copernicus_auth.is_configured():
        pytest.skip("Copernicus credentials not found in environment; skipping live integration test.")

    # 1. Authenticate
    token = copernicus_auth.get_access_token()
    assert token is not None and len(token) > 20

    # 2. Natural default AOI (~1.28 km x 1.28 km over Chennai, India)
    aoi_bbox = [80.264798, 13.076941, 80.276602, 13.088459]
    aoi_info = validate_and_estimate_aoi(aoi_bbox)
    assert aoi_info["valid"] is True
    assert aoi_info["inside_india"] is True

    # 3. Scene search
    scenes = copernicus_catalog.search_scenes(
        aoi=aoi_bbox,
        start_date="2023-06-01",
        end_date="2023-06-30",
        max_cloud_cover=30.0,
        limit=5,
    )
    assert len(scenes) > 0, "Expected at least one Sentinel-2 scene in June 2023 for test AOI"

    best_scene = copernicus_catalog.select_best_scene(scenes)
    assert best_scene is not None
    scene_id = best_scene["id"]
    scene_date = best_scene["datetime"]

    # 4. Acquire B02, B03, B04, B08 GeoTIFF
    dest_tif = tmp_path / "copernicus_source.tif"
    acquired_path, meta = copernicus_processing.acquire_aoi(
        aoi=aoi_bbox,
        scene_id=scene_id,
        date=scene_date,
        destination=dest_tif,
        use_cache=False,
    )
    assert acquired_path.exists()
    assert meta["source"] == "Copernicus Data Space Ecosystem"
    assert meta["collection"] == "Sentinel-2 L2A"

    # 5. Inspect and validate GeoTIFF
    inspection = inspect_raster(acquired_path)
    assert inspection.valid is True
    assert inspection.compatible is True
    assert inspection.bands == 4
    assert tuple(inspection.band_names) == ("B02", "B03", "B04", "B08")
    assert inspection.crs is not None

    # 6. Preprocess
    norm_path = tmp_path / "normalized.tif"
    insp_norm, ops = PixelSightEngine.preprocess(acquired_path, norm_path)
    assert norm_path.exists()
    assert insp_norm.compatible is True

    # 7. Super-resolution (with PIXELSIGHT_TEST_MODE=1 or LDSR model)
    sr_path = tmp_path / "sr_output.tif"
    preview_path = tmp_path / "sr_preview.png"
    engine_meta, device = PixelSightEngine.enhance(norm_path, sr_path, preview_path)
    assert sr_path.exists()

    # 8. Uncertainty
    unc_tif = tmp_path / "unc.tif"
    unc_png = tmp_path / "unc.png"
    unc_stats = PixelSightEngine.uncertainty(norm_path, sr_path, unc_tif, unc_png)
    assert unc_tif.exists()
    assert "mean" in unc_stats

    # 9. Downstream Crop Analysis
    crop_res = run_crop_analysis(norm_path, sr_path, tmp_path)
    assert "statistics" in crop_res
    assert "native" in crop_res["statistics"]
    assert "super_resolution" in crop_res["statistics"]
    assert "mean" in crop_res["statistics"]["super_resolution"]

    # 10. Write report and manifest
    report_path = tmp_path / "report.json"
    report = write_report(
        report_path,
        job_id="integration_test_copernicus",
        input_metadata=meta,
        preprocessing=ops,
        runtime_seconds=5.0,
        device=device,
        output_files=crop_res["outputs"],
        uncertainty=unc_stats,
        application="crop",
        application_data=crop_res,
    )
    assert report_path.exists()

    manifest = write_manifest(tmp_path, "integration_test_copernicus", report)
    assert (tmp_path / "manifest.json").exists()
    assert "schema_version" in manifest
    assert manifest["job_id"] == "integration_test_copernicus"
