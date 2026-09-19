from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from backend.app.services.crop import calculate_ndvi, run_crop_analysis
from backend.app.services.reporting import write_manifest, write_report


def _create_synthetic_multispectral_geotiff(path: Path, height: int = 64, width: int = 64) -> None:
    """Create a 4-band float32 GeoTIFF [B02, B03, B04, B08]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Band 1: B02 (Blue), Band 2: B03 (Green), Band 3: B04 (Red), Band 4: B08 (NIR)
    b02 = np.full((height, width), 0.1, dtype=np.float32)
    b03 = np.full((height, width), 0.15, dtype=np.float32)
    b04 = np.linspace(0.05, 0.3, height * width, dtype=np.float32).reshape(height, width)  # Red
    b08 = np.linspace(0.4, 0.8, height * width, dtype=np.float32).reshape(height, width)   # NIR
    bands = np.stack([b02, b03, b04, b08], axis=0)

    transform = from_origin(100.0, 200.0, 10.0, 10.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype="float32",
        crs="EPSG:32632",
        transform=transform,
    ) as dst:
        dst.write(bands)
        for idx, name in enumerate(("B02", "B03", "B04", "B08"), start=1):
            dst.set_band_description(idx, name)


def test_ndvi_calculation_accuracy():
    red = np.array([0.1, 0.2, 0.5], dtype=np.float32)
    nir = np.array([0.5, 0.6, 0.5], dtype=np.float32)
    ndvi = calculate_ndvi(red, nir)
    # (0.5 - 0.1)/(0.5 + 0.1) = 0.4/0.6 = 0.6667
    assert np.isclose(ndvi[0], (0.5 - 0.1) / (0.5 + 0.1), atol=1e-4)
    # (0.6 - 0.2)/(0.6 + 0.2) = 0.4/0.8 = 0.5
    assert np.isclose(ndvi[1], 0.5, atol=1e-4)
    # (0.5 - 0.5)/(0.5 + 0.5) = 0.0
    assert np.isclose(ndvi[2], 0.0, atol=1e-4)


def test_ndvi_bounds_and_nan_safety():
    red = np.array([0.0, -0.1, 1.0], dtype=np.float32)
    nir = np.array([0.0, 0.8, 0.0], dtype=np.float32)
    ndvi = calculate_ndvi(red, nir)
    # All values must be in [-1, 1] or nan
    assert np.nanmin(ndvi) >= -1.0
    assert np.nanmax(ndvi) <= 1.0


def test_crop_pipeline_execution(tmp_path: Path):
    job_dir = tmp_path / "ps_crop_test"
    native_path = job_dir / "input" / "native.tif"
    sr_path = job_dir / "super_resolution" / "sr.tif"

    _create_synthetic_multispectral_geotiff(native_path, height=32, width=32)
    _create_synthetic_multispectral_geotiff(sr_path, height=128, width=128)

    result = run_crop_analysis(native_path, sr_path, job_dir)

    assert result["application"] == "crop"
    assert result["scientific_status"] == "Native-vs-SR consistency analysis"

    # Check metrics
    stats = result["statistics"]
    assert "native" in stats and "super_resolution" in stats
    assert stats["native"]["mean"] > 0
    assert stats["super_resolution"]["mean"] > 0
    assert "mae" in result["consistency_metrics"]
    assert "rmse" in result["consistency_metrics"]

    # Verify generated rasters
    outputs = result["outputs"]
    for key, rel_path in outputs.items():
        full_path = job_dir / rel_path
        assert full_path.exists(), f"Expected crop output {rel_path} was not created"

    # Check that GeoTIFFs have valid CRS and profiles
    with rasterio.open(job_dir / outputs["ndvi_sr_geotiff"]) as src:
        assert src.count == 1
        assert src.crs is not None
        assert src.height == 128
        assert src.width == 128


def test_crop_manifest_and_report_integration(tmp_path: Path):
    job_dir = tmp_path / "ps_crop_manifest"
    native_path = job_dir / "input" / "native.tif"
    sr_path = job_dir / "super_resolution" / "sr.tif"

    _create_synthetic_multispectral_geotiff(native_path, height=32, width=32)
    _create_synthetic_multispectral_geotiff(sr_path, height=128, width=128)

    crop_result = run_crop_analysis(native_path, sr_path, job_dir)
    report_path = job_dir / "report" / "report.json"

    report = write_report(
        report_path,
        job_id="ps_crop_test",
        input_metadata={"width": 32, "height": 32, "bands": 4, "band_names": ["B02", "B03", "B04", "B08"], "crs": "EPSG:32632", "compatible": True},
        preprocessing=[],
        runtime_seconds=1.2,
        device="cpu",
        output_files=crop_result["outputs"],
        application="crop",
        application_data=crop_result,
    )

    assert report["application"] == "crop"
    assert "crop_analysis" in report
    assert any("NDVI" in lim or "pathogen" in lim for lim in report["scientific_limitations"])

    # Manifest check
    manifest = write_manifest(job_dir, "ps_crop_test", report)
    assert manifest["application"] == "crop"
    artifact_paths = [a["path"] for a in manifest["artifacts"]]
    assert "application/crop/ndvi_native.tif" in artifact_paths
    assert "application/crop/ndvi_sr.tif" in artifact_paths
    assert "application/crop/ndvi_difference.tif" in artifact_paths
