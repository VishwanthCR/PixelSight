from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from backend.app.services.reporting import write_manifest, write_report
from backend.app.services.urban import CLASS_NAMES, NUM_CLASSES, run_urban_pipeline


def _create_synthetic_multispectral_geotiff(path: Path, height: int = 128, width: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    b02 = np.full((height, width), 0.1, dtype=np.float32)
    b03 = np.full((height, width), 0.15, dtype=np.float32)
    b04 = np.full((height, width), 0.2, dtype=np.float32)
    b08 = np.full((height, width), 0.4, dtype=np.float32)
    bands = np.stack([b02, b03, b04, b08], axis=0)

    transform = from_origin(500.0, 500.0, 10.0, 10.0)
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


def test_urban_classes_structure():
    assert len(CLASS_NAMES) == 7
    assert CLASS_NAMES[4] == "built_up"
    assert CLASS_NAMES[0] == "tree"


def test_urban_pipeline_execution(tmp_path: Path):
    job_dir = tmp_path / "ps_urban_test"
    native_path = job_dir / "input" / "native.tif"
    sr_path = job_dir / "super_resolution" / "sr.tif"

    _create_synthetic_multispectral_geotiff(native_path, height=128, width=128)
    _create_synthetic_multispectral_geotiff(sr_path, height=128, width=128)

    result = run_urban_pipeline(native_path, sr_path, job_dir, device="cpu")

    assert result["application"] == "urban"
    assert result["scientific_status"] == "Model-output comparison"

    # Verify indicators
    indicators = result["indicators"]
    assert "built_up_area_pixels" in indicators
    assert "built_up_fraction" in indicators
    assert "vegetation_fraction" in indicators
    assert "water_fraction" in indicators

    # Verify outputs exist
    outputs = result["outputs"]
    for key, rel_path in outputs.items():
        assert (job_dir / rel_path).exists(), f"Output file {rel_path} was not created"

    # Verify builtup mask is binary (0 or 1)
    with rasterio.open(job_dir / outputs["builtup_mask_geotiff"]) as src:
        builtup = src.read(1)
        unique_vals = set(np.unique(builtup))
        assert unique_vals.issubset({0, 1})


def test_urban_manifest_and_report_integration(tmp_path: Path):
    job_dir = tmp_path / "ps_urban_manifest"
    native_path = job_dir / "input" / "native.tif"
    sr_path = job_dir / "super_resolution" / "sr.tif"

    _create_synthetic_multispectral_geotiff(native_path, height=128, width=128)
    _create_synthetic_multispectral_geotiff(sr_path, height=128, width=128)

    urban_result = run_urban_pipeline(native_path, sr_path, job_dir, device="cpu")
    report_path = job_dir / "report" / "report.json"

    report = write_report(
        report_path,
        job_id="ps_urban_test",
        input_metadata={"width": 128, "height": 128, "bands": 4, "band_names": ["B02", "B03", "B04", "B08"], "crs": "EPSG:32632", "compatible": True},
        preprocessing=[],
        runtime_seconds=2.0,
        device="cpu",
        output_files=urban_result["outputs"],
        application="urban",
        application_data=urban_result,
    )

    manifest = write_manifest(job_dir, "ps_urban_test", report)
    assert manifest["application"] == "urban"
    artifact_paths = [a["path"] for a in manifest["artifacts"]]
    assert "application/urban/segmentation_native.tif" in artifact_paths
    assert "application/urban/segmentation_sr.tif" in artifact_paths
    assert "application/urban/builtup.tif" in artifact_paths
    assert "application/urban/urban_difference.tif" in artifact_paths
