from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from backend.app.services.disaster import align_temporal_pair, run_disaster_analysis
from backend.app.services.reporting import write_manifest, write_report


def _create_synthetic_multispectral_geotiff(
    path: Path,
    height: int = 64,
    width: int = 64,
    crs: str = "EPSG:32632",
    pixel_size: float = 10.0,
    offset_val: float = 0.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    b02 = np.full((height, width), 0.1 + offset_val, dtype=np.float32)
    b03 = np.full((height, width), 0.15 + offset_val, dtype=np.float32)
    b04 = np.full((height, width), 0.2 + offset_val, dtype=np.float32)
    b08 = np.full((height, width), 0.4 + offset_val, dtype=np.float32)
    bands = np.clip(np.stack([b02, b03, b04, b08], axis=0), 0.0, 1.0)

    transform = from_origin(100.0, 200.0, pixel_size, pixel_size)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(bands)


def _create_synthetic_uncertainty_geotiff(path: Path, height: int = 64, width: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unc = np.linspace(0.1, 0.8, height * width, dtype=np.float32).reshape(height, width)
    transform = from_origin(100.0, 200.0, 2.5, 2.5)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:32632",
        transform=transform,
    ) as dst:
        dst.write(unc, 1)


def test_disaster_temporal_alignment(tmp_path: Path):
    pre_path = tmp_path / "pre.tif"
    post_path = tmp_path / "post.tif"
    aligned_pre = tmp_path / "aligned_pre.tif"
    aligned_post = tmp_path / "aligned_post.tif"

    _create_synthetic_multispectral_geotiff(pre_path, height=32, width=32)
    # Post has different dimensions
    _create_synthetic_multispectral_geotiff(post_path, height=48, width=48)

    alignment = align_temporal_pair(pre_path, post_path, aligned_pre, aligned_post)

    assert alignment["aligned"] is True
    assert aligned_pre.exists()
    assert aligned_post.exists()

    with rasterio.open(aligned_pre) as p1, rasterio.open(aligned_post) as p2:
        assert p1.shape == p2.shape
        assert p1.crs == p2.crs


def test_disaster_change_detection_and_uncertainty(tmp_path: Path):
    job_dir = tmp_path / "ps_disaster_test"
    sr_pre = job_dir / "sr_pre.tif"
    sr_post = job_dir / "sr_post.tif"
    pre_unc = job_dir / "pre_unc.tif"
    post_unc = job_dir / "post_unc.tif"

    _create_synthetic_multispectral_geotiff(sr_pre, height=64, width=64, offset_val=0.0)
    # Introduce significant change in post-event
    _create_synthetic_multispectral_geotiff(sr_post, height=64, width=64, offset_val=0.3)
    _create_synthetic_uncertainty_geotiff(pre_unc, height=64, width=64)
    _create_synthetic_uncertainty_geotiff(post_unc, height=64, width=64)

    result = run_disaster_analysis(
        sr_pre,
        sr_post,
        pre_unc,
        post_unc,
        job_dir,
        change_threshold=0.15,
    )

    assert result["application"] == "disaster"
    assert result["scientific_status"] == "Research analysis — no independent ground truth"

    stats = result["statistics"]
    assert stats["changed_pixels"] > 0
    assert stats["change_percentage"] > 0
    assert "reliability_breakdown" in stats

    rel = stats["reliability_breakdown"]
    assert "lower_uncertainty_pixels" in rel
    assert "moderate_uncertainty_pixels" in rel
    assert "high_uncertainty_pixels" in rel

    # Verify outputs
    outputs = result["outputs"]
    for key, rel_path in outputs.items():
        assert (job_dir / rel_path).exists(), f"Disaster output {rel_path} was not created"


def test_disaster_manifest_and_report_integration(tmp_path: Path):
    job_dir = tmp_path / "ps_disaster_manifest"
    sr_pre = job_dir / "sr_pre.tif"
    sr_post = job_dir / "sr_post.tif"
    pre_unc = job_dir / "pre_unc.tif"
    post_unc = job_dir / "post_unc.tif"

    _create_synthetic_multispectral_geotiff(sr_pre, height=64, width=64, offset_val=0.0)
    _create_synthetic_multispectral_geotiff(sr_post, height=64, width=64, offset_val=0.25)
    _create_synthetic_uncertainty_geotiff(pre_unc, height=64, width=64)
    _create_synthetic_uncertainty_geotiff(post_unc, height=64, width=64)

    disaster_result = run_disaster_analysis(sr_pre, sr_post, pre_unc, post_unc, job_dir)
    report_path = job_dir / "report" / "report.json"

    alignment_info = {"aligned": True, "grid_dimensions": [64, 64], "crs": "EPSG:32632", "transformations": ["matched"]}
    report = write_report(
        report_path,
        job_id="ps_disaster_test",
        input_metadata={"width": 64, "height": 64, "bands": 4, "band_names": ["B02", "B03", "B04", "B08"], "crs": "EPSG:32632", "compatible": True},
        preprocessing=[],
        runtime_seconds=3.5,
        device="cpu",
        output_files=disaster_result["outputs"],
        application="disaster",
        application_data=disaster_result,
        disaster_metadata=alignment_info,
    )

    manifest = write_manifest(job_dir, "ps_disaster_test", report)
    assert manifest["application"] == "disaster"
    assert "disaster_metadata" in manifest
    artifact_paths = [a["path"] for a in manifest["artifacts"]]
    assert "application/disaster/change_map.tif" in artifact_paths
    assert "application/disaster/affected_area.tif" in artifact_paths
    assert "application/disaster/uncertainty.tif" in artifact_paths
