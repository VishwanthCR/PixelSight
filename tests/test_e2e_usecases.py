"""
PixelSight Real End-to-End Smoke Tests & Ground Truth Pipeline Verification
===========================================================================
Implements Section 34 & Section 37 acceptance criteria:
- TEST A: Crop Monitoring (NDVI, crop metrics, crop report, no disaster outputs)
- TEST B: Urban Analysis (WorldCover segmentation, built-up mask, urban report, no crop/disaster outputs)
- TEST C: Disaster Management (Before SR, After SR, Alignment, Change Map, Uncertainty, Disaster Report, no crop/urban outputs)
- TEST D: Disaster Ground Truth (Validated GT annotation, common 2.5m grid, TP, FP, FN, TN, IoU, F1, Precision, Recall, Confusion Matrix)
- TEST E: Disaster Without Ground Truth (Change analysis works, label_status: NONE, message: 'Change detection without ground-truth validation', no fake damage accuracy)
"""

import json
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from backend.app.services.core_engine import PixelSightEngine
from backend.app.services.crop import run_crop_analysis
from backend.app.services.urban import run_urban_analysis
from backend.app.services.disaster import align_temporal_pair, run_disaster_analysis
from backend.app.services.ground_truth.disaster_evaluator import disaster_gt_evaluator
from backend.app.services.ground_truth.schema import LabelStatus, ValidationStatus


@pytest.fixture
def test_env(tmp_path: Path, monkeypatch):
    """Creates synthetic Sentinel-2 4-band rasters and directory structures for testing."""
    monkeypatch.setenv("PIXELSIGHT_TEST_MODE", "1")
    crop_dir = tmp_path / "crop_job"
    urban_dir = tmp_path / "urban_job"
    disaster_dir = tmp_path / "disaster_job"

    for d in (crop_dir, urban_dir, disaster_dir):
        (d / "input").mkdir(parents=True, exist_ok=True)
        (d / "preprocessing").mkdir(parents=True, exist_ok=True)
        (d / "super_resolution").mkdir(parents=True, exist_ok=True)
        (d / "application").mkdir(parents=True, exist_ok=True)
        (d / "report").mkdir(parents=True, exist_ok=True)

    # 4-band 10m Sentinel-2 raster (B02, B03, B04, B08)
    transform = from_origin(80.0, 13.0, 10.0, 10.0)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": None,
        "width": 64,
        "height": 64,
        "count": 4,
        "crs": "EPSG:32644",
        "transform": transform,
    }

    # Pre-event scene: baseline reflectance
    rng = np.random.RandomState(42)
    pre_data = rng.uniform(0.05, 0.45, size=(4, 64, 64)).astype(np.float32)
    # B08 (NIR) high in top half (vegetation)
    pre_data[3, :32, :] = 0.55
    # B04 (Red) low in top half
    pre_data[2, :32, :] = 0.10

    pre_file = disaster_dir / "input" / "pre_event.tif"
    with rasterio.open(pre_file, "w", **profile) as dst:
        dst.write(pre_data)
        for idx, name in enumerate(("B02", "B03", "B04", "B08"), start=1):
            dst.set_band_description(idx, name)

    # Post-event scene: flooded / altered bottom-right quadrant
    post_data = np.copy(pre_data)
    # Significant drop in NIR and Red in bottom right quadrant (simulating flooding/damage)
    post_data[:, 32:, 32:] = 0.05

    post_file = disaster_dir / "input" / "post_event.tif"
    with rasterio.open(post_file, "w", **profile) as dst:
        dst.write(post_data)
        for idx, name in enumerate(("B02", "B03", "B04", "B08"), start=1):
            dst.set_band_description(idx, name)

    return {
        "crop_dir": crop_dir,
        "urban_dir": urban_dir,
        "disaster_dir": disaster_dir,
        "pre_file": pre_file,
        "post_file": post_file,
        "profile": profile,
    }


def test_a_crop_pipeline(test_env):
    """TEST A: Crop Monitoring produces only crop-relevant outputs and no disaster results."""
    crop_dir = test_env["crop_dir"]
    input_file = test_env["pre_file"]

    # Preprocess & Super-resolve (using mock/test mode)
    norm_path = crop_dir / "preprocessing" / "normalized.tif"
    sr_path = crop_dir / "super_resolution" / "sr.tif"
    PixelSightEngine.preprocess(input_file, norm_path)
    PixelSightEngine.enhance(norm_path, sr_path)

    # Run Crop Analysis
    crop_results = run_crop_analysis(norm_path, sr_path, crop_dir)

    # Assert Crop Outputs exist
    assert "statistics" in crop_results
    assert "consistency_metrics" in crop_results
    assert "ndvi_native_geotiff" in crop_results["outputs"]
    assert "ndvi_sr_geotiff" in crop_results["outputs"]
    assert (crop_dir / "application" / "crop" / "ndvi_sr.tif").exists()

    # Assert NO disaster change detection outputs exist
    assert "disaster" not in crop_results["outputs"]
    assert not (crop_dir / "application" / "disaster").exists()


def test_b_urban_pipeline(test_env):
    """TEST B: Urban Analysis produces only urban-relevant outputs and no crop/disaster outputs."""
    urban_dir = test_env["urban_dir"]
    input_file = test_env["pre_file"]

    norm_path = urban_dir / "preprocessing" / "normalized.tif"
    sr_path = urban_dir / "super_resolution" / "sr.tif"
    PixelSightEngine.preprocess(input_file, norm_path)
    PixelSightEngine.enhance(norm_path, sr_path)

    from backend.app.services.urban import run_urban_pipeline
    urban_results = run_urban_pipeline(norm_path, sr_path, urban_dir)

    # Assert Urban Outputs exist
    assert "indicators" in urban_results or "class_distribution" in urban_results
    assert (urban_dir / "application" / "urban" / "segmentation_sr.tif").exists()

    # Assert NO crop or disaster outputs exist
    assert not (urban_dir / "application" / "crop").exists()
    assert not (urban_dir / "application" / "disaster").exists()


def test_c_disaster_pipeline(test_env):
    """TEST C: Disaster Management processes temporal pair, change map, uncertainty, and metrics."""
    disaster_dir = test_env["disaster_dir"]
    pre_file = test_env["pre_file"]
    post_file = test_env["post_file"]

    # Align temporal pair
    aligned_pre = disaster_dir / "preprocessing" / "aligned_pre.tif"
    aligned_post = disaster_dir / "preprocessing" / "aligned_post.tif"
    align_info = align_temporal_pair(pre_file, post_file, aligned_pre, aligned_post)
    assert align_info["aligned"] is True

    # 4x Super-resolution on both scenes
    sr_pre = disaster_dir / "application" / "disaster" / "pre_event" / "sr_pre.tif"
    sr_post = disaster_dir / "application" / "disaster" / "post_event" / "sr_post.tif"
    PixelSightEngine.enhance(aligned_pre, sr_pre)
    PixelSightEngine.enhance(aligned_post, sr_post)
    assert sr_pre.exists()
    assert sr_post.exists()

    # Generate uncertainty for both scenes
    pre_unc_tif = disaster_dir / "application" / "disaster" / "pre_event" / "unc_pre.tif"
    pre_unc_png = disaster_dir / "application" / "disaster" / "pre_event" / "unc_pre.png"
    post_unc_tif = disaster_dir / "application" / "disaster" / "post_event" / "unc_post.tif"
    post_unc_png = disaster_dir / "application" / "disaster" / "post_event" / "unc_post.png"
    PixelSightEngine.uncertainty(aligned_pre, sr_pre, pre_unc_tif, pre_unc_png)
    PixelSightEngine.uncertainty(aligned_post, sr_post, post_unc_tif, post_unc_png)

    # Run Disaster Analysis
    disaster_results = run_disaster_analysis(
        sr_pre,
        sr_post,
        pre_unc_tif,
        post_unc_tif,
        disaster_dir,
        change_threshold=0.10,
    )

    # Assert Disaster Outputs exist
    assert "statistics" in disaster_results
    assert disaster_results["statistics"]["changed_pixels"] > 0
    assert (disaster_dir / "application" / "disaster" / "change_map.tif").exists()
    assert (disaster_dir / "application" / "disaster" / "affected_area.tif").exists()
    assert (disaster_dir / "application" / "disaster" / "previews" / "pre_sr_preview.png").exists()
    assert (disaster_dir / "application" / "disaster" / "previews" / "post_sr_preview.png").exists()

    # Assert NO crop or urban outputs exist
    assert not (disaster_dir / "application" / "crop").exists()
    assert not (disaster_dir / "application" / "urban").exists()


def test_d_disaster_with_ground_truth(test_env):
    """TEST D: Disaster Ground Truth evaluates against validated ground-truth raster on common grid."""
    disaster_dir = test_env["disaster_dir"]
    change_map_tif = disaster_dir / "application" / "disaster" / "change_map.tif"

    if not change_map_tif.exists():
        test_c_disaster_pipeline(test_env)

    # Create synthetic ground truth damage raster matching change_map geometry
    with rasterio.open(change_map_tif) as src:
        gt_profile = src.profile.copy()
        h, w = src.height, src.width

    # Class 1 = confirmed damage, 0 = unchanged
    gt_data = np.zeros((h, w), dtype=np.uint8)
    # Mark bottom-right quadrant as actual damaged area
    gt_data[h // 2 :, w // 2 :] = 1

    gt_dir = disaster_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_tif = gt_dir / "ground_truth.tif"

    gt_profile.update(dtype="uint8", count=1, nodata=255)
    with rasterio.open(gt_tif, "w", **gt_profile) as dst:
        dst.write(gt_data, 1)

    gt_meta = {
        "validation_status": ValidationStatus.VALIDATED.value,
        "source": "Independent Disaster Damage Mapping Agency",
    }

    eval_result = disaster_gt_evaluator.evaluate(
        change_map_path=change_map_tif,
        ground_truth_path=gt_tif,
        change_threshold=0.10,
        evaluation_grid="2.5m",
        metadata=gt_meta,
    )

    assert eval_result["available"] is True
    assert eval_result["label_status"] == LabelStatus.GROUND_TRUTH.value
    assert "Evaluated against validated disaster ground-truth labels" in eval_result["message"]

    metrics = eval_result["metrics"]
    assert metrics is not None
    assert "iou" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "tp" in metrics
    assert "fp" in metrics
    assert "fn" in metrics
    assert "tn" in metrics
    assert metrics["tp"] > 0
    assert metrics["iou"] > 0.0


def test_e_disaster_without_ground_truth(test_env):
    """TEST E: When no ground truth exists, change analysis runs but damage accuracy/IoU/F1 are withheld."""
    disaster_dir = test_env["disaster_dir"]
    change_map_tif = disaster_dir / "application" / "disaster" / "change_map.tif"

    if not change_map_tif.exists():
        test_c_disaster_pipeline(test_env)

    # Evaluate without any ground truth file
    eval_result = disaster_gt_evaluator.evaluate(
        change_map_path=change_map_tif,
        ground_truth_path=None,
        change_threshold=0.10,
        evaluation_grid="2.5m",
        metadata={"validation_status": "none"},
    )

    assert eval_result["available"] is False
    assert eval_result["label_status"] == LabelStatus.NONE.value
    assert "Change detection without ground-truth validation." in eval_result["message"]
    # Metrics must be None to prevent displaying fake scientific claims
    assert eval_result["metrics"] is None
    assert eval_result["confusion_matrix"] is None
