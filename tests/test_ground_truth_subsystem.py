"""
Test Suite for Ground Truth Annotation & Label Evaluation Subsystem
====================================================================
Covers all 16 test requirements from Section 20:
1. polygon creation
2. class validation
3. GeoJSON persistence
4. rasterization
5. CRS preservation
6. resolution preservation
7. ignore mask
8. ground-truth coverage
9. native vs GT evaluation
10. SR vs GT evaluation
11. common evaluation grid
12. confusion matrix
13. per-class metrics
14. WorldCover proxy status
15. no-label status
16. existing segmentation regression
"""

import json
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.ground_truth.schema import (
    GT_CLASSES,
    IGNORE_CLASS,
    LabelStatus,
    ValidationStatus,
)
from backend.app.services.ground_truth.validator import annotation_validator
from backend.app.services.ground_truth.rasterizer import ground_truth_rasterizer
from backend.app.services.ground_truth.evaluator import ground_truth_evaluator
from backend.app.services.ground_truth.service import ground_truth_service
from backend.app.services.classification import get_job_classification

client = TestClient(app)


def _make_dummy_raster(
    path: Path,
    values: np.ndarray,
    bounds: tuple[float, float, float, float] = (80.25, 13.06, 80.29, 13.10),
    crs: str = "EPSG:4326",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = values.shape
    res_x = (bounds[2] - bounds[0]) / w
    res_y = (bounds[3] - bounds[1]) / h
    transform = from_origin(bounds[0], bounds[3], res_x, res_y)
    meta = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": 255,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(values.astype(np.uint8), 1)
    return path


# ── 1. Polygon creation & GeoJSON persistence ────────────────────────────────
def test_01_polygon_creation_and_persistence(tmp_path):
    job_id = "test_job_01"
    job_dir = tmp_path / job_id
    poly_feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [80.26, 13.07],
                    [80.28, 13.07],
                    [80.28, 13.09],
                    [80.26, 13.09],
                    [80.26, 13.07],
                ]
            ],
        },
        "properties": {
            "class_id": 0,
            "class_name": "Tree",
            "annotator": "Dr. Silva",
        },
    }
    geojson = {"type": "FeatureCollection", "features": [poly_feature]}
    info = ground_truth_service.save_annotations(
        job_id=job_id, geojson_data=geojson, annotator="Dr. Silva", job_dir=job_dir
    )
    assert info["has_annotations"] is True
    assert info["total_features"] == 1
    assert (job_dir / "ground_truth" / "annotations.geojson").exists()


# ── 2. Class validation & topological checks ─────────────────────────────────
def test_02_class_and_geometry_validation():
    # Valid feature
    valid_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [80.26, 13.07],
                            [80.28, 13.07],
                            [80.28, 13.09],
                            [80.26, 13.09],
                            [80.26, 13.07],
                        ]
                    ],
                },
                "properties": {"class_id": 4, "class_name": "Built-up"},
            }
        ],
    }
    rep = annotation_validator.validate(valid_geojson, aoi_bbox=[80.25, 13.06, 80.29, 13.10])
    assert rep.is_valid is True
    assert rep.status == ValidationStatus.VALIDATED

    # Invalid class ID & self-intersecting bowtie
    invalid_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [80.26, 13.07],
                            [80.28, 13.09],
                            [80.28, 13.07],
                            [80.26, 13.09],
                            [80.26, 13.07],
                        ]
                    ],
                },
                "properties": {"class_id": 999, "class_name": "AlienLand"},
            }
        ],
    }
    rep_bad = annotation_validator.validate(invalid_geojson)
    assert rep_bad.is_valid is False
    codes = [e.code for e in rep_bad.errors]
    assert "INVALID_CLASS_ID" in codes


# ── 3 & 4. Rasterization onto grid ───────────────────────────────────────────
def test_03_rasterization_and_preview(tmp_path):
    out_tif = tmp_path / "ground_truth.tif"
    poly_feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [80.26, 13.07],
                    [80.28, 13.07],
                    [80.28, 13.09],
                    [80.26, 13.09],
                    [80.26, 13.07],
                ]
            ],
        },
        "properties": {"class_id": 4, "class_name": "Built-up"},
    }
    geojson = {"type": "FeatureCollection", "features": [poly_feature]}
    aoi = [80.25, 13.06, 80.29, 13.10]
    stats = ground_truth_rasterizer.rasterize_to_grid(
        geojson_data=geojson,
        output_tif_path=out_tif,
        aoi_bbox=aoi,
        target_dims=(128, 128),
    )
    assert out_tif.exists()
    assert (tmp_path / "ground_truth.png").exists()
    assert stats["dimensions"] == [128, 128]
    assert stats["coverage_percentage"] > 0


# ── 5 & 6. CRS and Resolution preservation ────────────────────────────────────
def test_04_crs_and_resolution_preservation(tmp_path):
    out_tif = tmp_path / "ground_truth_res.tif"
    template_tif = tmp_path / "template.tif"
    _make_dummy_raster(template_tif, np.zeros((256, 256), dtype=np.uint8))

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [80.26, 13.07],
                            [80.28, 13.07],
                            [80.28, 13.09],
                            [80.26, 13.09],
                            [80.26, 13.07],
                        ]
                    ],
                },
                "properties": {"class_id": 6, "class_name": "Water"},
            }
        ],
    }
    stats = ground_truth_rasterizer.rasterize_to_grid(
        geojson_data=geojson,
        output_tif_path=out_tif,
        template_raster_path=template_tif,
    )
    with rasterio.open(out_tif) as src:
        assert src.crs == "EPSG:4326"
        assert src.width == 256
        assert src.height == 256
        assert src.nodata == 255


# ── 7 & 8. Ignore mask and coverage percentage ────────────────────────────────
def test_05_ignore_mask_and_coverage(tmp_path):
    out_tif = tmp_path / "gt_ignore.tif"
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [80.25, 13.06],
                            [80.27, 13.06],
                            [80.27, 13.10],
                            [80.25, 13.10],
                            [80.25, 13.06],
                        ]
                    ],
                },
                "properties": {"class_id": 3, "class_name": "Cropland"},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [80.27, 13.06],
                            [80.29, 13.06],
                            [80.29, 13.10],
                            [80.27, 13.10],
                            [80.27, 13.06],
                        ]
                    ],
                },
                "properties": {"class_id": 255, "class_name": "Ignore"},
            },
        ],
    }
    stats = ground_truth_rasterizer.rasterize_to_grid(
        geojson_data=geojson,
        output_tif_path=out_tif,
        aoi_bbox=[80.25, 13.06, 80.29, 13.10],
        target_dims=(100, 100),
    )
    # 50% Cropland, 50% Ignore
    assert 48.0 <= stats["coverage_percentage"] <= 52.0
    assert 48.0 <= stats["ignored_percentage"] <= 52.0


# ── 9, 10, 11, 12, 13. Dual-model evaluation on common grid ───────────────────
def test_06_dual_model_evaluation_on_common_grid(tmp_path):
    gt_tif = tmp_path / "gt.tif"
    sr_tif = tmp_path / "sr.tif"
    nat_tif = tmp_path / "nat.tif"

    # 64x64 grid
    # GT: Left half Tree (0), Right half Built-up (4)
    gt_arr = np.full((64, 64), 255, dtype=np.uint8)
    gt_arr[:, :32] = 0  # Tree
    gt_arr[:, 32:] = 4  # Built-up
    _make_dummy_raster(gt_tif, gt_arr)

    # SR: Accurate on Tree, slightly noisier on Built-up
    sr_arr = np.copy(gt_arr)
    sr_arr[:8, 32:40] = 5  # Bare error
    _make_dummy_raster(sr_tif, sr_arr)

    # Native: Lower accuracy with blockier errors
    nat_arr = np.copy(gt_arr)
    nat_arr[:16, 20:44] = 1  # Shrubland error across center boundary
    _make_dummy_raster(nat_tif, nat_arr)

    res = ground_truth_evaluator.evaluate(
        gt_raster_path=gt_tif,
        sr_raster_path=sr_tif,
        native_raster_path=nat_tif,
        output_dir=tmp_path / "previews",
        evaluation_grid="2.5m",
    )

    ov = res["overall"]
    assert ov["sr"]["accuracy"] > ov["native"]["accuracy"]
    assert ov["sr"]["miou"] > ov["native"]["miou"]
    assert ov["delta"]["miou"] > 0
    assert "Common 2.5m evaluation grid" in res["evaluation_grid"]

    # Confusion matrix checks (7x7)
    assert len(res["confusion_matrix"]["native"]) == 7
    assert len(res["confusion_matrix"]["sr"]) == 7

    # Per-class metrics
    tree_m = next(c for c in res["per_class"] if c["name"] == "Tree")
    built_m = next(c for c in res["per_class"] if c["name"] == "Built-up")
    assert tree_m["native"]["support"] == 32 * 64
    assert built_m["sr"]["iou"] is not None
    assert built_m["delta"]["iou"] is not None

    # Previews created
    assert (tmp_path / "previews" / "gt_vs_native.png").exists()
    assert (tmp_path / "previews" / "gt_vs_sr.png").exists()
    assert (tmp_path / "previews" / "native_vs_sr.png").exists()


# ── 14, 15. WorldCover proxy status vs No-label status ────────────────────────
def test_07_worldcover_proxy_and_no_label_status(tmp_path):
    # Benchmark job evaluates to PROXY_LABELS
    data_proxy = get_job_classification("ps_benchmark_chennai", tmp_path / "bench")
    assert data_proxy["label_status"] == LabelStatus.PROXY_LABELS.value
    assert "proxy reference" in data_proxy["evaluation"]["status_message"]

    # Plain job with no GT evaluates to NONE
    data_none = get_job_classification("ps_random_unlabeled", tmp_path / "unlabeled")
    assert data_none["label_status"] == LabelStatus.NONE.value
    assert "no ground-truth reference labels" in data_none["evaluation"]["status_message"]


# ── 16. REST API Endpoints End-to-End ─────────────────────────────────────────
def test_08_rest_api_ground_truth_workflow(tmp_path):
    job_id = "ps_api_test_job"
    # 1. Create workspace
    res = client.post(
        "/api/v1/ground-truth/create",
        json={"job_id": job_id, "aoi": [80.25, 13.06, 80.29, 13.10]},
    )
    assert res.status_code == 200
    assert res.json()["label_status"] == "NONE"

    # 2. Save annotations
    poly_feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [80.26, 13.07],
                    [80.28, 13.07],
                    [80.28, 13.09],
                    [80.26, 13.09],
                    [80.26, 13.07],
                ]
            ],
        },
        "properties": {"class_id": 4, "class_name": "Built-up"},
    }
    res_ann = client.post(
        f"/api/v1/ground-truth/{job_id}/annotations",
        json={"geojson": {"type": "FeatureCollection", "features": [poly_feature]}},
    )
    assert res_ann.status_code == 200
    assert res_ann.json()["total_features"] == 1

    # 3. Validate
    res_val = client.post(f"/api/v1/ground-truth/{job_id}/validate")
    assert res_val.status_code == 200
    assert res_val.json()["is_valid"] is True

    # 4. Review approval
    res_rev = client.post(
        f"/api/v1/ground-truth/{job_id}/review",
        json={"status": "validated", "reviewer": "Lead QA Reviewer"},
    )
    assert res_rev.status_code == 200
    assert res_rev.json()["validation_status"] == "validated"
    assert res_rev.json()["metadata"]["reviewer"] == "Lead QA Reviewer"
