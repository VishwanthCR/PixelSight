"""
Comprehensive Test Suite for Automatic HR Reference Discovery & Evaluation Subsystem
======================================================================================
Covers all 18 scientific and engineering test requirements specified in Section 22:
1. AOI intersects reference
2. AOI outside reference
3. Partial overlap
4. Full overlap
5. Temporal matching
6. Temporal mismatch
7. Compatible bands
8. Incompatible bands
9. GeoTIFF metadata discovery
10. PNG/JPEG without metadata
11. Reference unavailable
12. Cache hit
13. Cache miss
14. Alignment
15. Metric eligibility
16. No-reference fallback
17. Batch reference discovery
18. API response schema
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.reference.registry import (
    ReferenceRegistry,
    ReferenceSource,
    ReferenceTile,
)
from backend.app.services.reference.spatial_match import SpatialMatcher
from backend.app.services.reference.temporal_match import TemporalMatcher
from backend.app.services.reference.spectral_match import SpectralMatcher
from backend.app.services.reference.downloader import ReferenceDownloader
from backend.app.services.reference.alignment import ReferenceAligner
from backend.app.services.reference.evaluator import ReferenceEvaluator
from backend.app.services.reference.discovery import ReferenceDiscoveryService


client = TestClient(app)


def _create_dummy_geotiff(
    path: Path,
    crs: str = "EPSG:4326",
    bounds: tuple[float, float, float, float] = (80.25, 13.06, 80.29, 13.10),
    channels: int = 4,
    width: int = 64,
    height: int = 64,
) -> Path:
    """Helper to write a small test GeoTIFF with known metadata."""
    res_x = (bounds[2] - bounds[0]) / width
    res_y = (bounds[3] - bounds[1]) / height
    transform = from_origin(bounds[0], bounds[3], res_x, res_y)
    data = np.full((channels, height, width), 0.35, dtype=np.float32)

    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": channels,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data)
        if channels == 4:
            for i, name in enumerate(["B02", "B03", "B04", "B08"], 1):
                dst.set_band_description(i, name)
    return path


# ── Test 1: AOI Intersects Reference ──────────────────────────────────────────
def test_01_aoi_intersects_reference():
    matcher = SpatialMatcher(min_overlap_pct=50.0)
    aoi_bbox = [80.26, 13.07, 80.28, 13.09]
    ref_bbox = [80.25, 13.06, 80.29, 13.10]
    res = matcher.compute_match(aoi=aoi_bbox, candidate_bbox=ref_bbox)
    assert res.intersects is True
    assert res.overlap_pct > 99.0


# ── Test 2: AOI Outside Reference ─────────────────────────────────────────────
def test_02_aoi_outside_reference():
    matcher = SpatialMatcher(min_overlap_pct=50.0)
    aoi_bbox = [70.0, 20.0, 70.1, 20.1]
    ref_bbox = [80.25, 13.06, 80.29, 13.10]
    res = matcher.compute_match(aoi=aoi_bbox, candidate_bbox=ref_bbox)
    assert res.intersects is False
    assert res.overlap_pct == 0.0


# ── Test 3: Partial Overlap ───────────────────────────────────────────────────
def test_03_partial_overlap():
    matcher = SpatialMatcher(min_overlap_pct=50.0)
    # Half of the AOI lies inside ref_bbox
    aoi_bbox = [80.27, 13.06, 80.31, 13.10]  # width 0.04; 80.27-80.29 is inside (0.02)
    ref_bbox = [80.25, 13.06, 80.29, 13.10]
    res = matcher.compute_match(aoi=aoi_bbox, candidate_bbox=ref_bbox)
    assert 45.0 <= res.overlap_pct <= 55.0


# ── Test 4: Full Overlap ──────────────────────────────────────────────────────
def test_04_full_overlap():
    matcher = SpatialMatcher(min_overlap_pct=50.0)
    aoi_bbox = [80.26, 13.07, 80.28, 13.09]
    ref_bbox = [80.25, 13.06, 80.29, 13.10]
    res = matcher.compute_match(aoi=aoi_bbox, candidate_bbox=ref_bbox)
    assert res.overlap_pct == 100.0


# ── Test 5: Temporal Matching (Close / Exact) ──────────────────────────────────
def test_05_temporal_matching():
    matcher = TemporalMatcher()
    res_exact = matcher.evaluate("2023-06-15", "2023-06-15")
    assert res_exact.temporal_match_status == "EXACT"
    assert res_exact.temporal_difference_days == 0
    assert res_exact.is_acceptable is True

    res_close = matcher.evaluate("2023-06-15", "2023-06-18")
    assert res_close.temporal_match_status == "CLOSE"
    assert res_close.temporal_difference_days == 3
    assert res_close.is_acceptable is True


# ── Test 6: Temporal Mismatch (Large Difference) ──────────────────────────────
def test_06_temporal_mismatch():
    matcher = TemporalMatcher(acceptable_with_caveat_days=30)
    res = matcher.evaluate("2023-06-15", "2022-01-01")
    assert res.temporal_match_status == "LARGE_DIFFERENCE"
    assert res.temporal_difference_days > 300
    assert res.is_acceptable is False


# ── Test 7: Compatible Bands (Full 4-band VNIR) ───────────────────────────────
def test_07_compatible_bands():
    matcher = SpectralMatcher()
    res = matcher.evaluate(["B02", "B03", "B04", "B08"])
    assert res.compatibility == "FULL"
    assert res.has_rgb is True
    assert res.has_nir is True
    assert "sam" in res.eligible_metrics
    assert "psnr" in res.eligible_metrics


# ── Test 8: Incompatible Bands / RGB-only NIR Handling ────────────────────────
def test_08_incompatible_and_rgb_bands():
    matcher = SpectralMatcher()
    # RGB only without NIR
    res_rgb = matcher.evaluate(["RED", "GREEN", "BLUE"])
    assert res_rgb.compatibility == "RGB_ONLY"
    assert res_rgb.has_rgb is True
    assert res_rgb.has_nir is False
    assert "psnr" in res_rgb.eligible_metrics
    assert "sam" not in res_rgb.eligible_metrics
    assert "sam" in res_rgb.ineligible_metrics

    # Completely incompatible
    res_incomp = matcher.evaluate(["THERMAL_T1", "SWIR2"])
    assert res_incomp.compatibility == "INCOMPATIBLE"
    assert len(res_incomp.eligible_metrics) == 0


# ── Test 9: GeoTIFF Metadata Discovery ────────────────────────────────────────
def test_09_geotiff_metadata_discovery(tmp_path):
    tif_path = tmp_path / "scene.tif"
    # Matches Chennai regional reference tile
    _create_dummy_geotiff(tif_path, bounds=(80.26, 13.07, 80.28, 13.09))

    service = ReferenceDiscoveryService()
    res = service.discover_for_geotiff(tif_path)
    assert res.available is True
    assert res.source == "india_regional_reference"
    assert res.spatial_overlap > 95.0


# ── Test 10: PNG/JPEG Without Metadata Rejection ──────────────────────────────
def test_10_png_jpeg_without_metadata(tmp_path):
    png_path = tmp_path / "photo.png"
    png_path.write_bytes(b"\x89PNG\r\n\x1a\nfakecontent")

    service = ReferenceDiscoveryService()
    res = service.discover_for_geotiff(png_path)
    assert res.available is False
    assert res.match_status == "NON_GEOSPATIAL"
    assert "Geospatial metadata is unavailable" in res.selection_reason


# ── Test 11: Reference Unavailable ────────────────────────────────────────────
def test_11_reference_unavailable():
    service = ReferenceDiscoveryService()
    # Coordinates in remote ocean
    res = service.discover(aoi=[0.0, 0.0, 0.05, 0.05])
    assert res.available is False
    assert res.match_status == "NO_ELIGIBLE_REFERENCE"
    assert "No compatible HR reference was found" in res.selection_reason


# ── Test 12: Cache Hit ────────────────────────────────────────────────────────
def test_12_cache_hit(tmp_path):
    cache_dir = tmp_path / "cache"
    downloader = ReferenceDownloader(cache_dir=cache_dir)
    source_tif = tmp_path / "sample.tif"
    _create_dummy_geotiff(source_tif)

    # First retrieval (caches the file)
    p1 = downloader.retrieve_reference(source_tif, "test_source", "tile_1", bbox=(0, 0, 1, 1))
    assert p1 is not None
    assert p1.exists()

    # Second retrieval (cache hit)
    cache_key = downloader.generate_cache_key("test_source", "tile_1", bbox=(0, 0, 1, 1))
    assert downloader.is_cached(cache_key) is True
    p2 = downloader.retrieve_reference(source_tif, "test_source", "tile_1", bbox=(0, 0, 1, 1))
    assert p1 == p2


# ── Test 13: Cache Miss & Local Retrieval ─────────────────────────────────────
def test_13_cache_miss_and_retrieval(tmp_path):
    cache_dir = tmp_path / "cache"
    downloader = ReferenceDownloader(cache_dir=cache_dir)
    key = downloader.generate_cache_key("new_source", "tile_999")
    assert downloader.is_cached(key) is False


# ── Test 14: Spatial & Coordinate Alignment ───────────────────────────────────
def test_14_reference_alignment(tmp_path):
    ref_tif = tmp_path / "ref_10m.tif"
    template_tif = tmp_path / "template_10m.tif"
    _create_dummy_geotiff(ref_tif, width=32, height=32)
    _create_dummy_geotiff(template_tif, width=32, height=32)

    aligner = ReferenceAligner()
    out_dir = tmp_path / "aligned"
    aligned_path, report = aligner.align_reference(ref_tif, template_tif, out_dir, target_scale=4)

    assert aligned_path.exists()
    assert report.alignment_status == "SUCCESS"
    assert report.aligned_dimensions[1] == 128  # 32 * 4
    assert report.aligned_dimensions[2] == 128


# ── Test 15: Metric Eligibility Enforcement ───────────────────────────────────
def test_15_metric_eligibility(tmp_path):
    sr_tif = tmp_path / "sr.tif"
    ref_tif = tmp_path / "ref.tif"
    _create_dummy_geotiff(sr_tif, width=64, height=64)
    _create_dummy_geotiff(ref_tif, width=64, height=64)

    evaluator = ReferenceEvaluator()
    # SAM is marked ineligible
    metrics = evaluator.evaluate_reference_based(
        sr_path=sr_tif,
        aligned_reference_path=ref_tif,
        eligible_metrics=["psnr", "ssim"],
        ineligible_metrics={"sam": "Reference lacks NIR band"},
        provenance={"source": "RGB_Airborne"},
    )
    assert metrics["psnr"]["valid"] is True
    assert metrics["ssim"]["valid"] is True
    assert metrics["sam"]["valid"] is False
    assert "Reference lacks NIR band" in metrics["sam"]["reason"]


# ── Test 16: No-Reference Fallback ────────────────────────────────────────────
def test_16_no_reference_fallback():
    evaluator = ReferenceEvaluator()
    eval_dict = evaluator.build_no_reference_evaluation(
        reason="No compatible HR reference was found for this AOI."
    )
    assert eval_dict["status"] == "reference_unavailable"
    assert eval_dict["psnr"]["value"] is None
    assert eval_dict["psnr"]["valid"] is False
    assert eval_dict["ssim"]["value"] is None
    assert eval_dict["sam"]["value"] is None
    assert "uncertainty" in eval_dict["no_reference_metrics"]


# ── Test 17: Batch Reference Discovery ────────────────────────────────────────
def test_17_batch_reference_discovery(tmp_path):
    service = ReferenceDiscoveryService()
    # Item 1: In Chennai reference region
    r1 = service.discover(aoi=[80.26, 13.07, 80.28, 13.09])
    # Item 2: Outside any covered region
    r2 = service.discover(aoi=[10.0, 10.0, 10.1, 10.1])
    # Item 3: In Delhi reference region
    r3 = service.discover(aoi=[77.15, 28.60, 77.25, 28.70])

    assert r1.available is True
    assert r1.source == "india_regional_reference"

    assert r2.available is False
    assert r2.match_status == "NO_ELIGIBLE_REFERENCE"

    assert r3.available is True
    assert "DELHI" in r3.reference_id


# ── Test 18: API Response Schema Validation ───────────────────────────────────
def test_18_api_discovery_endpoint():
    # Valid India AOI query
    res = client.post(
        "/api/v1/evaluation/reference/discover",
        json={"aoi": [80.264, 13.076, 80.276, 13.088], "date": "2023-06-15"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["available"] is True
    assert data["source"] == "india_regional_reference"
    assert data["resolution_m"] == 2.5
    assert data["match_status"] == "ELIGIBLE"
    assert "limitations" in data
    assert "provenance" in data

    # Uncovered AOI query (No-Reference fallback)
    res_none = client.post(
        "/api/v1/evaluation/reference/discover",
        json={"aoi": [5.0, 5.0, 5.1, 5.1]},
    )
    assert res_none.status_code == 200
    data_none = res_none.json()
    assert data_none["available"] is False
    assert data_none["match_status"] == "NO_ELIGIBLE_REFERENCE"
    assert "No compatible HR reference was found" in data_none["selection_reason"]
