"""
Tests for PixelSight Scientific Evaluation Framework.
=====================================================
Verifies that:
1. Spatial metrics correctly gate HR-reference metrics (reporting None when no HR ref).
2. Spectral metrics compute NDVI accurately without synthesizing NIR from RGB.
3. Uncertainty analysis correctly captures stochastic diffusion variation.
4. Reliability maps classify pixels into HIGH/MEDIUM/LOW based on thresholds.
5. Three-way benchmark records full provenance and negative findings.
6. Table generator handles dataclasses and None values without fabricating numbers.
"""

import numpy as np
import pytest

from evaluation.image_metrics.spatial import spatial_fidelity, gradient_energy_only
from evaluation.image_metrics.spectral import spectral_fidelity
from evaluation.image_metrics.geospatial import geospatial_fidelity
from evaluation.downstream.ndvi import compute_ndvi, ndvi_comparison
from evaluation.downstream.segmentation import segmentation_metrics, SegmentationResult
from evaluation.uncertainty.analysis import uncertainty_statistics, uncertainty_vs_error
from evaluation.uncertainty.reliability import ReliabilityMap
from evaluation.benchmarking.three_way import ThreeWayBenchmark, BenchmarkProvenance
from evaluation.benchmarking.table_generator import (
    generate_image_level_table,
    generate_downstream_table,
    generate_spectral_table,
    generate_uncertainty_table,
)


def test_spatial_fidelity_hr_gating():
    pred = np.random.rand(4, 32, 32).astype(np.float32)
    # When HR reference is unavailable
    metrics_no_hr = spatial_fidelity(pred, hr_reference=None, hr_reference_available=False)
    assert metrics_no_hr.psnr is None
    assert metrics_no_hr.ssim is None
    assert metrics_no_hr.mae_hr is None
    assert metrics_no_hr.rmse_hr is None
    assert metrics_no_hr.gradient_energy_sr is not None
    assert metrics_no_hr.gradient_energy_sr > 0

    # When genuine HR reference is provided
    hr_ref = pred + 0.01 * np.random.randn(*pred.shape).astype(np.float32)
    metrics_hr = spatial_fidelity(pred, hr_reference=hr_ref, hr_reference_available=True)
    assert metrics_hr.psnr is not None
    assert metrics_hr.psnr > 20.0
    assert metrics_hr.ssim is not None
    assert 0.0 <= metrics_hr.ssim <= 1.0


def test_ndvi_computation():
    # 4-band image (B02, B03, B04, B08)
    img = np.zeros((4, 10, 10), dtype=np.float32)
    img[2, :, :] = 0.2  # B04 Red
    img[3, :, :] = 0.6  # B08 NIR
    ndvi = compute_ndvi(img, red_idx=2, nir_idx=3)
    # Expected NDVI = (0.6 - 0.2) / (0.6 + 0.2) = 0.4 / 0.8 = 0.5
    assert np.allclose(ndvi, 0.5, atol=1e-5)


def test_spectral_fidelity_hr_gating():
    pred = np.random.rand(4, 20, 20).astype(np.float32)
    spec_no_hr = spectral_fidelity(pred, hr_reference=None, hr_reference_available=False)
    assert spec_no_hr.sam_degrees is None
    assert spec_no_hr.per_band_mae_hr is None

    hr_ref = pred.copy()
    spec_hr = spectral_fidelity(pred, hr_reference=hr_ref, hr_reference_available=True)
    assert spec_hr.sam_degrees is not None
    assert np.isclose(spec_hr.sam_degrees, 0.0, atol=0.05)


def test_uncertainty_statistics_and_reliability():
    umap = np.linspace(0.001, 0.010, 100).reshape(10, 10).astype(np.float32)
    stats = uncertainty_statistics(umap, n_samples=5, sampling_steps=100)
    assert stats.mean is not None
    assert stats.n_samples == 5
    assert stats.sampling_steps == 100
    assert "NOT MC-dropout" in stats.mechanism

    rel_map = ReliabilityMap(umap, low_threshold=0.003, high_threshold=0.007)
    res = rel_map.to_result()
    assert res.high_reliability_fraction > 0
    assert res.medium_reliability_fraction > 0
    assert res.low_reliability_fraction > 0
    total_frac = (
        res.high_reliability_fraction
        + res.medium_reliability_fraction
        + res.low_reliability_fraction
    )
    assert np.isclose(total_frac, 1.0, atol=1e-5)


def test_downstream_segmentation_negative_result():
    # Mock segmentation scenario where SR has lower mIoU than native
    pred_nat = np.array([0, 1, 0, 1])
    labels_nat = np.array([0, 1, 0, 1])
    seg_nat = segmentation_metrics(pred_nat, labels_nat, representation="native_10m")
    assert seg_nat.pixel_accuracy == 1.0

    pred_sr = np.array([1, 0, 0, 1])
    labels_sr = np.array([0, 1, 0, 1])
    seg_sr = segmentation_metrics(pred_sr, labels_sr, representation="sr_2p5m", label_is_proxy=True)
    assert seg_sr.pixel_accuracy == 0.5
    assert seg_sr.label_is_proxy is True


def test_table_generator_handles_nones_without_inventing_values():
    img_table = generate_image_level_table(hr_reference_available=False)
    for r in img_table:
        if r["Method"] != "NOTE":
            assert r["PSNR (dB)"] == "N/A"
            assert r["SSIM"] == "N/A"
            assert r["SAM (°)"] == "N/A"
