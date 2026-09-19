"""
Tests for PixelSight Reference Subsystem, Master Pipeline & Novel Research Modules
==================================================================================
Covers:
1. Reference discovery and controlled synthetic degradation fallback.
2. Frequency-domain MTF & Fourier Ring Correlation (FRC) resolution analysis.
3. Super-Resolution Mapping (SRM) Sub-Pixel Abundance Conservation (SPFE, ACI, HCR).
4. Uncertainty-Gated Hybrid Fusion & Decision Safety.
5. Master reporting generation (JSON, Markdown, HTML).
"""

import os
from pathlib import Path
import numpy as np
import pytest

from evaluation.reference.discovery import ReferenceDiscovery, DiscoveredReference
from evaluation.reference.synthetic import SyntheticBenchmarkGenerator, DegradationConfig
from evaluation.image_metrics.frequency_mtf import (
    FrequencyResolutionAnalyzer,
    compute_radial_power_spectrum,
    compute_fourier_ring_correlation,
)
from evaluation.downstream.srm_conservation import (
    SubPixelMappingValidator,
    SubPixelConservationResult,
)
from evaluation.uncertainty.decision_support import (
    UncertaintyGatedFusionEngine,
    DecisionSupportResult,
)
from evaluation.reporting.reporter import MasterReporter


def test_reference_discovery_and_synthetic_fallback(tmp_path):
    # Discovery with empty directory triggers synthetic benchmark fallback
    discovery = ReferenceDiscovery(reference_dirs=[tmp_path])
    res = discovery.discover(
        input_crs="EPSG:32643",
        input_bounds=(852000, 1142000, 857000, 1147000),
        input_resolution=10.0,
    )
    assert not res.found
    assert res.reference_type == "synthetic_hr"
    assert len(res.discovery_notes) > 0

    # Test synthetic generator degradation
    cfg = DegradationConfig(degradation_scale=4, blur_sigma=1.0, noise_level=0.001)
    gen = SyntheticBenchmarkGenerator(config=cfg)
    ref_arr = np.random.uniform(0.1, 0.8, (4, 64, 64)).astype(np.float32)
    ref_out, degraded_out, out_cfg = gen.generate_from_array(ref_arr, tmp_path / "synth")

    assert ref_out.shape == (4, 64, 64)
    assert degraded_out.shape == (4, 16, 16)
    assert np.all(degraded_out >= 0.0)
    assert np.all(degraded_out <= 1.0)


def test_frequency_domain_mtf_and_frc():
    # Construct synthetic test images
    img1 = np.random.uniform(0.1, 0.9, (4, 64, 64)).astype(np.float32)
    img2 = img1 + np.random.normal(0, 0.02, (4, 64, 64)).astype(np.float32)

    # 1. Radial power spectrum
    freqs, psd = compute_radial_power_spectrum(img1[0])
    assert len(freqs) == len(psd)
    assert freqs[0] == 0.0
    assert np.all(psd >= 0.0)

    # 2. Fourier Ring Correlation
    f_arr, frc_curve, cutoff = compute_fourier_ring_correlation(img1[0], img2[0])
    assert len(f_arr) == len(frc_curve)
    assert 0.0 <= cutoff <= 0.50

    # 3. FrequencyResolutionAnalyzer
    analyzer = FrequencyResolutionAnalyzer(nominal_gsd_meters=2.5)
    metrics = analyzer.analyze(sr_mean=img1, sr_realization_b=img2)
    assert metrics.effective_gsd_meters is not None
    assert metrics.effective_gsd_meters > 0.0
    assert metrics.nominal_gsd_meters == 2.5
    assert len(metrics.diagnostic_summary) > 0


def test_srm_subpixel_conservation_analysis():
    # Native 10m image (4, 16, 16)
    native = np.zeros((4, 16, 16), dtype=np.float32)
    native[2, :, :] = 0.1  # Red
    native[3, :, :] = 0.7  # NIR (high vegetation)

    # SR classes 2.5m (64, 64)
    sr_classes = np.ones((64, 64), dtype=np.int32)  # all vegetation (class 1)

    validator = SubPixelMappingValidator(scale_factor=4)
    result = validator.evaluate_conservation(native, sr_classes)

    assert isinstance(result, SubPixelConservationResult)
    assert result.scale_factor == 4
    assert result.num_subpixels_per_coarse == 16
    assert "vegetation" in result.class_spfe
    assert 0.0 <= result.subpixel_preservation_score <= 1.0


def test_uncertainty_gated_decision_support():
    sr = np.random.uniform(0.2, 0.8, (4, 32, 32)).astype(np.float32)
    nat = np.random.uniform(0.2, 0.8, (4, 8, 8)).astype(np.float32)
    umap = np.random.uniform(0.001, 0.010, (32, 32)).astype(np.float32)

    engine = UncertaintyGatedFusionEngine(uncertainty_percentile_cutoff=75.0)
    fused, res = engine.fuse(sr, nat, umap)

    assert fused.shape == sr.shape
    assert isinstance(res, DecisionSupportResult)
    assert 0.0 <= res.sr_retention_percentage <= 100.0
    assert 0.0 <= res.native_fallback_percentage <= 100.0
    assert np.isclose(res.sr_retention_percentage + res.native_fallback_percentage, 100.0)
    assert 0.0 <= res.decision_safety_score <= 1.0


def test_master_reporter_export(tmp_path):
    reporter = MasterReporter(experiment_id="test_exp", reference_type="synthetic_hr")
    reporter.add_section("image_metrics", {"psnr": 28.5, "ssim": 0.82})
    reporter.add_section("spectral_metrics", {"native_mean": 0.25, "sr_mean": 0.24})

    json_path = reporter.save_json(tmp_path / "report.json")
    md_path = reporter.save_markdown(tmp_path / "report.md")
    html_path = reporter.save_html(tmp_path / "report.html")

    assert json_path.exists()
    assert md_path.exists()
    assert html_path.exists()
    assert "test_exp" in md_path.read_text(encoding="utf-8")
    assert "PixelSight" in html_path.read_text(encoding="utf-8")
