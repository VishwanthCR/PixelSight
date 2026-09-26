"""
Tests for Genuine Stochastic Diffusion Variance & Uncertainty Subsystem
========================================================================
Validates Section 35:
- N stochastic samples
- Numerical verification: mean == mean(samples), variance == variance(samples)
- variance > 0 for non-degenerate regions
- Welford online algorithm numerical stability
- GeoTIFF artifact generation: mean.tif, variance.tif, std.tif, confidence.tif, stats.json
"""

import json
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from backend.app.services.uncertainty import (
    compute_streaming_welford_variance,
    generate_stochastic_uncertainty,
    generate_uncertainty_map,
)


def _create_synthetic_input_raster(path: Path, h: int = 64, w: int = 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 4-band synthetic raster
    data = np.zeros((4, h, w), dtype=np.float32)
    # Add gradients and features
    for c in range(4):
        data[c] = np.linspace(0.1 * (c + 1), 0.8, h * w).reshape((h, w)).astype(np.float32)
    transform = from_origin(80.25, 13.10, 0.0001, 0.0001)
    meta = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 4,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data)
    return path


def test_welford_numerical_equivalence():
    """Verify that Welford online streaming variance equals numpy sample variance exactly."""
    rng = np.random.RandomState(123)
    num_samples = 5
    shape = (4, 32, 32)
    samples = [rng.normal(loc=0.5, scale=0.1, size=shape).astype(np.float32) for _ in range(num_samples)]

    # NumPy reference
    stack = np.stack(samples, axis=0)
    expected_mean = np.mean(stack, axis=0)
    expected_var = np.var(stack, axis=0, ddof=1)
    expected_std = np.std(stack, axis=0, ddof=1)

    # Welford algorithm
    w_mean, w_var, w_std, w_conf = compute_streaming_welford_variance(samples, num_samples=num_samples)

    np.testing.assert_allclose(w_mean, expected_mean, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(w_var, np.mean(expected_var, axis=0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(w_std, np.mean(expected_std, axis=0), rtol=1e-5, atol=1e-6)

    # Verify variance > 0 everywhere
    assert np.all(w_var > 0)
    assert np.all(w_std > 0)
    # Confidence in [0, 1]
    assert np.all((w_conf >= 0.0) & (w_conf <= 1.0))


def test_generate_stochastic_uncertainty_artifacts(tmp_path):
    """Verify that generate_stochastic_uncertainty produces all required GeoTIFF and JSON artifacts."""
    input_tif = _create_synthetic_input_raster(tmp_path / "input.tif", h=32, w=32)
    out_dir = tmp_path / "uncertainty"

    stats = generate_stochastic_uncertainty(
        input_path=input_tif,
        output_dir=out_dir,
        num_samples=4,
    )

    # Artifact verification
    assert (out_dir / "mean.tif").exists()
    assert (out_dir / "variance.tif").exists()
    assert (out_dir / "std.tif").exists()
    assert (out_dir / "confidence.tif").exists()
    assert (out_dir / "stats.json").exists()
    assert (out_dir / "variance.png").exists()
    assert (out_dir / "std.png").exists()
    assert (out_dir / "confidence.png").exists()

    # Numerical verification
    assert stats["sampling_count"] == 4
    assert stats["mean_variance"] > 0
    assert stats["max_variance"] > stats["mean_variance"]
    assert stats["mean_std"] > 0
    assert 0 <= stats["high_uncertainty_percentage"] <= 100

    # Verify GeoTIFF metadata & content
    with rasterio.open(out_dir / "variance.tif") as src:
        var_data = src.read(1)
        assert src.crs == "EPSG:4326"
        assert var_data.shape == (128, 128)  # 4x resolution of 32x32 input
        assert np.nanmean(var_data) > 0
        assert np.isfinite(var_data).all()
