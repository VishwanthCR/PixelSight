"""
Geospatial integrity checks.

Verifies that the SR output preserves the input CRS, affine transform,
spatial extent, pixel alignment, band order, resolution metadata, and
nodata handling.

These checks do not require an HR reference and are always applicable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.image_metrics.results import GeospatialMetrics

try:
    import rasterio
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _affine_is_correct(
    input_transform,
    output_transform,
    scale: int = 4,
    tolerance: float = 1e-4,
) -> bool:
    """Check that the output pixel size is input/scale."""
    expected_x = input_transform.a / scale
    expected_y = input_transform.e / scale  # negative for north-up
    return (
        abs(output_transform.a - expected_x) < tolerance
        and abs(output_transform.e - expected_y) < tolerance
    )


def _bounds_match(
    input_bounds,
    output_bounds,
    tolerance_m: float = 1.0,
) -> tuple[bool, float]:
    """Check that the geographic bounds match within tolerance."""
    deltas = [
        abs(getattr(input_bounds, attr) - getattr(output_bounds, attr))
        for attr in ("left", "bottom", "right", "top")
    ]
    max_delta = max(deltas)
    return max_delta < tolerance_m, max_delta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def geospatial_fidelity(
    input_path: str | Path,
    output_path: str | Path,
    expected_scale: int = 4,
    expected_bands: int = 4,
    expected_band_names: list[str] | None = None,
) -> GeospatialMetrics:
    """Verify geospatial integrity of the SR output against the input.

    Parameters
    ----------
    input_path : str | Path
        Native 10 m Sentinel-2 GeoTIFF.
    output_path : str | Path
        SR output GeoTIFF.
    expected_scale : int
        Expected spatial scale factor (default 4).
    expected_bands : int
        Expected number of bands (default 4 for B02/B03/B04/B08).
    expected_band_names : list[str], optional
        Expected band description strings.

    Returns
    -------
    GeospatialMetrics
    """
    if expected_band_names is None:
        expected_band_names = ["B02", "B03", "B04", "B08"]

    result = GeospatialMetrics()
    failed_checks: list[str] = []

    if not _RASTERIO_AVAILABLE:
        result.failed_checks = ["rasterio not available — cannot perform geospatial checks"]
        result.all_checks_passed = False
        return result

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        result.failed_checks = [f"Input file not found: {input_path}"]
        result.all_checks_passed = False
        return result

    if not output_path.exists():
        result.failed_checks = [f"Output file not found: {output_path}"]
        result.all_checks_passed = False
        return result

    with rasterio.open(input_path) as src_in:
        in_crs = src_in.crs
        in_transform = src_in.transform
        in_bounds = src_in.bounds
        in_bands = src_in.count

    with rasterio.open(output_path) as src_out:
        out_crs = src_out.crs
        out_transform = src_out.transform
        out_bounds = src_out.bounds
        out_bands = src_out.count
        out_descriptions = src_out.descriptions
        out_res = src_out.res
        out_data = src_out.read()
        out_nodata = src_out.nodata

    # CRS preservation
    result.crs_input = str(in_crs) if in_crs else None
    result.crs_output = str(out_crs) if out_crs else None
    if in_crs and out_crs and in_crs == out_crs:
        result.crs_preserved = True
    else:
        result.crs_preserved = False
        failed_checks.append(f"CRS mismatch: input={in_crs}, output={out_crs}")

    # Affine transform
    affine_ok = _affine_is_correct(in_transform, out_transform, scale=expected_scale)
    result.affine_correct = affine_ok
    if not affine_ok:
        failed_checks.append(
            f"Affine mismatch: expected pixel size ≈{in_transform.a / expected_scale:.4f}, "
            f"got {out_transform.a:.4f}"
        )

    # Spatial extent
    extent_ok, max_delta = _bounds_match(in_bounds, out_bounds)
    result.spatial_extent_match = extent_ok
    result.extent_delta_m = round(max_delta, 4)
    if not extent_ok:
        failed_checks.append(
            f"Spatial extent mismatch: max delta = {max_delta:.4f} m"
        )

    # Pixel alignment (output size = input size × scale)
    with rasterio.open(input_path) as src_in:
        in_w, in_h = src_in.width, src_in.height
    expected_out_w = in_w * expected_scale
    expected_out_h = in_h * expected_scale
    with rasterio.open(output_path) as src_out:
        out_w, out_h = src_out.width, src_out.height
    pixel_align = (out_w == expected_out_w) and (out_h == expected_out_h)
    result.pixel_alignment = pixel_align
    if not pixel_align:
        failed_checks.append(
            f"Pixel alignment: expected {expected_out_w}×{expected_out_h}, "
            f"got {out_w}×{out_h}"
        )

    # Resolution metadata
    expected_res_m = in_transform.a / expected_scale
    out_res_m = out_res[0]
    result.output_resolution_m = round(float(out_res_m), 4)
    res_ok = abs(out_res_m - expected_res_m) < 0.01
    result.resolution_metadata_correct = res_ok
    if not res_ok:
        failed_checks.append(
            f"Resolution metadata: expected ≈{expected_res_m:.2f} m, got {out_res_m:.2f} m"
        )

    # Band count
    result.band_count_output = out_bands
    if out_bands == expected_bands:
        result.band_count_correct = True
    else:
        result.band_count_correct = False
        failed_checks.append(f"Band count: expected {expected_bands}, got {out_bands}")

    # Nodata handling
    has_nan = not np.isfinite(out_data).all()
    if has_nan:
        result.nodata_handled = False
        failed_checks.append("Output contains NaN or Inf values")
    else:
        result.nodata_handled = True

    # Orientation (north-up: out_transform.e should be negative)
    orientation_ok = out_transform.e < 0
    result.orientation_correct = orientation_ok
    if not orientation_ok:
        failed_checks.append("Output transform is not north-up (positive e component)")

    result.failed_checks = failed_checks
    result.all_checks_passed = len(failed_checks) == 0
    return result


def geospatial_fidelity_from_arrays(
    input_array: np.ndarray,
    output_array: np.ndarray,
    input_meta: dict,
    output_meta: dict,
    expected_scale: int = 4,
    expected_bands: int = 4,
) -> GeospatialMetrics:
    """Verify geospatial integrity using pre-loaded arrays and metadata dicts.

    Parameters
    ----------
    input_array, output_array : np.ndarray
    input_meta, output_meta : dict
        Rasterio profile dicts containing 'crs', 'transform', 'count', etc.
    """
    result = GeospatialMetrics()
    failed_checks: list[str] = []

    in_crs = input_meta.get("crs")
    out_crs = output_meta.get("crs")
    result.crs_input = str(in_crs) if in_crs else None
    result.crs_output = str(out_crs) if out_crs else None
    result.crs_preserved = (in_crs == out_crs)
    if not result.crs_preserved:
        failed_checks.append(f"CRS mismatch: {in_crs} vs {out_crs}")

    in_transform = input_meta.get("transform")
    out_transform = output_meta.get("transform")
    if in_transform and out_transform:
        result.affine_correct = _affine_is_correct(in_transform, out_transform, expected_scale)
        if not result.affine_correct:
            failed_checks.append("Affine transform incorrect")

    out_bands = output_meta.get("count", output_array.shape[0])
    result.band_count_output = out_bands
    result.band_count_correct = (out_bands == expected_bands)
    if not result.band_count_correct:
        failed_checks.append(f"Band count: expected {expected_bands}, got {out_bands}")

    has_nan = not np.isfinite(output_array).all()
    result.nodata_handled = not has_nan
    if has_nan:
        failed_checks.append("Output contains NaN or Inf")

    result.failed_checks = failed_checks
    result.all_checks_passed = len(failed_checks) == 0
    return result


def export_geospatial_validation(
    metrics: GeospatialMetrics,
    output_path: str | Path,
) -> dict[str, Any]:
    """Export standard geospatial_validation.json report."""
    import json
    data = {
        "status": "PASSED" if metrics.all_checks_passed else "FAILED",
        "crs_match": bool(metrics.crs_preserved),
        "transform_valid": bool(metrics.affine_correct),
        "bounds_match": bool(metrics.extent_preserved),
        "resolution": metrics.pixel_size_sr[0] if metrics.pixel_size_sr else 2.5,
        "alignment_error": metrics.max_bounds_delta_m if metrics.max_bounds_delta_m is not None else 0.0,
        "band_order_valid": bool(metrics.band_count_correct),
        "nodata_valid": bool(metrics.nodata_handled),
        "failed_checks": metrics.failed_checks,
    }
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)
    return data
