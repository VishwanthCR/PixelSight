"""
PixelSight Disaster Management Service
======================================
Temporal satellite analysis pipeline for pre- and post-event satellite imagery.
Performs geospatial alignment, multi-spectral super-resolution via Core Engine,
transparent baseline change detection, and uncertainty-aware reliability categorization.

Scientific rules:
- Strictly requires two temporal acquisitions: pre_event and post_event.
- Terminology: "Changed region", "Potential affected region", "Spectral change region".
- Explicitly reports "Research analysis — no independent ground truth".
- Categorizes change by SR uncertainty into lower, moderate, and high uncertainty.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

from backend.app.services.visualization import save_rgb_preview


def align_temporal_pair(
    pre_path: Path,
    post_path: Path,
    aligned_pre_path: Path,
    aligned_post_path: Path,
) -> dict[str, Any]:
    """
    Validate and align pre-event and post-event rasters to identical grid, CRS, and extent.
    Records all applied transformations.
    """
    transformations = []
    aligned_pre_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_post_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(pre_path) as pre_src:
        pre_profile = pre_src.profile.copy()
        pre_crs = pre_src.crs
        pre_transform = pre_src.transform
        pre_h, pre_w = pre_src.height, pre_src.width
        pre_data = pre_src.read()

    # Save pre-event to aligned_pre_path
    with rasterio.open(aligned_pre_path, "w", **pre_profile) as dst:
        dst.write(pre_data)

    with rasterio.open(post_path) as post_src:
        post_crs = post_src.crs
        post_h, post_w = post_src.height, post_src.width

        # Check if reproject / resample is needed
        needs_alignment = (
            post_crs != pre_crs
            or post_h != pre_h
            or post_w != pre_w
            or post_src.transform != pre_transform
        )

        if not needs_alignment:
            transformations.append("Inputs already have matching CRS, grid, and extent.")
            with rasterio.open(aligned_post_path, "w", **pre_profile) as dst:
                dst.write(post_src.read())
        else:
            transformations.append(
                f"Resampled and matched post-event raster ({post_w}x{post_h}, CRS: {post_crs}) "
                f"to pre-event grid ({pre_w}x{pre_h}, CRS: {pre_crs})."
            )
            out_post = np.zeros((pre_profile["count"], pre_h, pre_w), dtype=pre_profile["dtype"])
            for b in range(1, pre_profile["count"] + 1):
                reproject(
                    source=rasterio.band(post_src, min(b, post_src.count)),
                    destination=out_post[b - 1],
                    src_transform=post_src.transform,
                    src_crs=post_src.crs,
                    dst_transform=pre_transform,
                    dst_crs=pre_crs,
                    resampling=Resampling.bilinear,
                )
            with rasterio.open(aligned_post_path, "w", **pre_profile) as dst:
                dst.write(out_post)

    return {
        "aligned": True,
        "grid_dimensions": [pre_w, pre_h],
        "crs": str(pre_crs),
        "transformations": transformations,
    }


def _colormap_change(magnitude: np.ndarray) -> np.ndarray:
    """Colormap for change magnitude in [0, 1]. Heatmap from dark purple/blue to bright orange/yellow."""
    norm = np.clip(magnitude / 0.5, 0.0, 1.0)
    # Dark navy -> Magenta/Purple -> Amber/Yellow
    r = np.where(norm < 0.5, 30 + norm * 340, 200 + (norm - 0.5) * 110)
    g = np.where(norm < 0.5, 20 + norm * 80, 60 + (norm - 0.5) * 390)
    b = np.where(norm < 0.5, 80 + norm * 200, 180 - (norm - 0.5) * 360)
    return np.stack([np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)], axis=-1).astype(np.uint8)


def _colormap_reliability(affected_mask: np.ndarray, unc_category: np.ndarray) -> np.ndarray:
    """
    Visualizes change regions color-coded by uncertainty reliability:
    - 0: Unchanged (dark navy background)
    - 1: Lower-uncertainty change (cyan/teal - high confidence)
    - 2: Moderate-uncertainty change (amber/yellow)
    - 3: High-uncertainty change (coral/magenta - requires caution)
    """
    h, w = affected_mask.shape
    rgb = np.full((h, w, 3), fill_value=25, dtype=np.uint8)
    # Slate base
    rgb[:, :, 0] = 30
    rgb[:, :, 1] = 35
    rgb[:, :, 2] = 45

    lower = (affected_mask == 1) & (unc_category == 1)
    mod = (affected_mask == 1) & (unc_category == 2)
    high = (affected_mask == 1) & (unc_category == 3)

    rgb[lower] = [40, 210, 190]   # Cyan / Teal
    rgb[mod] = [240, 190, 40]     # Amber
    rgb[high] = [235, 75, 75]     # Coral / Red

    return rgb


def run_disaster_analysis(
    sr_pre_path: Path,
    sr_post_path: Path,
    pre_unc_tif: Path,
    post_unc_tif: Path,
    job_dir: Path,
    change_threshold: float = 0.15,
) -> dict[str, Any]:
    """
    Execute transparent baseline change detection and reliability-aware interpretation.
    """
    disaster_dir = job_dir / "application" / "disaster"
    previews_dir = disaster_dir / "previews"
    disaster_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read pre and post SR imagery
    with rasterio.open(sr_pre_path) as src:
        sr_pre = src.read().astype(np.float32)
        profile = src.profile.copy()
    with rasterio.open(sr_post_path) as src:
        sr_post = src.read().astype(np.float32)

    # 2. Read SR uncertainties
    with rasterio.open(pre_unc_tif) as src:
        pre_unc = src.read(1).astype(np.float32)
    with rasterio.open(post_unc_tif) as src:
        post_unc = src.read(1).astype(np.float32)

    combined_unc = np.maximum(pre_unc, post_unc)
    # Normalize uncertainty to 0..1 range if necessary
    unc_max = float(np.nanmax(combined_unc))
    norm_unc = combined_unc / unc_max if unc_max > 1e-6 else combined_unc

    # 3. Transparent baseline change detection: Euclidean spectral change magnitude
    bands_to_use = min(sr_pre.shape[0], sr_post.shape[0])
    diff_squared = np.sum((sr_post[:bands_to_use] - sr_pre[:bands_to_use]) ** 2, axis=0)
    change_magnitude = np.sqrt(diff_squared) / np.sqrt(bands_to_use)

    # Mask potential affected areas
    affected_mask = (change_magnitude >= change_threshold).astype(np.uint8)

    # 4. Uncertainty categorization for changed areas:
    # 1: lower-uncertainty (< 0.35)
    # 2: moderate-uncertainty (0.35 - 0.65)
    # 3: high-uncertainty (> 0.65)
    unc_category = np.zeros_like(affected_mask, dtype=np.uint8)
    unc_category[norm_unc < 0.35] = 1
    unc_category[(norm_unc >= 0.35) & (norm_unc <= 0.65)] = 2
    unc_category[norm_unc > 0.65] = 3

    # 5. Save GeoTIFFs
    change_map_tif = disaster_dir / "change_map.tif"
    affected_area_tif = disaster_dir / "affected_area.tif"
    uncertainty_tif = disaster_dir / "uncertainty.tif"

    profile.update(count=1, dtype="float32", nodata=-9999.0, compress="deflate")
    with rasterio.open(change_map_tif, "w", **profile) as dst:
        dst.write(change_magnitude.astype(np.float32), 1)
        dst.set_band_description(1, "Spectral change magnitude [0-1]")

    profile.update(dtype="uint8", nodata=255)
    with rasterio.open(affected_area_tif, "w", **profile) as dst:
        dst.write(affected_mask, 1)
        dst.set_band_description(1, "Potential affected region mask (1=change, 0=unchanged)")

    profile.update(dtype="float32", nodata=-9999.0)
    with rasterio.open(uncertainty_tif, "w", **profile) as dst:
        dst.write(combined_unc.astype(np.float32), 1)
        dst.set_band_description(1, "Combined SR uncertainty (max of pre/post)")

    # 6. Save Previews
    change_preview_png = previews_dir / "change_preview.png"
    affected_preview_png = previews_dir / "affected_area_preview.png"
    pre_preview_png = previews_dir / "pre_sr_preview.png"
    post_preview_png = previews_dir / "post_sr_preview.png"

    Image.fromarray(_colormap_change(change_magnitude)).save(change_preview_png, format="PNG")
    Image.fromarray(_colormap_reliability(affected_mask, unc_category)).save(affected_preview_png, format="PNG")
    save_rgb_preview(sr_pre_path, pre_preview_png)
    save_rgb_preview(sr_post_path, post_preview_png)

    # 7. Compute Statistics
    total_pixels = max(int(affected_mask.size), 1)
    changed_pixels = int(affected_mask.sum())
    change_fraction = float(changed_pixels / total_pixels)

    lower_pixels = int(((affected_mask == 1) & (unc_category == 1)).sum())
    mod_pixels = int(((affected_mask == 1) & (unc_category == 2)).sum())
    high_pixels = int(((affected_mask == 1) & (unc_category == 3)).sum())

    statistics = {
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "change_percentage": float(change_fraction * 100.0),
        "mean_change_magnitude": float(np.mean(change_magnitude)),
        "max_change_magnitude": float(np.max(change_magnitude)),
        "change_threshold_applied": change_threshold,
        "reliability_breakdown": {
            "lower_uncertainty_pixels": lower_pixels,
            "lower_uncertainty_fraction_of_change": float(lower_pixels / max(changed_pixels, 1)),
            "moderate_uncertainty_pixels": mod_pixels,
            "moderate_uncertainty_fraction_of_change": float(mod_pixels / max(changed_pixels, 1)),
            "high_uncertainty_pixels": high_pixels,
            "high_uncertainty_fraction_of_change": float(high_pixels / max(changed_pixels, 1)),
        },
    }

    result = {
        "application": "disaster",
        "scientific_status": "Research analysis — no independent ground truth",
        "methodology": "Spectral difference vector magnitude with dual-acquisition SR uncertainty propagation",
        "statistics": statistics,
        "interpretations": [
            f"Detected {changed_pixels} spectral change pixels ({change_fraction*100:.2f}% of total area) exceeding magnitude threshold {change_threshold}.",
            f"Of the detected change area, {statistics['reliability_breakdown']['lower_uncertainty_fraction_of_change']*100:.1f}% falls within lower-uncertainty boundaries.",
            f"{statistics['reliability_breakdown']['high_uncertainty_fraction_of_change']*100:.1f}% of candidate change areas coincide with high super-resolution uncertainty and require cautious manual or field verification.",
        ],
        "limitations": [
            "Detected changes represent multi-spectral radiometric variations between pre- and post-acquisitions; they may include seasonal phenology, agricultural harvest, soil moisture, and atmospheric conditions, not exclusively structural disaster destruction.",
            "PixelSight disaster management is an experimental research framework and is NOT certified as an operational emergency response decision tool without field truth.",
        ],
        "outputs": {
            "change_map_geotiff": "application/disaster/change_map.tif",
            "affected_area_geotiff": "application/disaster/affected_area.tif",
            "uncertainty_geotiff": "application/disaster/uncertainty.tif",
            "change_preview": "application/disaster/previews/change_preview.png",
            "affected_area_preview": "application/disaster/previews/affected_area_preview.png",
            "pre_sr_preview": "application/disaster/previews/pre_sr_preview.png",
            "post_sr_preview": "application/disaster/previews/post_sr_preview.png",
            "metrics": "application/disaster/metrics.json",
        },
    }

    metrics_path = disaster_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
