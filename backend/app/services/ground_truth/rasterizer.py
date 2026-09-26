"""
Ground Truth Annotation Rasterizer
===================================
Converts validated vector GeoJSON annotations into calibrated land-cover
ground-truth GeoTIFF rasters aligned to the target common evaluation grid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from PIL import Image
from shapely.geometry import shape

from backend.app.services.ground_truth.schema import (
    CLASS_BY_ID,
    GT_CLASSES,
    IGNORE_CLASS,
)


class GroundTruthRasterizer:
    """Burns GeoJSON vector annotations to single-band uint8 GeoTIFF on evaluation grid."""

    def rasterize_to_grid(
        self,
        geojson_data: Dict[str, Any],
        output_tif_path: Path,
        template_raster_path: Optional[Path] = None,
        aoi_bbox: Optional[List[float]] = None,
        target_resolution_m: float = 2.5,
        target_dims: Tuple[int, int] = (512, 512),
        crs: str = "EPSG:4326",
    ) -> Dict[str, Any]:
        """
        Burns annotations to GeoTIFF.
        If template_raster_path is provided, matches its transform, CRS, and width/height.
        Otherwise constructs transform from aoi_bbox and target_dims.
        """
        output_tif_path = Path(output_tif_path)
        output_tif_path.parent.mkdir(parents=True, exist_ok=True)

        if template_raster_path and Path(template_raster_path).exists():
            with rasterio.open(template_raster_path) as src:
                width = src.width
                height = src.height
                transform = src.transform
                crs = str(src.crs or "EPSG:4326")
                bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
                res_x, res_y = src.res
        elif aoi_bbox and len(aoi_bbox) == 4:
            width, height = target_dims
            bounds = aoi_bbox
            transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
            res_x = (bounds[2] - bounds[0]) / width
            res_y = (bounds[3] - bounds[1]) / height
        else:
            raise ValueError("Either valid template_raster_path or aoi_bbox must be provided.")

        features = geojson_data.get("features", [])
        shapes_and_values = []

        # Sort features so that Ignore (255) burns first or specific classes burn predictably
        # Normal land-cover classes override general base classes
        for feat in features:
            geom = feat.get("geometry")
            if not geom:
                continue
            props = feat.get("properties", {})
            class_id = int(props.get("class_id", 255))
            try:
                sh_geom = shape(geom)
                if sh_geom.is_valid and not sh_geom.is_empty:
                    shapes_and_values.append((sh_geom, class_id))
            except Exception:
                continue

        # Burn shapes onto raster. Background default is 255 (Ignore/Unannotated)
        if shapes_and_values:
            burned_raster = rasterize(
                shapes=shapes_and_values,
                out_shape=(height, width),
                fill=255,
                transform=transform,
                all_touched=False,
                dtype=np.uint8,
            )
        else:
            burned_raster = np.full((height, width), 255, dtype=np.uint8)

        # Write GeoTIFF
        meta = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "uint8",
            "crs": crs,
            "transform": transform,
            "nodata": 255,
        }
        with rasterio.open(output_tif_path, "w", **meta) as dst:
            dst.write(burned_raster, 1)
            dst.set_band_description(1, "Ground_Truth_Land_Cover")

        # Generate companion preview PNG
        preview_png_path = output_tif_path.with_suffix(".png")
        self._generate_colored_preview(burned_raster, preview_png_path)

        # Compute area statistics
        stats = self._compute_statistics(burned_raster, res_x, res_y, crs)
        stats["raster_path"] = str(output_tif_path)
        stats["preview_path"] = str(preview_png_path)
        stats["dimensions"] = [width, height]
        stats["crs"] = crs
        stats["bounds"] = bounds

        return stats

    def _generate_colored_preview(self, label_array: np.ndarray, output_path: Path):
        """Maps class IDs to canonical RGB colors and saves 8-bit PNG."""
        h, w = label_array.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Default background is dark slate #0f172a
        rgb[:] = [15, 23, 42]

        for cid, meta in CLASS_BY_ID.items():
            hex_color = meta["color"].lstrip("#")
            c_rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            mask = label_array == cid
            if mask.any():
                rgb[mask] = c_rgb

        img = Image.fromarray(rgb, mode="RGB")
        img.save(output_path, format="PNG")

    def _compute_statistics(
        self,
        raster: np.ndarray,
        res_x: float,
        res_y: float,
        crs: str,
    ) -> Dict[str, Any]:
        total_pixels = raster.size
        ignored_pixels = int((raster == 255).sum())
        labeled_pixels = total_pixels - ignored_pixels

        coverage_pct = round((labeled_pixels / max(1, total_pixels)) * 100.0, 2)
        ignored_pct = round((ignored_pixels / max(1, total_pixels)) * 100.0, 2)

        # Area conversion
        is_degrees = "4326" in crs or abs(res_x) < 0.1
        if is_degrees:
            meter_x = abs(res_x) * 111320.0
            meter_y = abs(res_y) * 111320.0
        else:
            meter_x = abs(res_x)
            meter_y = abs(res_y)

        pixel_area_m2 = max(meter_x * meter_y, 1.0)
        ha_per_px = pixel_area_m2 / 10000.0
        km2_per_px = pixel_area_m2 / 1000000.0

        class_stats = {}
        for c in GT_CLASSES:
            cid = c["id"]
            cname = c["name"]
            cnt = int((raster == cid).sum())
            pct_total = round((cnt / max(1, total_pixels)) * 100.0, 2)
            pct_labeled = round((cnt / max(1, labeled_pixels)) * 100.0, 2) if labeled_pixels > 0 else 0.0

            class_stats[cname] = {
                "id": cid,
                "color": c["color"],
                "pixel_count": cnt,
                "percent_total": pct_total,
                "percent_labeled": pct_labeled,
                "area_ha": round(cnt * ha_per_px, 3),
                "area_km2": round(cnt * km2_per_px, 4),
            }

        return {
            "total_pixels": total_pixels,
            "labeled_pixels": labeled_pixels,
            "ignored_pixels": ignored_pixels,
            "coverage_percentage": coverage_pct,
            "ignored_percentage": ignored_pct,
            "class_area_statistics": class_stats,
        }


ground_truth_rasterizer = GroundTruthRasterizer()
