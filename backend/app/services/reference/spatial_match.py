"""
Spatial Matching Component
==========================
Performs exact geospatial polygon intersection and coverage computation
between AOI geometries / raster bounds and reference candidate footprints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shapely.geometry import shape, box, Polygon, mapping

try:
    from rasterio.warp import transform_geom, transform_bounds
    _WARP_AVAILABLE = True
except ImportError:
    _WARP_AVAILABLE = False


@dataclass
class SpatialMatchResult:
    """Result of spatial overlap computation."""
    intersects: bool
    overlap_pct: float            # Percentage of the AOI covered by the reference
    candidate_coverage_pct: float # Percentage of the reference covered by the AOI
    intersection_area_deg2: float
    aoi_area_deg2: float
    candidate_area_deg2: float
    intersection_geometry: dict[str, Any] | None
    notes: list[str]


class SpatialMatcher:
    """Computes exact geospatial polygon intersections in EPSG:4326 coordinates."""

    def __init__(self, min_overlap_pct: float = 50.0) -> None:
        self.min_overlap_pct = min_overlap_pct

    def normalize_aoi_polygon(
        self,
        aoi: list[float] | tuple[float, float, float, float] | dict[str, Any],
        src_crs: str = "EPSG:4326",
    ) -> Polygon:
        """Converts bbox or GeoJSON geometry into a Shapely Polygon in EPSG:4326."""
        if isinstance(aoi, (list, tuple)):
            if len(aoi) != 4:
                raise ValueError("BBox AOI must contain exactly 4 coordinates: [west, south, east, north]")
            w, s, e, n = aoi
            if src_crs != "EPSG:4326" and _WARP_AVAILABLE:
                w, s, e, n = transform_bounds(src_crs, "EPSG:4326", w, s, e, n)
            return box(w, s, e, n)

        if isinstance(aoi, dict):
            geom_dict = aoi.get("geometry", aoi)
            if src_crs != "EPSG:4326" and _WARP_AVAILABLE:
                geom_dict = transform_geom(src_crs, "EPSG:4326", geom_dict)
            poly = shape(geom_dict)
            if not isinstance(poly, Polygon):
                poly = poly.convex_hull
            return poly

        raise TypeError(f"Unsupported AOI type: {type(aoi)}")

    def compute_match(
        self,
        aoi: list[float] | tuple[float, float, float, float] | dict[str, Any],
        candidate_bbox: list[float] | tuple[float, float, float, float],
        candidate_geometry: dict[str, Any] | None = None,
        aoi_crs: str = "EPSG:4326",
        candidate_crs: str = "EPSG:4326",
    ) -> SpatialMatchResult:
        """Calculate exact polygon intersection and overlap percentage."""
        notes: list[str] = []
        try:
            aoi_poly = self.normalize_aoi_polygon(aoi, aoi_crs)
        except Exception as e:
            return SpatialMatchResult(
                intersects=False,
                overlap_pct=0.0,
                candidate_coverage_pct=0.0,
                intersection_area_deg2=0.0,
                aoi_area_deg2=0.0,
                candidate_area_deg2=0.0,
                intersection_geometry=None,
                notes=[f"Failed to normalize AOI geometry: {e}"],
            )

        if candidate_geometry and candidate_geometry.get("coordinates"):
            try:
                cand_geom_dict = candidate_geometry
                if candidate_crs != "EPSG:4326" and _WARP_AVAILABLE:
                    cand_geom_dict = transform_geom(candidate_crs, "EPSG:4326", cand_geom_dict)
                cand_poly = shape(cand_geom_dict)
            except Exception:
                w, s, e, n = candidate_bbox
                cand_poly = box(w, s, e, n)
        else:
            w, s, e, n = candidate_bbox
            if candidate_crs != "EPSG:4326" and _WARP_AVAILABLE:
                w, s, e, n = transform_bounds(candidate_crs, "EPSG:4326", w, s, e, n)
            cand_poly = box(w, s, e, n)

        aoi_area = float(aoi_poly.area)
        cand_area = float(cand_poly.area)

        if aoi_area <= 0:
            return SpatialMatchResult(
                intersects=False,
                overlap_pct=0.0,
                candidate_coverage_pct=0.0,
                intersection_area_deg2=0.0,
                aoi_area_deg2=0.0,
                candidate_area_deg2=cand_area,
                intersection_geometry=None,
                notes=["AOI polygon has zero area"],
            )

        if not aoi_poly.intersects(cand_poly):
            return SpatialMatchResult(
                intersects=False,
                overlap_pct=0.0,
                candidate_coverage_pct=0.0,
                intersection_area_deg2=0.0,
                aoi_area_deg2=aoi_area,
                candidate_area_deg2=cand_area,
                intersection_geometry=None,
                notes=["No spatial intersection between AOI and candidate reference footprint"],
            )

        intersection = aoi_poly.intersection(cand_poly)
        inter_area = float(intersection.area)
        overlap_pct = float(min(100.0, max(0.0, (inter_area / aoi_area) * 100.0)))
        cand_coverage_pct = float(min(100.0, max(0.0, (inter_area / cand_area) * 100.0))) if cand_area > 0 else 0.0

        is_eligible = overlap_pct >= self.min_overlap_pct
        notes.append(
            f"Overlap: {overlap_pct:.2f}% (minimum required: {self.min_overlap_pct:.1f}%). "
            f"Candidate coverage: {cand_coverage_pct:.2f}%."
        )

        return SpatialMatchResult(
            intersects=is_eligible,
            overlap_pct=overlap_pct,
            candidate_coverage_pct=cand_coverage_pct,
            intersection_area_deg2=inter_area,
            aoi_area_deg2=aoi_area,
            candidate_area_deg2=cand_area,
            intersection_geometry=mapping(intersection) if not intersection.is_empty else None,
            notes=notes,
        )
