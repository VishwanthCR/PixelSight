import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from shapely.geometry import box, shape

from backend.app.config import (
    AOI_INTERACTIVE_MAX_AREA_SQKM,
    AOI_MAX_AREA_SQKM,
    PIXELSIGHT_SUPPORTED_COUNTRY,
)


class AOIValidationError(Exception):
    """Raised when AOI coordinates, polygon, or dimensions are invalid."""
    pass


_INDIA_GEOJSON_PATH = Path(__file__).resolve().parent.parent / "data" / "india_boundary.geojson"
_INDIA_GEOMETRY = None
_INDIA_BUFFERED = None


def get_india_geometry():
    """Loads and caches the official India boundary polygon and its coastal buffer."""
    global _INDIA_GEOMETRY, _INDIA_BUFFERED
    if _INDIA_GEOMETRY is None:
        if _INDIA_GEOJSON_PATH.exists():
            data = json.loads(_INDIA_GEOJSON_PATH.read_text(encoding="utf-8"))
            geom_data = data.get("geometry", data)
            _INDIA_GEOMETRY = shape(geom_data)
            # 0.08 degree buffer (~8.8 km) accommodates coastal ports, beaches, and territorial water boundaries
            _INDIA_BUFFERED = _INDIA_GEOMETRY.buffer(0.08)
    return _INDIA_GEOMETRY, _INDIA_BUFFERED


def is_aoi_inside_india(bbox: List[float]) -> Tuple[bool, str]:
    """
    Checks if an AOI [min_lon, min_lat, max_lon, max_lat] is inside the supported India region.
    Returns (is_valid, message).
    """
    india_geom, india_buf = get_india_geometry()
    if india_geom is None:
        return True, "India boundary data unavailable; skipping check."

    min_lon, min_lat, max_lon, max_lat = bbox
    aoi_box = box(min_lon, min_lat, max_lon, max_lat)

    if not india_geom.intersects(aoi_box):
        return False, "Selected processing area is outside India. PixelSight currently supports satellite processing only within India."
    if not india_buf.contains(aoi_box):
        return False, "Processing area must be completely inside the supported India region."

    return True, "Processing area is within supported India region."


def parse_bbox_from_geometry(geometry: Dict[str, Any]) -> List[float]:
    """
    Extracts [min_lon, min_lat, max_lon, max_lat] from a GeoJSON geometry or bbox list.
    """
    if isinstance(geometry, list) and len(geometry) == 4:
        return [float(x) for x in geometry]

    if not isinstance(geometry, dict):
        raise AOIValidationError("AOI must be a GeoJSON dictionary or [min_lon, min_lat, max_lon, max_lat] list.")

    geom_type = geometry.get("type", "")
    coordinates = geometry.get("coordinates")

    if geom_type == "Polygon" and coordinates:
        coords = coordinates[0]
        lons = [pt[0] for pt in coords]
        lats = [pt[1] for pt in coords]
        return [min(lons), min(lats), max(lons), max(lats)]

    elif geom_type == "Point" and coordinates:
        lon, lat = coordinates[0], coordinates[1]
        # 1 km buffer approx (~0.01 deg)
        return [lon - 0.005, lat - 0.005, lon + 0.005, lat + 0.005]

    raise AOIValidationError(f"Unsupported geometry type: {geom_type}. Expected 'Polygon'.")


def bbox_to_geojson_polygon(bbox: List[float]) -> Dict[str, Any]:
    min_lon, min_lat, max_lon, max_lat = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]
        ],
    }


def validate_and_estimate_aoi(
    aoi: Any,
    max_area_sqkm: float = AOI_MAX_AREA_SQKM,
    interactive_max_sqkm: float = AOI_INTERACTIVE_MAX_AREA_SQKM,
    enforce_india_boundary: bool = True,
) -> Dict[str, Any]:
    """
    Validates an AOI bounding box or GeoJSON polygon and computes spatial metrics:
    area, dimensions, 10m input pixel dimensions, 4x SR output dimensions (~2.5m equivalent),
    and 128x128 tile counts. Enforces India-only geographic boundary.
    """
    bbox = parse_bbox_from_geometry(aoi)
    min_lon, min_lat, max_lon, max_lat = bbox

    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
        raise AOIValidationError(f"Longitude values must be between -180 and 180. Received [{min_lon}, {max_lon}].")
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise AOIValidationError(f"Latitude values must be between -90 and 90. Received [{min_lat}, {max_lat}].")
    if min_lon >= max_lon:
        raise AOIValidationError(f"min_lon ({min_lon}) must be strictly less than max_lon ({max_lon}).")
    if min_lat >= max_lat:
        raise AOIValidationError(f"min_lat ({min_lat}) must be strictly less than max_lat ({max_lat}).")

    # India-only boundary enforcement
    if enforce_india_boundary:
        is_valid_india, err_msg = is_aoi_inside_india(bbox)
        if not is_valid_india:
            raise AOIValidationError(err_msg)

    mid_lat = (min_lat + max_lat) / 2.0
    lat_rad = math.radians(mid_lat)

    # 1 degree of latitude ~ 111.132 km
    # 1 degree of longitude ~ 111.320 * cos(lat) km
    width_km = max(0.01, (max_lon - min_lon) * 111.320 * math.cos(lat_rad))
    height_km = max(0.01, (max_lat - min_lat) * 111.132)
    area_sqkm = round(width_km * height_km, 4)

    if area_sqkm > max_area_sqkm:
        raise AOIValidationError(
            f"Selected AOI area ({area_sqkm:.2f} km²) exceeds maximum allowed limit ({max_area_sqkm:.2f} km²). "
            "Please select a smaller region."
        )

    # Input dimensions at Sentinel-2 native 10m resolution
    input_width = max(16, int(round((width_km * 1000.0) / 10.0)))
    input_height = max(16, int(round((height_km * 1000.0) / 10.0)))

    # Output dimensions at 4x super-resolution (~2.5m equivalent)
    sr_width = input_width * 4
    sr_height = input_height * 4

    # Exact LDSR-S2 tiling: 128x128 patches with 12px overlap (stride 116)
    tile_size = 128
    overlap = 12
    tiles_x = len(get_tile_positions(input_width, tile_size, overlap))
    tiles_y = len(get_tile_positions(input_height, tile_size, overlap))
    total_tiles = tiles_x * tiles_y

    category = "interactive" if area_sqkm <= interactive_max_sqkm else "background"

    geojson_polygon = bbox_to_geojson_polygon(bbox)

    return {
        "valid": True,
        "inside_india": True,
        "supported_country": "INDIA",
        "bbox": [round(x, 6) for x in bbox],
        "geojson": geojson_polygon,
        "dimensions_km": {
            "width_km": round(width_km, 3),
            "height_km": round(height_km, 3),
        },
        "area_sqkm": area_sqkm,
        "category": category,
        "native_10m": {
            "width": input_width,
            "height": input_height,
            "total_pixels": input_width * input_height,
        },
        "super_resolution_2_5m": {
            "width": sr_width,
            "height": sr_height,
            "total_pixels": sr_width * sr_height,
            "scale_factor": 4,
            "label": "4× super-resolved representation (~2.5 m equivalent)",
        },
        "tiles": {
            "tile_size": tile_size,
            "overlap": overlap,
            "stride": tile_size - overlap,
            "tiles_x": tiles_x,
            "tiles_y": tiles_y,
            "total_tiles": total_tiles,
        },
    }


def get_tile_positions(length: int, patch_size: int = 128, overlap: int = 12) -> List[int]:
    """Computes exact 1D patch starting positions with overlap, matching infer_ldsr_s2."""
    if length <= patch_size:
        return [0]
    stride = patch_size - overlap
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def compute_bbox_for_center(lat: float, lon: float, width_km: float = 1.28, height_km: float = 1.28) -> List[float]:
    """
    Computes a bounding box [min_lon, min_lat, max_lon, max_lat] centered at (lat, lon)
    for given dimensions in kilometers. Default is 1.28 km x 1.28 km (128x128 10m pixels).
    """
    lat_rad = math.radians(lat)
    delta_lat = height_km / 111.132
    cos_lat = max(0.01, math.cos(lat_rad))
    delta_lon = width_km / (111.320 * cos_lat)

    min_lat = round(lat - delta_lat / 2.0, 6)
    max_lat = round(lat + delta_lat / 2.0, 6)
    min_lon = round(lon - delta_lon / 2.0, 6)
    max_lon = round(lon + delta_lon / 2.0, 6)

    return [min_lon, min_lat, max_lon, max_lat]
