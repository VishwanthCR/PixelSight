import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import requests

from backend.app.config import CACHE_ROOT, COPERNICUS_PROCESS_URL
from backend.app.services.aoi import parse_bbox_from_geometry, validate_and_estimate_aoi
from backend.app.services.copernicus_auth import copernicus_auth, CopernicusAuthError


class CopernicusProcessingError(Exception):
    """Raised when Copernicus Processing API acquisition fails."""
    pass


class CopernicusProcessingService:
    def __init__(self, process_url: str = COPERNICUS_PROCESS_URL, cache_dir: Path = CACHE_ROOT):
        self.process_url = process_url
        self.cache_dir = cache_dir

    def _compute_cache_key(
        self,
        scene_id: Optional[str],
        bbox: List[float],
        bands: List[str],
        width: int,
        height: int,
        date: Optional[str],
    ) -> str:
        key_str = f"s2l2a_{scene_id}_{bbox}_{bands}_{width}x{height}_{date}"
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def acquire_aoi(
        self,
        aoi: Any,
        scene_id: Optional[str] = None,
        date: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
        bands: Tuple[str, ...] = ("B02", "B03", "B04", "B08"),
        destination: Optional[Path] = None,
        use_cache: bool = True,
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Acquires B02, B03, B04, B08 10m bands for the specified AOI and Sentinel-2 scene.
        Returns the path to a standard GeoTIFF with 4 bands and valid CRS/transform,
        plus the acquisition metadata dictionary.
        """
        aoi_info = validate_and_estimate_aoi(aoi)
        bbox = aoi_info["bbox"]
        width = aoi_info["native_10m"]["width"]
        height = aoi_info["native_10m"]["height"]

        # If date is provided (e.g., '2023-06-29T10:00:31Z' or '2023-06-29'), compute timeRange
        if date:
            d_clean = date.split("T")[0]
            from_time = f"{d_clean}T00:00:00Z"
            to_time = f"{d_clean}T23:59:59Z"
        elif date_range:
            from_time = date_range[0] if "T" in date_range[0] else f"{date_range[0]}T00:00:00Z"
            to_time = date_range[1] if "T" in date_range[1] else f"{date_range[1]}T23:59:59Z"
        else:
            raise CopernicusProcessingError("Either acquisition date or date_range must be specified.")

        cache_key = self._compute_cache_key(scene_id, bbox, list(bands), width, height, date)
        cache_path = self.cache_dir / f"aoi_{cache_key}.tif"
        meta_path = self.cache_dir / f"aoi_{cache_key}_meta.json"

        if use_cache and cache_path.exists() and meta_path.exists():
            try:
                with open(meta_path, "r") as mf:
                    meta = json.load(mf)
                if destination:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(cache_path, destination)
                    return destination, meta
                return cache_path, meta
            except Exception:
                # If cache read fails, proceed to live acquisition
                pass

        token = copernicus_auth.get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "image/tiff",
            "Content-Type": "application/json",
        }

        evalscript = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02", "B03", "B04", "B08"] }],
    output: { id: "default", bands: 4, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [sample.B02, sample.B03, sample.B04, sample.B08];
}
"""

        request_payload = {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {
                        "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                    }
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": from_time,
                            "to": to_time
                        }
                    }
                }]
            },
            "output": {
                "width": width,
                "height": height,
                "responses": [{
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff"
                    }
                }]
            },
            "evalscript": evalscript
        }

        try:
            response = requests.post(
                self.process_url,
                json=request_payload,
                headers=headers,
                timeout=60,
            )
        except requests.RequestException as exc:
            raise CopernicusProcessingError(
                f"Network error querying Copernicus Processing API: {exc}"
            ) from exc

        if response.status_code == 401 or response.status_code == 403:
            raise CopernicusProcessingError(
                "Unable to access Copernicus data. Authentication or authorization failed. Check backend credentials."
            )
        elif response.status_code == 429:
            raise CopernicusProcessingError(
                "Copernicus Processing API rate limit reached. Please wait a moment and try again."
            )
        elif response.status_code != 200:
            raise CopernicusProcessingError(
                f"Copernicus Processing API failed with status {response.status_code}: {response.text[:300]}"
            )

        content = response.content
        if not content or len(content) < 100:
            raise CopernicusProcessingError("Received empty or corrupt response from Copernicus Processing API.")

        target_file = destination or cache_path
        target_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with rasterio.open(io.BytesIO(content)) as src:
                raw_data = src.read()
                profile = src.profile.copy()
                src_crs = src.crs or "EPSG:4326"
                src_transform = src.transform

                # Ensure transform matches bounds if missing or zero
                if abs(src_transform.a) == 0 or abs(src_transform.e) == 0:
                    src_transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)

                profile.pop("blockxsize", None)
                profile.pop("blockysize", None)
                profile.pop("tiled", None)

                profile.update(
                    driver="GTiff",
                    dtype="float32",
                    count=4,
                    width=width,
                    height=height,
                    crs=src_crs,
                    transform=src_transform,
                    compress="deflate",
                    tiled=False,
                )

                with rasterio.open(target_file, "w", **profile) as dst:
                    dst.write(raw_data.astype(np.float32))
                    for idx, bname in enumerate(bands, start=1):
                        dst.set_band_description(idx, bname)
                        dst.update_tags(idx, band_name=bname)

        except Exception as exc:
            raise CopernicusProcessingError(
                f"Failed to process and save Sentinel-2 GeoTIFF from Copernicus API: {exc}"
            ) from exc

        metadata = {
            "source": "Copernicus Data Space Ecosystem",
            "collection": "Sentinel-2 L2A",
            "scene_id": scene_id,
            "acquisition_date": date or f"{from_time}/{to_time}",
            "time_range": {"from": from_time, "to": to_time},
            "aoi_bbox": bbox,
            "bands": list(bands),
            "resolution_m": 10.0,
            "crs": str(profile.get("crs")),
            "raster_dimensions": [width, height],
            "sr_dimensions_estimate": [width * 4, height * 4],
            "tile_count_estimate": aoi_info["tiles"]["total_tiles"],
            "area_sqkm": aoi_info["area_sqkm"],
            "cached": True,
            "cache_key": cache_key,
        }

        # Cache copy if destination was custom
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if target_file != cache_path:
            import shutil
            shutil.copy2(target_file, cache_path)

        with open(meta_path, "w") as mf:
            json.dump(metadata, mf, indent=2)

        return target_file, metadata


copernicus_processing = CopernicusProcessingService()
