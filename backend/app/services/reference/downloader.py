"""
Reference Downloader & Local Cache Manager
===========================================
Manages local retrieval, hashing, integrity validation, and persistent disk caching
for external high-resolution satellite reference files.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
from typing import Any
import urllib.request

try:
    import rasterio
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False


class ReferenceDownloader:
    """Retrieves and caches external HR references with cryptographic deduplication."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        root_dir = Path(__file__).resolve().parents[4]
        self.cache_dir = Path(cache_dir) if cache_dir else root_dir / "results" / "cache" / "reference"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_cache_key(
        self,
        source: str,
        reference_id: str,
        bbox: list[float] | tuple[float, float, float, float] | None = None,
        resolution_m: float = 2.5,
    ) -> str:
        """Create deterministic SHA-256 hash identifying this specific spatial slice."""
        key_raw = f"{source}::{reference_id}::{resolution_m}"
        if bbox:
            key_raw += f"::{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}"
        return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()[:16]

    def get_cached_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.tif"

    def is_cached(self, cache_key: str) -> bool:
        path = self.get_cached_path(cache_key)
        if not path.exists() or path.stat().st_size == 0:
            return False
        # Optional rasterio validation
        if _RASTERIO_AVAILABLE:
            try:
                with rasterio.open(path) as src:
                    return src.width > 0 and src.height > 0
            except Exception:
                return False
        return True

    def retrieve_reference(
        self,
        uri_or_path: str | Path,
        source: str,
        reference_id: str,
        bbox: list[float] | tuple[float, float, float, float] | None = None,
        resolution_m: float = 2.5,
    ) -> Path | None:
        """Retrieve reference raster from cache, local storage, or remote endpoint."""
        cache_key = self.generate_cache_key(source, reference_id, bbox, resolution_m)
        cached_file = self.get_cached_path(cache_key)

        if self.is_cached(cache_key):
            return cached_file

        uri_str = str(uri_or_path).strip()
        root_dir = Path(__file__).resolve().parents[4]

        # 1. Local filesystem path check
        local_cand = Path(uri_str)
        if not local_cand.is_absolute():
            local_cand = root_dir / local_cand

        if local_cand.exists() and local_cand.is_file():
            # Validate and copy to cache
            try:
                shutil.copyfile(local_cand, cached_file)
                return cached_file
            except Exception:
                return local_cand

        # 2. Remote URL retrieval (HTTP / HTTPS)
        if uri_str.startswith("http://") or uri_str.startswith("https://"):
            try:
                temp_download = self.cache_dir / f"{cache_key}.tmp"
                req = urllib.request.Request(
                    uri_str,
                    headers={"User-Agent": "PixelSight-Scientific-Evaluation/1.0"}
                )
                with urllib.request.urlopen(req, timeout=15) as response, open(temp_download, "wb") as out_f:
                    shutil.copyfileobj(response, out_f)

                # Validate downloaded raster
                if _RASTERIO_AVAILABLE:
                    with rasterio.open(temp_download) as test_src:
                        if test_src.width <= 0 or test_src.height <= 0:
                            raise ValueError("Corrupt GeoTIFF downloaded")

                temp_download.replace(cached_file)
                return cached_file
            except Exception as e:
                # Log without blocking pipeline
                return None

        return None
