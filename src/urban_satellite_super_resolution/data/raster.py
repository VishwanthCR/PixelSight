"""Raster validation, pairing, and overlapping tile extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


@dataclass(frozen=True)
class RasterInfo:
    path: Path
    shape: tuple[int, int]
    count: int
    crs: object
    transform: object
    bounds: object
    nodata: float | None
    descriptions: tuple[str | None, ...]


def inspect_raster(path: str | Path) -> RasterInfo:
    path = Path(path)
    with rasterio.open(path) as src:
        return RasterInfo(path, (src.height, src.width), src.count, src.crs, src.transform, src.bounds, src.nodata, src.descriptions)


def assert_aligned(reference: str | Path, candidate: str | Path, *, require_shape: bool = True) -> None:
    left, right = inspect_raster(reference), inspect_raster(candidate)
    if left.crs != right.crs or left.transform != right.transform:
        raise ValueError(f"Rasters are not aligned: {left.path} and {right.path}")
    if require_shape and left.shape != right.shape:
        raise ValueError(f"Raster shapes differ: {left.shape} versus {right.shape}")
    if left.bounds != right.bounds:
        raise ValueError("Raster bounds differ")


def read_reflectance(path: str | Path, *, bands: tuple[int, ...] | None = None) -> tuple[np.ndarray, RasterInfo]:
    with rasterio.open(path) as src:
        indexes = bands or tuple(range(1, src.count + 1))
        data = src.read(indexes).astype(np.float32)
        scales = [src.scales[index - 1] or 1.0 for index in indexes]
        offsets = [src.offsets[index - 1] or 0.0 for index in indexes]
        for position, (scale, offset) in enumerate(zip(scales, offsets)):
            data[position] = data[position] * scale + offset
        if float(np.nanmax(data)) > 1.5:
            data /= 10000.0
        data = np.clip(np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        info = RasterInfo(Path(path), (src.height, src.width), src.count, src.crs, src.transform, src.bounds, src.nodata, src.descriptions)
    return data, info


def pair_files(root: str | Path, split: str) -> list[tuple[Path, Path, Path | None]]:
    root = Path(root) / split
    lr_dir, hr_dir, label_dir = root / "lr", root / "hr", root / "labels"
    pairs = []
    for lr in sorted(lr_dir.glob("*")):
        if not lr.is_file():
            continue
        hr = next((candidate for candidate in hr_dir.glob(f"{lr.stem}.*") if candidate.is_file()), None)
        if hr is None:
            raise FileNotFoundError(f"Missing HR pair for {lr.name}")
        labels = next((candidate for candidate in label_dir.glob(f"{lr.stem}.*") if candidate.is_file()), None) if label_dir.exists() else None
        pairs.append((lr, hr, labels))
    return pairs


def iter_tiles(array: np.ndarray, tile_size: int, stride: int) -> Iterator[tuple[np.ndarray, tuple[int, int]]]:
    if array.ndim != 3:
        raise ValueError("Expected array shaped (bands, height, width)")
    height, width = array.shape[-2:]
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")
    for top in range(0, max(height - tile_size + 1, 1), stride):
        for left in range(0, max(width - tile_size + 1, 1), stride):
            bottom, right = min(top + tile_size, height), min(left + tile_size, width)
            tile = np.zeros((array.shape[0], tile_size, tile_size), dtype=array.dtype)
            tile[:, : bottom - top, : right - left] = array[:, top:bottom, left:right]
            yield tile, (top, left)


def resample_to_match(source: str | Path, reference: str | Path, *, categorical: bool = False) -> np.ndarray:
    with rasterio.open(reference) as ref:
        destination = np.zeros((ref.count, ref.height, ref.width), dtype=np.float32)
        destination_transform, destination_crs = ref.transform, ref.crs
    with rasterio.open(source) as src:
        for index in range(src.count):
            reproject(
                source=rasterio.band(src, index + 1), destination=destination[index],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=destination_transform, dst_crs=destination_crs,
                resampling=Resampling.nearest if categorical else Resampling.bilinear,
            )
    return destination
