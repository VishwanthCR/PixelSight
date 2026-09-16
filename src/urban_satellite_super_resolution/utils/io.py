"""Georeferenced output writing and report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import Affine


def output_profile(reference: str | Path, shape: tuple[int, int], count: int, dtype: str = "float32") -> dict[str, Any]:
    with rasterio.open(reference) as src:
        profile = src.profile.copy()
        profile.update(height=shape[0], width=shape[1], count=count, dtype=dtype, transform=src.transform * Affine.scale(src.width / shape[1], src.height / shape[0]), compress="deflate")
    return profile


def write_geotiff(path: str | Path, data: np.ndarray, reference: str | Path, *, categorical: bool = False) -> Path:
    path = Path(path)
    array = np.asarray(data)
    if array.ndim == 2:
        array = array[None, ...]
    dtype = "uint8" if categorical else "float32"
    with rasterio.open(path, "w", **output_profile(reference, array.shape[-2:], array.shape[0], dtype)) as dst:
        dst.write(array.astype(dtype))
    return path


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path
