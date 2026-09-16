"""Synthetic urban multispectral data for tests and demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter, zoom


def degrade(high_resolution: np.ndarray, scale: int = 4, noise_std: float = 0.01, blur_sigma: float = 1.0, seed: int = 42) -> np.ndarray:
    """Create an LR observation with blur, downsampling, and sensor noise."""
    if high_resolution.ndim != 3:
        raise ValueError("Expected CHW array")
    blurred = gaussian_filter(high_resolution, sigma=(0, blur_sigma, blur_sigma))
    low_resolution = zoom(blurred, (1, 1 / scale, 1 / scale), order=3)
    generator = np.random.default_rng(seed)
    return np.clip(low_resolution + generator.normal(0, noise_std, low_resolution.shape), 0.0, 1.0).astype(np.float32)


def generate_sample(output: str | Path, *, width: int = 128, height: int = 128, seed: int = 42) -> dict[str, Path]:
    """Write a small georeferenced HR/LR/label sample without downloading data."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:height, :width]
    buildings = ((x - width * 0.28) ** 2 / 500 + (y - height * 0.35) ** 2 / 260 < 1) | ((x - width * 0.7) ** 2 / 750 + (y - height * 0.65) ** 2 / 500 < 1)
    roads = (np.abs(y - height * 0.5) < 4) | (np.abs(x - width * 0.55) < 3)
    trees = ((x - width * 0.2) ** 2 + (y - height * 0.78) ** 2 < (width * 0.16) ** 2)
    water = (y < height * 0.12)
    hr = np.stack([
        np.where(water, 0.06, np.where(trees, 0.08, np.where(buildings, 0.35, 0.18))),
        np.where(water, 0.18, np.where(trees, 0.25, np.where(buildings, 0.30, 0.20))),
        np.where(water, 0.10, np.where(trees, 0.10, np.where(buildings, 0.32, 0.22))),
        np.where(trees, 0.55, np.where(water, 0.04, 0.25)),
    ]).astype(np.float32)
    hr += rng.normal(0, 0.01, hr.shape).astype(np.float32)
    hr = np.clip(hr, 0.0, 1.0)
    labels = np.full((height, width), 0, dtype=np.uint8)
    labels[trees] = 3
    labels[buildings] = 0
    labels[roads] = 1
    labels[water] = 4
    transform = from_origin(500000, 2000000, 2.5, 2.5)
    paths = {"hr": output / "scene_hr.tif", "lr": output / "scene_lr.tif", "labels": output / "scene_labels.tif"}
    lr = degrade(hr, scale=4, seed=seed)
    for key, data, resolution, descriptions in [
        ("hr", hr, 2.5, ("B02", "B03", "B04", "B08")),
        ("lr", lr, 10.0, ("B02", "B03", "B04", "B08")),
    ]:
        path = paths[key]
        with rasterio.open(path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2], count=4, dtype="float32", crs="EPSG:32643", transform=from_origin(500000, 2000000, resolution, resolution)) as dst:
            dst.write(data)
            for index, name in enumerate(descriptions, 1):
                dst.set_band_description(index, name)
    with rasterio.open(paths["labels"], "w", driver="GTiff", height=height, width=width, count=1, dtype="uint8", crs="EPSG:32643", transform=transform, nodata=255) as dst:
        dst.write(labels, 1)
    return paths
