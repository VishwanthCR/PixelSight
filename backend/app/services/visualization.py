from pathlib import Path

import numpy as np
from PIL import Image
import rasterio


def _stretch(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip((values - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def save_rgb_preview(raster_path: Path, output_path: Path) -> None:
    with rasterio.open(raster_path) as src:
        if src.count < 4:
            raise ValueError("RGB preview requires B02, B03, B04 and B08 bands.")
        blue, green, red = src.read([1, 2, 3]).astype(np.float32)

    rgb = np.stack([_stretch(red), _stretch(green), _stretch(blue)], axis=-1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path, format="PNG")
