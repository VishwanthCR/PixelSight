"""Uncertainty proxy generation for the deterministic LDSR-S2 application path."""

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import zoom


def _normalize(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def generate_uncertainty_map(input_path: Path, sr_path: Path, output_tif: Path, output_png: Path) -> dict:
    """Create an interpretable reconstruction-uncertainty proxy for the SR output.

    LDSR-S2 inference in the web pipeline is deterministic. This map therefore
    measures normalized spectral disagreement from bicubic interpolation, not
    calibrated Bayesian uncertainty or observed ground-truth error.
    """
    with rasterio.open(input_path) as source:
        native = source.read().astype(np.float32)
    with rasterio.open(sr_path) as source:
        sr = source.read().astype(np.float32)
        profile = source.profile.copy()

    if native.shape[0] != sr.shape[0]:
        raise ValueError("Input and SR rasters must have the same number of bands.")
    for array in (native, sr):
        if float(np.nanmax(array)) > 1.5:
            array /= 10000.0
    native = np.clip(np.nan_to_num(native, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    sr = np.clip(np.nan_to_num(sr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    scale_y = sr.shape[1] / native.shape[1]
    scale_x = sr.shape[2] / native.shape[2]
    bicubic = zoom(native, (1, scale_y, scale_x), order=3)
    if bicubic.shape != sr.shape:
        bicubic = bicubic[:, : sr.shape[1], : sr.shape[2]]
    deviation = np.mean(np.abs(sr - bicubic), axis=0)
    uncertainty = _normalize(deviation)

    output_tif.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=1, dtype="float32", nodata=None, compress="deflate")
    with rasterio.open(output_tif, "w", **profile) as destination:
        destination.write(uncertainty.astype(np.float32), 1)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((uncertainty * 255.0).astype(np.uint8), mode="L").save(output_png, format="PNG")
    high = uncertainty >= 0.66
    medium = (uncertainty >= 0.33) & ~high
    low = uncertainty < 0.33
    return {
        "status": "proxy",
        "method": "normalized spectral deviation from bicubic baseline",
        "map": "uncertainty/uncertainty_map.png",
        "geotiff": "uncertainty/uncertainty_map.tif",
        "mean": float(uncertainty.mean()),
        "peak": float(uncertainty.max()),
        "low_percent": float(low.mean() * 100.0),
        "medium_percent": float(medium.mean() * 100.0),
        "high_percent": float(high.mean() * 100.0),
        "limitation": "This is a reconstruction-uncertainty proxy because the current web LDSR-S2 run is deterministic; it is not calibrated predictive uncertainty.",
    }
