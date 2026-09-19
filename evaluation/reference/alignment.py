"""
Reference Alignment Component
=============================
Reprojects, aligns geographic footprints, resamples to target resolution,
and generates aligned_reference.tif and reference_alignment_report.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False


@dataclass
class AlignmentReport:
    """Report detailing all alignment transformations performed on the reference."""
    original_crs: str = ""
    target_crs: str = ""
    original_resolution: float = 0.0
    target_resolution: float = 0.0
    original_dimensions: tuple[int, int, int] = (0, 0, 0)  # (C, H, W)
    aligned_dimensions: tuple[int, int, int] = (0, 0, 0)   # (C, H, W)
    geographic_bounds: list[float] = field(default_factory=list)  # [left, bottom, right, top]
    resampling_method: str = "bilinear"
    valid_pixel_percentage: float = 100.0
    alignment_status: str = "SUCCESS"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReferenceAligner:
    """Aligns an external HR satellite reference image with target scene specifications."""

    def __init__(self, resampling_method: str = "bilinear") -> None:
        self.resampling_method = resampling_method

    def align_geotiff(
        self,
        reference_path: str | Path,
        target_template_path: str | Path,
        output_dir: str | Path,
        target_scale: int = 4,
    ) -> tuple[Path, AlignmentReport]:
        """Reproject, crop, and resample reference_path to match target_template_path at 4x SR grid.

        Parameters
        ----------
        reference_path : Path to raw external HR reference GeoTIFF.
        target_template_path : Path to native 10 m input GeoTIFF.
        output_dir : Destination for aligned_reference.tif and report.
        target_scale : Resolution multiplier (default 4x, e.g. 10m -> 2.5m).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        aligned_tif_path = output_dir / "aligned_reference.tif"
        report_path = output_dir / "reference_alignment_report.json"

        report = AlignmentReport(resampling_method=self.resampling_method)

        if not _RASTERIO_AVAILABLE:
            report.alignment_status = "FAILED_NO_RASTERIO"
            report.notes.append("rasterio library is required for GeoTIFF reprojection and alignment.")
            with open(report_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            return aligned_tif_path, report

        with rasterio.open(target_template_path) as tgt:
            target_crs = tgt.crs
            tgt_bounds = tgt.bounds
            tgt_res = float(tgt.res[0]) / float(target_scale)  # target SR resolution (e.g. 2.5m)
            tgt_width = tgt.width * target_scale
            tgt_height = tgt.height * target_scale
            # Construct target affine transform
            from rasterio.transform import from_bounds
            target_transform = from_bounds(
                tgt_bounds.left, tgt_bounds.bottom, tgt_bounds.right, tgt_bounds.top,
                tgt_width, tgt_height
            )

        with rasterio.open(reference_path) as src:
            report.original_crs = str(src.crs)
            report.target_crs = str(target_crs)
            report.original_resolution = float(src.res[0])
            report.target_resolution = tgt_res
            report.original_dimensions = (src.count, src.height, src.width)
            report.aligned_dimensions = (src.count, tgt_height, tgt_width)
            report.geographic_bounds = [tgt_bounds.left, tgt_bounds.bottom, tgt_bounds.right, tgt_bounds.top]

            # Allocate destination array
            dest_data = np.zeros((src.count, tgt_height, tgt_width), dtype=np.float32)

            resampling_enum = Resampling.bilinear if self.resampling_method == "bilinear" else Resampling.cubic

            for b in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, b),
                    destination=dest_data[b - 1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=resampling_enum,
                )

            # Valid pixels check
            valid_mask = np.isfinite(dest_data) & (dest_data > 0)
            valid_pct = float(np.mean(valid_mask) * 100.0)
            report.valid_pixel_percentage = valid_pct

            # Write aligned GeoTIFF
            out_meta = {
                "driver": "GTiff",
                "count": src.count,
                "dtype": "float32",
                "width": tgt_width,
                "height": tgt_height,
                "crs": target_crs,
                "transform": target_transform,
            }

            with rasterio.open(aligned_tif_path, "w", **out_meta) as dst:
                dst.write(dest_data)

        report.alignment_status = "SUCCESS"
        report.notes.append(f"Aligned successfully to {tgt_width}x{tgt_height} @ {tgt_res}m spacing.")

        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        return aligned_tif_path, report

    def align_arrays(
        self,
        reference_arr: np.ndarray,
        target_shape: tuple[int, int],
    ) -> tuple[np.ndarray, AlignmentReport]:
        """Pure array-based alignment/resampling fallback for memory buffers."""
        from scipy.ndimage import zoom
        ref = np.asarray(reference_arr, dtype=np.float32)
        if ref.ndim == 2:
            ref = ref[np.newaxis]

        c, h, w = ref.shape
        scale_y = target_shape[0] / h
        scale_x = target_shape[1] / w

        aligned = np.zeros((c, target_shape[0], target_shape[1]), dtype=np.float32)
        for i in range(c):
            aligned[i] = zoom(ref[i], (scale_y, scale_x), order=1)

        valid_pct = float(np.mean(np.isfinite(aligned)) * 100.0)
        report = AlignmentReport(
            original_dimensions=(c, h, w),
            aligned_dimensions=(c, target_shape[0], target_shape[1]),
            valid_pixel_percentage=valid_pct,
            alignment_status="SUCCESS_ARRAY",
        )
        return aligned, report
