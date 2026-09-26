"""
Reference Alignment Service
===========================
Reprojects, clips to AOI, resamples to the exact 2.5m Sentinel-2 SR grid,
and generates aligned_reference.tif and alignment provenance reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False


@dataclass
class AlignmentReport:
    """Detailed provenance of reference alignment transformations."""
    original_crs: str = ""
    target_crs: str = ""
    original_resolution: float = 0.0
    target_resolution: float = 0.0
    original_dimensions: tuple[int, int, int] = (0, 0, 0)
    aligned_dimensions: tuple[int, int, int] = (0, 0, 0)
    geographic_bounds: list[float] = field(default_factory=list)
    resampling_method: str = "bilinear"
    valid_pixel_percentage: float = 100.0
    alignment_status: str = "SUCCESS"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReferenceAligner:
    """Aligns an external HR satellite reference with target SR specifications."""

    def __init__(self, resampling_method: str = "bilinear") -> None:
        self.resampling_method = resampling_method

    def align_reference(
        self,
        reference_path: str | Path,
        target_template_path: str | Path,
        output_dir: str | Path,
        band_mapping: dict[str, int | None] | None = None,
        target_scale: int = 4,
    ) -> tuple[Path, AlignmentReport]:
        """Reproject, crop, and resample reference_path to match target_template_path at 4x SR grid."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        aligned_tif_path = output_dir / "aligned_reference.tif"
        report_path = output_dir / "reference_alignment_report.json"

        report = AlignmentReport(resampling_method=self.resampling_method)

        if not _RASTERIO_AVAILABLE:
            report.alignment_status = "FAILED_NO_RASTERIO"
            report.notes.append("rasterio library is required for GeoTIFF reprojection.")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
            return aligned_tif_path, report

        with rasterio.open(target_template_path) as tgt:
            target_crs = tgt.crs
            tgt_bounds = tgt.bounds
            # If target template is already super-resolved (e.g. width >= 512), do not upscale again
            eff_scale = 1 if tgt.width >= 512 else target_scale
            tgt_res = float(tgt.res[0]) / float(eff_scale)
            tgt_width = tgt.width * eff_scale
            tgt_height = tgt.height * eff_scale
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

            # Determine bands to read
            out_channels = 4
            dest_data = np.zeros((out_channels, tgt_height, tgt_width), dtype=np.float32)

            resampling_enum = Resampling.bilinear if self.resampling_method == "bilinear" else Resampling.cubic

            # Map channels if provided, else copy available bands up to 4
            if band_mapping:
                for target_idx, band_name in enumerate(["B02", "B03", "B04", "B08"]):
                    src_idx = band_mapping.get(band_name)
                    if src_idx is not None and 0 <= src_idx < src.count:
                        reproject(
                            source=rasterio.band(src, src_idx + 1),
                            destination=dest_data[target_idx],
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=target_transform,
                            dst_crs=target_crs,
                            resampling=resampling_enum,
                        )
            else:
                for b in range(1, min(src.count, out_channels) + 1):
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
            valid_pct = float(np.mean(valid_mask) * 100.0) if valid_mask.size > 0 else 0.0
            report.valid_pixel_percentage = valid_pct

            out_meta = {
                "driver": "GTiff",
                "count": out_channels,
                "dtype": "float32",
                "width": tgt_width,
                "height": tgt_height,
                "crs": target_crs,
                "transform": target_transform,
            }

            with rasterio.open(aligned_tif_path, "w", **out_meta) as dst:
                dst.write(dest_data)

        report.alignment_status = "SUCCESS"
        report.notes.append(f"Aligned and clipped to {tgt_width}x{tgt_height} @ {tgt_res:.2f}m resolution.")

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        return aligned_tif_path, report
