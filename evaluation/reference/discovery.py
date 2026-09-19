"""
Reference Discovery Component
=============================
Inspects input scene spatial metadata and searches configured reference directories
for compatible high-resolution satellite imagery.

Priority:
1. real_hr: Already aligned and validated high-resolution reference.
2. external_hr: High-resolution image requiring alignment/reprojection.
3. synthetic_hr: Fallback to controlled synthetic degradation benchmark.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rasterio
    from rasterio.warp import transform_bounds
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False


@dataclass
class DiscoveredReference:
    """Metadata regarding a discovered reference."""
    found: bool = False
    reference_type: str = "synthetic_hr"  # "real_hr" | "external_hr" | "synthetic_hr"
    path: Path | None = None
    crs: str | None = None
    resolution: float | None = None
    bounds: tuple[float, float, float, float] | None = None
    overlap_pct: float = 0.0
    needs_alignment: bool = False
    band_count: int = 0
    valid_pixel_pct: float = 100.0
    discovery_notes: list[str] = field(default_factory=list)


class ReferenceDiscovery:
    """Discovers and validates external HR references for a given input scene."""

    def __init__(
        self,
        reference_dirs: list[str | Path] | None = None,
        min_overlap_pct: float = 50.0,
        max_acceptable_resolution_m: float = 5.0,
    ) -> None:
        self.reference_dirs = [
            Path(d) for d in (reference_dirs or ["reference_data", "data/reference", "dataset/reference"])
        ]
        self.min_overlap_pct = min_overlap_pct
        self.max_acceptable_resolution_m = max_acceptable_resolution_m

    def discover_for_geotiff(
        self,
        input_path: str | Path,
    ) -> DiscoveredReference:
        """Inspect input GeoTIFF and locate matching HR reference."""
        input_path = Path(input_path)
        notes: list[str] = []

        if not _RASTERIO_AVAILABLE:
            notes.append("rasterio not available; proceeding directly to synthetic benchmark.")
            return DiscoveredReference(
                found=False,
                reference_type="synthetic_hr",
                discovery_notes=notes,
            )

        if not input_path.exists():
            notes.append(f"Input file not found: {input_path}")
            return DiscoveredReference(
                found=False,
                reference_type="synthetic_hr",
                discovery_notes=notes,
            )

        try:
            with rasterio.open(input_path) as src:
                input_crs = str(src.crs)
                input_bounds = src.bounds
                input_res = float(src.res[0])
                input_bands = src.count
        except Exception as e:
            notes.append(f"Failed to read input GeoTIFF metadata: {e}")
            return DiscoveredReference(found=False, reference_type="synthetic_hr", discovery_notes=notes)

        return self.discover(
            input_crs=input_crs,
            input_bounds=(input_bounds.left, input_bounds.bottom, input_bounds.right, input_bounds.top),
            input_resolution=input_res,
            input_bands=input_bands,
            scene_name=input_path.stem,
        )

    def discover(
        self,
        input_crs: str,
        input_bounds: tuple[float, float, float, float],
        input_resolution: float,
        input_bands: int = 4,
        scene_name: str = "",
    ) -> DiscoveredReference:
        """Search reference directories for candidate matching references."""
        notes: list[str] = [
            f"Searching for reference matching CRS={input_crs}, bounds={input_bounds}, res={input_resolution}m"
        ]

        candidate_files: list[Path] = []
        for ref_dir in self.reference_dirs:
            if not ref_dir.exists():
                continue
            # Look inside scene subfolder or root of ref_dir
            candidate_files.extend(list(ref_dir.glob("**/*.tif")))
            candidate_files.extend(list(ref_dir.glob("**/*.tiff")))

        if not candidate_files:
            notes.append(
                f"No reference GeoTIFFs found across scanned directories: {[str(d) for d in self.reference_dirs]}. "
                "Triggering fallback: Synthetic Paired Benchmark (Priority 3)."
            )
            return DiscoveredReference(
                found=False,
                reference_type="synthetic_hr",
                discovery_notes=notes,
            )

        # Inspect candidate files
        for cand in candidate_files:
            try:
                with rasterio.open(cand) as rsrc:
                    cand_crs = str(rsrc.crs)
                    cand_bounds = rsrc.bounds
                    cand_res = float(rsrc.res[0])
                    cand_count = rsrc.count

                    # Resolution check: must be finer resolution than input
                    if cand_res >= input_resolution or cand_res > self.max_acceptable_resolution_m:
                        continue

                    # Spatial overlap check
                    overlap = self._compute_overlap(
                        (input_bounds[0], input_bounds[1], input_bounds[2], input_bounds[3]),
                        input_crs,
                        (cand_bounds.left, cand_bounds.bottom, cand_bounds.right, cand_bounds.top),
                        cand_crs,
                    )

                    if overlap >= self.min_overlap_pct:
                        needs_align = (cand_crs != input_crs) or (cand_res != input_resolution / 4.0) or (cand_bounds != input_bounds)
                        ref_type = "real_hr" if not needs_align else "external_hr"
                        notes.append(
                            f"Discovered candidate reference: {cand.name} (res={cand_res:.2f}m, overlap={overlap:.1f}%, type={ref_type})"
                        )
                        return DiscoveredReference(
                            found=True,
                            reference_type=ref_type,
                            path=cand,
                            crs=cand_crs,
                            resolution=cand_res,
                            bounds=(cand_bounds.left, cand_bounds.bottom, cand_bounds.right, cand_bounds.top),
                            overlap_pct=overlap,
                            needs_alignment=needs_align,
                            band_count=cand_count,
                            discovery_notes=notes,
                        )
            except Exception as ex:
                notes.append(f"Error inspecting candidate {cand.name}: {ex}")
                continue

        notes.append(
            "Scanned candidate files did not meet spatial overlap or resolution criteria. "
            "Triggering fallback: Synthetic Paired Benchmark (Priority 3)."
        )
        return DiscoveredReference(
            found=False,
            reference_type="synthetic_hr",
            discovery_notes=notes,
        )

    def _compute_overlap(
        self,
        b1: tuple[float, float, float, float],
        crs1: str,
        b2: tuple[float, float, float, float],
        crs2: str,
    ) -> float:
        """Compute intersection percentage between b1 and b2 in crs1 space."""
        try:
            if crs1 != crs2:
                b2 = transform_bounds(crs2, crs1, *b2)
            # Intersection of [left, bottom, right, top]
            inter_left = max(b1[0], b2[0])
            inter_bottom = max(b1[1], b2[1])
            inter_right = min(b1[2], b2[2])
            inter_top = min(b1[3], b2[3])

            if inter_right <= inter_left or inter_top <= inter_bottom:
                return 0.0

            inter_area = (inter_right - inter_left) * (inter_top - inter_bottom)
            b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
            if b1_area <= 0:
                return 0.0
            return float((inter_area / b1_area) * 100.0)
        except Exception:
            return 0.0
