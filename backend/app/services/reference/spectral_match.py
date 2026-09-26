"""
Spectral Compatibility Component
=================================
Validates spectral channels, band wavelengths, and determines metric eligibility
for external HR references evaluated against PixelSight 4-band LDSR-S2 outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Standard Sentinel-2 L2A target bands for LDSR-S2
PIXELSIGHT_TARGET_BANDS = ["B02", "B03", "B04", "B08"]
PIXELSIGHT_BAND_DESCRIPTIONS = {
    "B02": "Blue (~490 nm, 10m)",
    "B03": "Green (~560 nm, 10m)",
    "B04": "Red (~665 nm, 10m)",
    "B08": "NIR (~842 nm, 10m)",
}


@dataclass
class SpectralMatchResult:
    """Result of spectral band compatibility validation."""
    compatibility: str  # "FULL" | "RGB_ONLY" | "PARTIAL" | "INCOMPATIBLE"
    target_bands: list[str]
    reference_bands: list[str]
    band_mapping: dict[str, int | None]  # Map target band name -> reference band 0-indexed position
    has_rgb: bool
    has_nir: bool
    eligible_metrics: list[str]          # Metrics permitted to be computed
    ineligible_metrics: dict[str, str]   # Metric name -> reason ineligible
    notes: list[str]


class SpectralMatcher:
    """Enforces spectral fidelity and metric eligibility rules."""

    def __init__(self, target_bands: list[str] | None = None) -> None:
        self.target_bands = [b.upper() for b in (target_bands or PIXELSIGHT_TARGET_BANDS)]

    def evaluate(
        self,
        reference_bands: list[str] | None,
        reference_band_count: int | None = None,
    ) -> SpectralMatchResult:
        """Inspect reference bands and determine valid evaluation metrics."""
        notes: list[str] = []
        raw_bands = [str(b).upper().strip() for b in (reference_bands or [])]

        if not raw_bands and reference_band_count:
            if reference_band_count >= 4:
                raw_bands = ["B02", "B03", "B04", "B08"]
            elif reference_band_count == 3:
                raw_bands = ["RED", "GREEN", "BLUE"]
            else:
                raw_bands = [f"BAND_{i+1}" for i in range(reference_band_count)]

        # Normalized mapping
        band_map: dict[str, int | None] = {"B02": None, "B03": None, "B04": None, "B08": None}

        # Check for Sentinel-2 naming or standard optical colors
        for idx, b in enumerate(raw_bands):
            if b in ("B02", "BLUE", "B2", "B"):
                if band_map["B02"] is None:
                    band_map["B02"] = idx
            elif b in ("B03", "GREEN", "B3", "G"):
                if band_map["B03"] is None:
                    band_map["B03"] = idx
            elif b in ("B04", "RED", "B4", "R"):
                if band_map["B04"] is None:
                    band_map["B04"] = idx
            elif b in ("B08", "NIR", "NEAR_INFRARED", "B8", "N"):
                if band_map["B08"] is None:
                    band_map["B08"] = idx

        # Fallback for standard 4-band order (R, G, B, NIR) or (B, G, R, NIR)
        if len(raw_bands) >= 4 and all(v is None for v in band_map.values()):
            # Typical S2 L2A 4-band order: B02, B03, B04, B08
            band_map["B02"] = 0
            band_map["B03"] = 1
            band_map["B04"] = 2
            band_map["B08"] = 3
            notes.append("Inferred standard 4-band Sentinel-2 order [B02, B03, B04, B08].")
        elif len(raw_bands) == 3 and all(v is None for v in band_map.values()):
            # Standard RGB order: R=0, G=1, B=2
            band_map["B04"] = 0
            band_map["B03"] = 1
            band_map["B02"] = 2
            band_map["B08"] = None
            notes.append("Inferred standard 3-band RGB order without NIR.")

        has_b02 = band_map["B02"] is not None
        has_b03 = band_map["B03"] is not None
        has_b04 = band_map["B04"] is not None
        has_b08 = band_map["B08"] is not None

        has_rgb = has_b02 and has_b03 and has_b04
        has_nir = has_b08

        eligible_metrics: list[str] = []
        ineligible_metrics: dict[str, str] = {}

        if has_rgb and has_nir:
            compatibility = "FULL"
            eligible_metrics = ["mse", "rmse", "mae", "psnr", "ssim", "sam", "spectral_mae", "edge_sharpness"]
            notes.append("Full spectral compatibility: All 4 multispectral bands (B02, B03, B04, B08) available.")
        elif has_rgb and not has_nir:
            compatibility = "RGB_ONLY"
            eligible_metrics = ["mse", "rmse", "mae", "psnr", "ssim", "edge_sharpness"]
            ineligible_metrics["sam"] = "Reference does not provide compatible NIR band (B08); SAM cannot be computed."
            ineligible_metrics["spectral_mae"] = "Reference lacks NIR band; full 4-band spectral MAE disabled."
            notes.append("RGB-only reference: PSNR and SSIM eligible on visible bands; SAM strictly withheld.")
        elif (has_b04 and has_b08) or (has_b02 or has_b03):
            compatibility = "PARTIAL"
            eligible_metrics = ["mae", "edge_sharpness"]
            ineligible_metrics["psnr"] = "Incomplete visible channels; standard full-spectrum PSNR invalid."
            ineligible_metrics["ssim"] = "Incomplete visible channels; multi-channel SSIM invalid."
            ineligible_metrics["sam"] = "Incomplete 4-band spectral coverage; SAM withheld."
            notes.append("Partial spectral overlap: Only subset of channels mapped.")
        else:
            compatibility = "INCOMPATIBLE"
            ineligible_metrics = {
                "psnr": "Reference bands incompatible with Sentinel-2 VNIR spectrum.",
                "ssim": "Reference bands incompatible with Sentinel-2 VNIR spectrum.",
                "sam": "Reference bands incompatible with Sentinel-2 VNIR spectrum.",
                "mae": "Reference bands incompatible with Sentinel-2 VNIR spectrum.",
                "rmse": "Reference bands incompatible with Sentinel-2 VNIR spectrum.",
            }
            notes.append("Incompatible reference: No reliable spectral mapping to B02, B03, B04, or B08.")

        return SpectralMatchResult(
            compatibility=compatibility,
            target_bands=self.target_bands,
            reference_bands=raw_bands,
            band_mapping=band_map,
            has_rgb=has_rgb,
            has_nir=has_nir,
            eligible_metrics=eligible_metrics,
            ineligible_metrics=ineligible_metrics,
            notes=notes,
        )
