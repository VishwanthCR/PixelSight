"""
PixelSight Super-Resolution Mapping (SRM) Sub-Pixel Conservation Analysis
========================================================================
Addresses SIH 2026 Problem Statement SIH26142:
"Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries"

Scientific Background:
----------------------
In classic remote sensing Super-Resolution Mapping (SRM) / Sub-Pixel Mapping,
a coarse (e.g., 10 m) mixed pixel consists of sub-pixel land cover fractions.
When a super-resolution model generates fine (e.g., 2.5 m, 4x4 sub-pixels) spatial
predictions, the aggregate abundance of each class across the 16 sub-pixels must
conserve the native coarse pixel's unmixed fractional abundance:

    sum_{j in subpixels} A_{j, c} = F_{c} * A_coarse

This module validates whether deep learning diffusion super-resolution obeys
physical sub-pixel conservation laws, or if the stochastic generative process
hallucinates or suppresses genuine land-cover classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SubPixelConservationResult:
    """Quantitative metrics measuring sub-pixel land-cover conservation."""

    # Sub-pixel scale factor (e.g., 4 for 10 m -> 2.5 m, yielding 16 sub-pixels per coarse pixel)
    scale_factor: int = 4
    num_subpixels_per_coarse: int = 16

    # Class-wise Sub-Pixel Fraction Error (SPFE): Mean Absolute Error between coarse abundance and SR sub-pixel fraction
    class_spfe: Dict[str, float] = field(default_factory=dict)
    mean_spfe: float = 0.0

    # Area Conservation Index (ACI): Ratio of total SR sub-pixel area to coarse unmixed area (1.0 = perfect conservation)
    class_aci: Dict[str, float] = field(default_factory=dict)
    mean_aci: float = 1.0

    # Unsupported Sub-Pixel Class Rate (UCR) [formerly Hallucinated Class Rate]:
    # Fraction of coarse pixels where coarse fraction < 0.05, but SR sub-pixels predict class presence.
    # Note: Without independent ground-truth surveys, this indicates lack of coarse spectral support,
    # not necessarily a synthetic fabrication/hallucination.
    unsupported_class_rate: Dict[str, float] = field(default_factory=dict)
    hallucinated_class_rate: Dict[str, float] = field(default_factory=dict)  # backward-compatible alias
    mean_unsupported_rate: float = 0.0
    mean_hallucinated_rate: float = 0.0  # backward-compatible alias

    # Omission Class Rate (OCR): Fraction of pixels where coarse fraction > 0.25, but SR sub-pixels predict 0
    omission_class_rate: Dict[str, float] = field(default_factory=dict)
    mean_omission_rate: float = 0.0

    # Overall Sub-Pixel Preservation Score (SPS) in [0, 1] (1.0 = perfect adherence to physical unmixing)
    subpixel_preservation_score: float = 0.0

    # Metric specifications providing formula, range, ideal value, and interpretation
    metric_specifications: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "SPFE": {
            "name": "Sub-Pixel Fraction Error",
            "formula": "mean(|f_SR - f_coarse|)",
            "range": "[0, 1]",
            "ideal": "0.0",
            "interpretation": "Average deviation between coarse mixed-pixel unmixing and 4x4 sub-pixel classification.",
        },
        "ACI": {
            "name": "Area Conservation Index",
            "formula": "sum(area_SR) / sum(area_coarse)",
            "range": "[0, inf)",
            "ideal": "1.0",
            "interpretation": "Macro-scale land cover area proportion conservation across the scene.",
        },
        "UCR": {
            "name": "Unsupported Sub-Pixel Class Rate",
            "formula": "P(f_SR >= 2/16 | f_coarse < 0.05)",
            "range": "[0, 1]",
            "ideal": "0.0",
            "interpretation": "Frequency of sub-pixel class emergence unsupported by native coarse endmembers.",
        },
        "SPS": {
            "name": "Sub-Pixel Preservation Score",
            "formula": "1.0 - (SPFE + 0.5*UCR + 0.5*OCR)",
            "range": "[0, 1]",
            "ideal": "1.0",
            "interpretation": "Overall composite measure of physical conservation fidelity.",
        },
    })

    scientific_interpretation: str = (
        "Quantifies physical adherence to sub-pixel land-cover abundance conservation. "
        "High SPFE or UCR indicates that the generative diffusion model assigns sub-pixel "
        "classes lacking strong optical evidence in the coarse native spectral mixture."
    )


class SubPixelMappingValidator:
    """Validates Super-Resolution Mapping (SRM) physical conservation laws."""

    def __init__(self, scale_factor: int = 4):
        self.scale_factor = scale_factor
        self.num_subpixels = scale_factor * scale_factor

    def estimate_coarse_endmembers(
        self,
        native_10m: np.ndarray,
        red_idx: int = 2,
        nir_idx: int = 3,
        green_idx: int = 1,
        blue_idx: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Estimate soft fractional abundances from native 10 m 4-band imagery.

        Computes standard optical spectral index fractions:
        - Vegetation: normalized NDVI in [0, 1]
        - Water: normalized NDWI in [0, 1]
        - Built-up / Bare: normalized brightness/soil index
        """
        arr = np.asarray(native_10m, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] in (3, 4):
            red = arr[red_idx]
            nir = arr[nir_idx]
            green = arr[green_idx]
            blue = arr[blue_idx]
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            red = arr[:, :, red_idx]
            nir = arr[:, :, nir_idx]
            green = arr[:, :, green_idx]
            blue = arr[:, :, blue_idx]
        else:
            raise ValueError(f"Expected 3D array with 4 bands, got shape {arr.shape}")

        with np.errstate(divide="ignore", invalid="ignore"):
            # NDVI for vegetation fraction
            ndvi = np.where((nir + red) > 1e-6, (nir - red) / (nir + red), 0.0)
            f_veg = np.clip((ndvi - 0.1) / 0.6, 0.0, 1.0)

            # NDWI (McFeeters: Green - NIR) for water fraction
            ndwi = np.where((green + nir) > 1e-6, (green - nir) / (green + nir), 0.0)
            f_water = np.clip((ndwi - 0.0) / 0.5, 0.0, 1.0)

            # Built-up / Soil fraction
            brightness = (red + green + blue) / 3.0
            f_built = np.clip((brightness - 0.15) / 0.4, 0.0, 1.0)

            # Normalize fractions so they sum to 1.0 per coarse pixel
            total = f_veg + f_water + f_built + 1e-8
            f_veg /= total
            f_water /= total
            f_built /= total

        return {
            "vegetation": f_veg.astype(np.float32),
            "water": f_water.astype(np.float32),
            "builtup_soil": f_built.astype(np.float32),
        }

    def compute_subpixel_fractions(
        self,
        sr_classes_2p5m: np.ndarray,
        class_mapping: Optional[Dict[str, int]] = None,
    ) -> Dict[str, np.ndarray]:
        """Aggregate 2.5 m sub-pixel classifications into coarse 10 m fractional blocks.

        Parameters
        ----------
        sr_classes_2p5m : np.ndarray, shape (H_sr, W_sr)
            Discrete classification map at 2.5 m resolution.
        class_mapping : Dict[str, int], optional
            Mapping of class name to integer label.
            Defaults to {"vegetation": 1, "water": 2, "builtup_soil": 3}.
        """
        if class_mapping is None:
            class_mapping = {"vegetation": 1, "water": 2, "builtup_soil": 3}

        sr_map = np.asarray(sr_classes_2p5m, dtype=np.int32)
        H_sr, W_sr = sr_map.shape
        s = self.scale_factor
        H_c = H_sr // s
        W_c = W_sr // s

        # Truncate to exact multiple of scale_factor
        sr_trimmed = sr_map[: H_c * s, : W_c * s]

        subpixel_fractions = {}
        for cls_name, cls_id in class_mapping.items():
            # Boolean mask for this class
            is_cls = (sr_trimmed == cls_id).astype(np.float32)
            # Reshape into (H_c, s, W_c, s) and average over sub-pixel dimensions
            reshaped = is_cls.reshape(H_c, s, W_c, s)
            coarse_fraction = reshaped.mean(axis=(1, 3))
            subpixel_fractions[cls_name] = coarse_fraction

        return subpixel_fractions

    def evaluate_conservation(
        self,
        native_10m: np.ndarray,
        sr_classes_2p5m: np.ndarray,
        class_mapping: Optional[Dict[str, int]] = None,
    ) -> SubPixelConservationResult:
        """Execute full Sub-Pixel Conservation Analysis."""
        if class_mapping is None:
            class_mapping = {"vegetation": 1, "water": 2, "builtup_soil": 3}

        # 1. Coarse unmixed fractions from 10 m
        coarse_abundances = self.estimate_coarse_endmembers(native_10m)

        # 2. Aggregated fractions from 2.5 m SR sub-pixels
        subpixel_fractions = self.compute_subpixel_fractions(sr_classes_2p5m, class_mapping)

        result = SubPixelConservationResult(
            scale_factor=self.scale_factor,
            num_subpixels_per_coarse=self.num_subpixels,
        )

        spfes = []
        acis = []
        hcrs = []
        ocrs = []

        for cls_name in class_mapping.keys():
            if cls_name not in coarse_abundances or cls_name not in subpixel_fractions:
                continue

            c_frac = coarse_abundances[cls_name]
            sr_frac = subpixel_fractions[cls_name]

            # Match spatial dimensions
            min_h = min(c_frac.shape[0], sr_frac.shape[0])
            min_w = min(c_frac.shape[1], sr_frac.shape[1])
            c_crop = c_frac[:min_h, :min_w]
            s_crop = sr_frac[:min_h, :min_w]

            # Sub-Pixel Fraction Error (SPFE)
            abs_diff = np.abs(s_crop - c_crop)
            spfe = float(np.mean(abs_diff))
            result.class_spfe[cls_name] = spfe
            spfes.append(spfe)

            # Area Conservation Index (ACI)
            c_sum = float(np.sum(c_crop))
            s_sum = float(np.sum(s_crop))
            aci = float(s_sum / (c_sum + 1e-8))
            result.class_aci[cls_name] = aci
            acis.append(aci)

            # Unsupported Sub-Pixel Class Rate (UCR): coarse fraction < 0.05 but SR fraction >= 2/16 sub-pixels
            unsupported_mask = (c_crop < 0.05) & (s_crop >= (2.0 / self.num_subpixels))
            ucr = float(np.mean(unsupported_mask))
            result.unsupported_class_rate[cls_name] = ucr
            result.hallucinated_class_rate[cls_name] = ucr  # backward-compatible alias
            hcrs.append(ucr)

            # Omission Class Rate (OCR): coarse fraction > 0.25 but SR fraction == 0.0
            omission_mask = (c_crop > 0.25) & (s_crop == 0.0)
            ocr = float(np.mean(omission_mask))
            result.omission_class_rate[cls_name] = ocr
            ocrs.append(ocr)

        result.mean_spfe = float(np.mean(spfes)) if spfes else 0.0
        result.mean_aci = float(np.mean(acis)) if acis else 1.0
        result.mean_unsupported_rate = float(np.mean(hcrs)) if hcrs else 0.0
        result.mean_hallucinated_rate = result.mean_unsupported_rate
        result.mean_omission_rate = float(np.mean(ocrs)) if ocrs else 0.0

        # Composite score in [0, 1]: 1.0 - (SPFE + 0.5 * UCR + 0.5 * OCR)
        penalty = result.mean_spfe + 0.5 * result.mean_unsupported_rate + 0.5 * result.mean_omission_rate
        result.subpixel_preservation_score = float(np.clip(1.0 - penalty, 0.0, 1.0))

        return result
