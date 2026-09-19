"""
Scientific result table generator.

Produces the four standard research tables for PixelSight experiments.
Values that are unavailable (no HR reference, missing data) are represented
as None/NaN in the data structures and as empty strings or "N/A" in output.

DO NOT invent missing values.
"""

from __future__ import annotations

import json
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _fmt(value: float | None, decimals: int = 4) -> str:
    """Format a value for table display."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    return f"{value:.{decimals}f}"


def _row(label: str, d: dict[str, Any], keys: list[str], decimals: int = 4) -> dict[str, str]:
    row = {"Method": label}
    for key, display_key in keys:
        row[display_key] = _fmt(d.get(key), decimals)
    return row


# ---------------------------------------------------------------------------
# Image-level table
# ---------------------------------------------------------------------------

def generate_image_level_table(
    benchmark_result: Any | None = None,
    bicubic_data: dict | None = None,
    pixelsight_data: dict | None = None,
    hr_reference_available: bool = False,
) -> list[dict[str, str]]:
    """Generate the image-level comparison table.

    | Method | PSNR | SSIM | SAM | MAE | RMSE |
    |--------|------|------|-----|-----|------|

    Parameters
    ----------
    benchmark_result : BenchmarkResult, optional
        Pre-computed benchmark result.
    bicubic_data, pixelsight_data : dict, optional
        Fallback: raw metric dicts.
    hr_reference_available : bool
        If False, PSNR/SSIM/SAM/MAE/RMSE are all N/A.
    """
    rows = []
    note = (
        "N/A = metric requires genuine HR reference which is unavailable."
        if not hr_reference_available
        else ""
    )

    if benchmark_result is not None:
        bic = benchmark_result.bicubic
        pxs = benchmark_result.pixelsight
        rows.append({
            "Method": "Bicubic 2.5 m",
            "PSNR (dB)": _fmt(bic.psnr) if hr_reference_available else "N/A",
            "SSIM": _fmt(bic.ssim) if hr_reference_available else "N/A",
            "SAM (°)": _fmt(bic.sam_degrees) if hr_reference_available else "N/A",
            "MAE": _fmt(bic.mae_hr) if hr_reference_available else "N/A",
            "RMSE": _fmt(bic.rmse_hr) if hr_reference_available else "N/A",
        })
        rows.append({
            "Method": "PixelSight LDSR-S2 (~2.5 m)",
            "PSNR (dB)": _fmt(pxs.psnr) if hr_reference_available else "N/A",
            "SSIM": _fmt(pxs.ssim) if hr_reference_available else "N/A",
            "SAM (°)": _fmt(pxs.sam_degrees) if hr_reference_available else "N/A",
            "MAE": _fmt(pxs.mae_hr) if hr_reference_available else "N/A",
            "RMSE": _fmt(pxs.rmse_hr) if hr_reference_available else "N/A",
        })
    elif bicubic_data and pixelsight_data:
        for label, data in [("Bicubic 2.5 m", bicubic_data),
                            ("PixelSight LDSR-S2 (~2.5 m)", pixelsight_data)]:
            rows.append({
                "Method": label,
                "PSNR (dB)": _fmt(data.get("psnr")) if hr_reference_available else "N/A",
                "SSIM": _fmt(data.get("ssim")) if hr_reference_available else "N/A",
                "SAM (°)": _fmt(data.get("sam_degrees")) if hr_reference_available else "N/A",
                "MAE": _fmt(data.get("mae_hr")) if hr_reference_available else "N/A",
                "RMSE": _fmt(data.get("rmse_hr")) if hr_reference_available else "N/A",
            })
    rows.append({"Method": "NOTE", "PSNR (dB)": note, "SSIM": "", "SAM (°)": "", "MAE": "", "RMSE": ""})
    return rows


# ---------------------------------------------------------------------------
# Downstream table
# ---------------------------------------------------------------------------

def generate_downstream_table(
    native_seg: dict | None = None,
    bicubic_seg: dict | None = None,
    pixelsight_seg: dict | None = None,
    label_is_proxy: bool = True,
) -> list[dict[str, str]]:
    """Generate the downstream segmentation comparison table.

    | Method | IoU | F1 | Dice | Accuracy | Precision | Recall |
    """
    if is_dataclass(native_seg):
        native_seg = asdict(native_seg)
    if is_dataclass(bicubic_seg):
        bicubic_seg = asdict(bicubic_seg)
    if is_dataclass(pixelsight_seg):
        pixelsight_seg = asdict(pixelsight_seg)
    rows = []
    entries = [
        ("Native 10 m (observed)", native_seg),
        ("Bicubic 2.5 m", bicubic_seg),
        ("PixelSight LDSR-S2 (~2.5 m)", pixelsight_seg),
    ]
    for label, data in entries:
        if data is None:
            continue
        rows.append({
            "Method": label,
            "mIoU": _fmt(data.get("mean_iou") or data.get("mIoU")),
            "mF1/Dice": _fmt(data.get("mean_dice") or data.get("dice")),
            "Accuracy": _fmt(data.get("pixel_accuracy")),
            "Precision": _fmt(data.get("mean_precision") or data.get("precision")),
            "Recall": _fmt(data.get("mean_recall") or data.get("recall")),
        })

    if label_is_proxy:
        rows.append({
            "Method": "NOTE",
            "mIoU": (
                "Labels are 10 m WorldCover proxy replicated to 2.5 m.  "
                "The SR evaluation does not use genuine 2.5 m labels."
            ),
            "mF1/Dice": "", "Accuracy": "", "Precision": "", "Recall": "",
        })
    return rows


# ---------------------------------------------------------------------------
# Spectral table
# ---------------------------------------------------------------------------

def generate_spectral_table(
    bicubic_spectral: dict | None = None,
    pixelsight_spectral: dict | None = None,
    hr_reference_available: bool = False,
) -> list[dict[str, str]]:
    """Generate the spectral consistency comparison table.

    | Method | NDVI MAE vs native | NDVI RMSE vs native | SAM (°) |
    """
    if is_dataclass(bicubic_spectral):
        bicubic_spectral = asdict(bicubic_spectral)
    if is_dataclass(pixelsight_spectral):
        pixelsight_spectral = asdict(pixelsight_spectral)
    rows = []
    entries = [
        ("Bicubic 2.5 m", bicubic_spectral),
        ("PixelSight LDSR-S2 (~2.5 m)", pixelsight_spectral),
    ]
    for label, data in entries:
        if data is None:
            continue
        rows.append({
            "Method": label,
            "NDVI MAE vs native": _fmt(
                data.get("ndvi_mae_vs_native") or data.get("ndvi_mae_bicubic_vs_native")
                or data.get("LDSR_NDVI_MAE_vs_native")
            ),
            "NDVI RMSE vs native": _fmt(
                data.get("ndvi_rmse_vs_native") or data.get("ndvi_rmse_bicubic_vs_native")
                or data.get("LDSR_NDVI_RMSE_vs_native")
            ),
            "SAM (°)": _fmt(data.get("sam_degrees")) if hr_reference_available else "N/A",
            "NDVI Mean": _fmt(data.get("ndvi_mean") or data.get("ndvi_mean_bicubic") or data.get("ndvi_mean_sr")),
        })

    note = (
        "NDVI comparisons use native 10 m NDVI as reference baseline (consistency diagnostic). "
        + ("SAM N/A — HR reference unavailable." if not hr_reference_available else "")
    )
    rows.append({"Method": "NOTE", "NDVI MAE vs native": note, "NDVI RMSE vs native": "",
                 "SAM (°)": "", "NDVI Mean": ""})
    return rows


# ---------------------------------------------------------------------------
# Uncertainty table
# ---------------------------------------------------------------------------

def generate_uncertainty_table(
    uncertainty_stats: Any | None = None,
    uncertainty_error_relation: Any | None = None,
) -> list[dict[str, str]]:
    """Generate the uncertainty breakdown table.

    | Uncertainty Level | Region Error Rate | Mean Uncertainty | Region Pixels |
    """
    if is_dataclass(uncertainty_stats):
        uncertainty_stats = asdict(uncertainty_stats)
    if is_dataclass(uncertainty_error_relation):
        uncertainty_error_relation = asdict(uncertainty_error_relation)

    rows = []

    if uncertainty_error_relation:
        for level, err_key, unc_key in [
            ("Low",    "low_uncertainty_region_error",    "low_threshold"),
            ("Medium", "medium_uncertainty_region_error", None),
            ("High",   "high_uncertainty_region_error",   "high_threshold"),
        ]:
            rows.append({
                "Uncertainty Level": level,
                "Region Error Rate": _fmt(uncertainty_error_relation.get(err_key)),
                "Pearson Corr (unc-err)": _fmt(
                    uncertainty_error_relation.get("pearson_correlation"), 4
                ) if level == "Low" else "",
            })

    if uncertainty_stats:
        rows.append({"Uncertainty Level": "---", "Region Error Rate": "---",
                     "Pearson Corr (unc-err)": "---"})
        rows.append({"Uncertainty Level": "Mean uncertainty",
                     "Region Error Rate": _fmt(uncertainty_stats.get("mean"), 6),
                     "Pearson Corr (unc-err)": ""})
        rows.append({"Uncertainty Level": "Median uncertainty",
                     "Region Error Rate": _fmt(uncertainty_stats.get("median"), 6),
                     "Pearson Corr (unc-err)": ""})
        rows.append({"Uncertainty Level": "P90 uncertainty",
                     "Region Error Rate": _fmt(uncertainty_stats.get("p90"), 6),
                     "Pearson Corr (unc-err)": ""})
        rows.append({"Uncertainty Level": "Max uncertainty",
                     "Region Error Rate": _fmt(uncertainty_stats.get("max"), 6),
                     "Pearson Corr (unc-err)": ""})

    rows.append({
        "Uncertainty Level": "NOTE",
        "Region Error Rate": (
            "Uncertainty = stochastic diffusion variation (std across 5 seeds). "
            "This is NOT MC-dropout or Bayesian predictive uncertainty."
        ),
        "Pearson Corr (unc-err)": "",
    })
    return rows


# ---------------------------------------------------------------------------
# Save all tables
# ---------------------------------------------------------------------------

def save_tables(
    output_dir: str | Path,
    image_level_rows: list[dict] | None = None,
    downstream_rows: list[dict] | None = None,
    spectral_rows: list[dict] | None = None,
    uncertainty_rows: list[dict] | None = None,
) -> dict[str, Path]:
    """Save all tables as CSV and JSON.

    Returns
    -------
    dict mapping table name to Path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = {}
    for name, rows in [
        ("image_level", image_level_rows),
        ("downstream", downstream_rows),
        ("spectral", spectral_rows),
        ("uncertainty", uncertainty_rows),
    ]:
        if rows is None:
            continue

        json_path = output_dir / f"{name}_table.json"
        csv_path = output_dir / f"{name}_table.csv"

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)

        # CSV: simple manual writer (avoids pandas requirement)
        if rows:
            headers = list(rows[0].keys())
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                fh.write(",".join(f'"{h}"' for h in headers) + "\n")
                for row in rows:
                    fh.write(",".join(f'"{row.get(h, "")}"' for h in headers) + "\n")

        saved[name] = csv_path

    if _PANDAS_AVAILABLE and saved:
        # Also write markdown tables
        for name, csv_path in saved.items():
            try:
                df = pd.read_csv(csv_path)
                md_path = output_dir / f"{name}_table.md"
                md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
            except Exception:
                pass  # markdown export is best-effort

    return saved
