"""
PixelSight Scientific Experiment: Spectral Fidelity & NDVI Analysis
====================================================================

Evaluates spectral fidelity across Native (10 m), Bicubic (2.5 m), and
PixelSight LDSR-S2 (~2.5 m) representations.

Scientific Conventions:
- NDVI = (B08 - B04) / (B08 + B04).
- Band indices: B04 (Red) = 2, B08 (NIR) = 3 in Sentinel-2 4-band stack.
- B08 must come from genuine Sentinel-2 data (NEVER synthesized from RGB).
- Native 10 m NDVI is the reference baseline for consistency diagnostics.
- When HR reference is unavailable, HR-referenced metrics (SAM, MAE vs HR)
  are explicitly reported as null / N/A.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.image_metrics.spectral import spectral_fidelity
from evaluation.downstream.ndvi import ndvi_comparison, compute_ndvi
from evaluation.benchmarking.table_generator import generate_spectral_table, save_tables


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PixelSight spectral fidelity and NDVI consistency analysis"
    )
    parser.add_argument(
        "--native-patch",
        type=Path,
        default=PROJECT_ROOT / "dataset/plan2/patches/train/patch_00000.npz",
        help="Path to native 10m Sentinel-2 patch (.npz or .npy)",
    )
    parser.add_argument(
        "--sr-image",
        type=Path,
        default=PROJECT_ROOT / "results/plan3/mean_sr.npy",
        help="Path to PixelSight LDSR-S2 SR image (.npy)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/evaluations/spectral",
        help="Directory to save spectral evaluation outputs",
    )
    parser.add_argument(
        "--hr-reference",
        type=Path,
        default=None,
        help="Optional genuine HR reference (.npy)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without saving persistent files to disk",
    )
    return parser.parse_args()


def load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        if "image" in data:
            arr = data["image"]
        else:
            arr = data[list(data.keys())[0]]
    else:
        arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    # Ensure (C, H, W)
    if arr.ndim == 3 and arr.shape[2] in (3, 4) and arr.shape[0] not in (3, 4):
        arr = np.transpose(arr, (2, 0, 1))
    return arr


def run():
    args = parse_args()
    print("=" * 70)
    print("PIXELSIGHT EXPERIMENT: SPECTRAL FIDELITY & NDVI ANALYSIS")
    print("=" * 70)
    print(f"Native input  : {args.native_patch}")
    print(f"SR input      : {args.sr_image}")
    print(f"HR reference  : {args.hr_reference} (Available: {args.hr_reference is not None})")
    print(f"Dry run mode  : {args.dry_run}")
    print("-" * 70)

    if not args.native_patch.exists():
        print(f"ERROR: Native patch not found at {args.native_patch}")
        sys.exit(1)
    if not args.sr_image.exists():
        print(f"ERROR: SR image not found at {args.sr_image}")
        sys.exit(1)

    native = load_array(args.native_patch)
    sr = load_array(args.sr_image)
    print(f"Loaded Native shape: {native.shape} (min={native.min():.4f}, max={native.max():.4f})")
    print(f"Loaded SR shape    : {sr.shape} (min={sr.min():.4f}, max={sr.max():.4f})")

    # Compute bicubic baseline matching SR shape
    from scipy.ndimage import zoom
    scale = sr.shape[1] / native.shape[1]
    bicubic = np.zeros_like(sr)
    for b in range(min(native.shape[0], sr.shape[0])):
        bicubic[b] = zoom(native[b], scale, order=3)
    bicubic = np.clip(bicubic, 0.0, 1.0)
    print(f"Computed Bicubic shape: {bicubic.shape}")

    # HR reference check
    hr_ref = None
    hr_available = False
    if args.hr_reference and args.hr_reference.exists():
        hr_ref = load_array(args.hr_reference)
        hr_available = True
        print(f"Loaded genuine HR reference: {hr_ref.shape}")
    else:
        print("Genuine HR reference unavailable. Reporting null/N/A for reference metrics.")

    # 1. NDVI consistency analysis
    ndvi_res = ndvi_comparison(
        native=native,
        sr=sr,
        bicubic=bicubic,
        reference_is_genuine_hr=hr_available,
    )

    print("\n[1] NDVI Consistency Diagnostics (Native 10 m as baseline):")
    print(f"  Native NDVI  : Mean = {ndvi_res.native_mean:.4f}, Std = {ndvi_res.native_std:.4f}")
    print(f"  Bicubic NDVI : Mean = {ndvi_res.bicubic_mean:.4f}, Std = {ndvi_res.bicubic_std:.4f}")
    print(f"                 MAE vs Native = {ndvi_res.bicubic_mae_vs_native:.4f}, RMSE = {ndvi_res.bicubic_rmse_vs_native:.4f}")
    print(f"  SR NDVI      : Mean = {ndvi_res.sr_mean:.4f}, Std = {ndvi_res.sr_std:.4f}")
    print(f"                 MAE vs Native = {ndvi_res.sr_mae_vs_native:.4f}, RMSE = {ndvi_res.sr_rmse_vs_native:.4f}")
    if ndvi_res.mae_improvement_pct is not None:
        print(f"  MAE change vs Bicubic : {ndvi_res.mae_improvement_pct:+.2f}%")

    # 2. Spectral fidelity metrics
    spec_sr = spectral_fidelity(
        prediction=sr,
        native_array=native,
        bicubic_array=bicubic,
        hr_reference=hr_ref,
        hr_reference_available=hr_available,
    )
    spec_bic = spectral_fidelity(
        prediction=bicubic,
        native_array=native,
        bicubic_array=bicubic,
        hr_reference=hr_ref,
        hr_reference_available=hr_available,
    )

    print("\n[2] Spectral Metrics Summary:")
    print(f"  HR reference available : {spec_sr.hr_reference_available}")
    print(f"  SAM (SR vs HR)         : {spec_sr.sam_degrees}")
    print(f"  SAM (Bicubic vs HR)    : {spec_bic.sam_degrees}")

    # 3. Generate standard spectral table
    bic_dict = {
        "ndvi_mae_vs_native": ndvi_res.bicubic_mae_vs_native,
        "ndvi_rmse_vs_native": ndvi_res.bicubic_rmse_vs_native,
        "ndvi_mean": ndvi_res.bicubic_mean,
        "sam_degrees": spec_bic.sam_degrees,
    }
    sr_dict = {
        "ndvi_mae_vs_native": ndvi_res.sr_mae_vs_native,
        "ndvi_rmse_vs_native": ndvi_res.sr_rmse_vs_native,
        "ndvi_mean": ndvi_res.sr_mean,
        "sam_degrees": spec_sr.sam_degrees,
    }
    table_rows = generate_spectral_table(
        bicubic_spectral=bic_dict,
        pixelsight_spectral=sr_dict,
        hr_reference_available=hr_available,
    )
    print("\n[3] Standard Spectral Table:")
    for r in table_rows:
        print(f"  {r}")

    if args.dry_run:
        print("\nDry-run complete. No files written.")
        return

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "hr_reference_available": hr_available,
        "native_ndvi": {
            "mean": ndvi_res.native_mean,
            "std": ndvi_res.native_std,
            "min": ndvi_res.native_min,
            "max": ndvi_res.native_max,
        },
        "bicubic_ndvi": {
            "mean": ndvi_res.bicubic_mean,
            "std": ndvi_res.bicubic_std,
            "mae_vs_native": ndvi_res.bicubic_mae_vs_native,
            "rmse_vs_native": ndvi_res.bicubic_rmse_vs_native,
        },
        "sr_ndvi": {
            "mean": ndvi_res.sr_mean,
            "std": ndvi_res.sr_std,
            "mae_vs_native": ndvi_res.sr_mae_vs_native,
            "rmse_vs_native": ndvi_res.sr_rmse_vs_native,
        },
        "mae_improvement_pct": ndvi_res.mae_improvement_pct,
        "rmse_improvement_pct": ndvi_res.rmse_improvement_pct,
        "sam_degrees": spec_sr.sam_degrees,
        "scientific_note": (
            "NDVI comparisons use native 10 m NDVI as a reference baseline (consistency diagnostic). "
            "HR reference metrics (SAM, MAE vs HR) are null because no independent HR reference is present."
        ),
    }

    with open(args.output_dir / "spectral_fidelity.json", "w") as f:
        json.dump(summary, f, indent=2)

    save_tables(args.output_dir, spectral_rows=table_rows)
    print(f"\nSaved spectral evaluation to: {args.output_dir}")


if __name__ == "__main__":
    run()
