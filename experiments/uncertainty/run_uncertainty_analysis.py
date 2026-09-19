"""
PixelSight Scientific Experiment: Uncertainty and Reliability Analysis
======================================================================

Evaluates stochastic diffusion variation uncertainty produced by LDSR-S2 across
N random seeds (100 diffusion sampling steps per run).
Categorizes spatial reliability into HIGH / MEDIUM / LOW reconstruction-risk regions.
Evaluates the statistical relationship between uncertainty and downstream error.

Scientific Conventions:
- Described as "stochastic diffusion variation", NEVER "MC-dropout".
- Low reliability means "exercise caution", NOT "hallucinated pixels".
- Preserves negative or null correlation results.
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

from evaluation.uncertainty.analysis import (
    UncertaintyAnalysis,
    uncertainty_statistics,
    uncertainty_vs_error,
)
from evaluation.uncertainty.reliability import (
    ReliabilityMap,
    ReliabilityResult,
)
from evaluation.benchmarking.table_generator import generate_uncertainty_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PixelSight uncertainty and reliability analysis"
    )
    parser.add_argument(
        "--uncertainty-map",
        type=Path,
        default=PROJECT_ROOT / "results/plan3/uncertainty_map.npy",
        help="Path to per-pixel uncertainty numpy array",
    )
    parser.add_argument(
        "--mean-sr",
        type=Path,
        default=PROJECT_ROOT / "results/plan3/mean_sr.npy",
        help="Path to mean SR numpy array",
    )
    parser.add_argument(
        "--error-map",
        type=Path,
        default=PROJECT_ROOT / "results/plan4/error_map.npy",
        help="Path to optional error map array for correlation analysis",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/uncertainty",
        help="Directory to save uncertainty analysis outputs",
    )
    parser.add_argument(
        "--reliability-dir",
        type=Path,
        default=PROJECT_ROOT / "results/reliability",
        help="Directory to save reliability map outputs",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of stochastic diffusion samples",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=100,
        help="Diffusion sampling steps per sample (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without saving persistent files to disk",
    )
    return parser.parse_args()


def run():
    args = parse_args()
    print("=" * 70)
    print("PIXELSIGHT EXPERIMENT: UNCERTAINTY & RELIABILITY ANALYSIS")
    print("=" * 70)
    print(f"Uncertainty map path : {args.uncertainty_map}")
    print(f"Sampling steps       : {args.sampling_steps}")
    print(f"Diffusion samples    : {args.n_samples}")
    print(f"Dry run mode         : {args.dry_run}")
    print("-" * 70)

    if not args.uncertainty_map.exists():
        print(f"ERROR: Uncertainty map not found at {args.uncertainty_map}")
        sys.exit(1)

    umap = np.load(args.uncertainty_map)
    print(f"Loaded uncertainty map shape: {umap.shape}, dtype: {umap.dtype}")

    # 1. Compute descriptive uncertainty statistics
    analysis = UncertaintyAnalysis(
        uncertainty_map_path=args.uncertainty_map,
        mean_sr_path=args.mean_sr if args.mean_sr.exists() else None,
        n_samples=args.n_samples,
        sampling_steps=args.sampling_steps,
    )
    stats = analysis.compute_statistics()

    print("\n[1] Descriptive Statistics (Stochastic Diffusion Variation):")
    print(f"  Mean uncertainty   : {stats.mean:.6f}")
    print(f"  Median uncertainty : {stats.median:.6f}")
    print(f"  Std deviation      : {stats.std:.6f}")
    print(f"  Min / Max          : {stats.min:.6f} / {stats.max:.6f}")
    print(f"  p10 / p50 / p90    : {stats.p10:.6f} / {stats.median:.6f} / {stats.p90:.6f}")
    print(f"  Low threshold (p33): {stats.low_threshold:.6f}")
    print(f"  High threshold(p67): {stats.high_threshold:.6f}")
    print(f"  Low fraction       : {stats.low_uncertainty_fraction*100:.2f}%")
    print(f"  Medium fraction    : {stats.medium_uncertainty_fraction*100:.2f}%")
    print(f"  High fraction      : {stats.high_uncertainty_fraction*100:.2f}%")

    # 2. Compute reliability map
    rel_map = ReliabilityMap(
        uncertainty_map=umap,
        low_threshold=stats.low_threshold,
        high_threshold=stats.high_threshold,
    )
    error_arr = None
    if args.error_map.exists():
        error_arr = np.load(args.error_map)
        print(f"\n[2] Loaded downstream error map: {error_arr.shape}")
    else:
        print("\n[2] No error map supplied; skipping error validation on reliability map.")

    rel_res = rel_map.to_result(error_map=error_arr)
    print("Reliability Map Summary:")
    print(f"  High reliability fraction  : {rel_res.high_reliability_fraction*100:.2f}%")
    print(f"  Medium reliability fraction: {rel_res.medium_reliability_fraction*100:.2f}%")
    print(f"  Low reliability fraction   : {rel_res.low_reliability_fraction*100:.2f}%")
    if rel_res.validated_against_hr or error_arr is not None:
        print(f"  High reliability region error  : {rel_res.high_reliability_mean_error}")
        print(f"  Medium reliability region error: {rel_res.medium_reliability_mean_error}")
        print(f"  Low reliability region error   : {rel_res.low_reliability_mean_error}")

    # 3. Uncertainty vs Error relationship
    err_rel = None
    if error_arr is not None:
        err_rel = analysis.compute_error_relation(
            error_map=error_arr,
            error_type="downstream_segmentation_error_indicator",
            hr_reference_used=False,
            low_threshold=stats.low_threshold,
            high_threshold=stats.high_threshold,
        )
        print("\n[3] Uncertainty vs Error Correlation:")
        print(f"  Pearson correlation            : {err_rel.pearson_correlation}")
        print(f"  Spearman rank correlation      : {err_rel.spearman_correlation}")
        print(f"  Low uncertainty region error   : {err_rel.low_uncertainty_region_error}")
        print(f"  High uncertainty region error  : {err_rel.high_uncertainty_region_error}")
        if err_rel.high_vs_low_error_diff_pct_points is not None:
            print(f"  High vs Low error difference   : {err_rel.high_vs_low_error_diff_pct_points:.2f} percentage points")

    # 4. Generate structured table
    table_rows = generate_uncertainty_table(stats, err_rel)
    print("\n[4] Standard Uncertainty Table:")
    for r in table_rows:
        print(f"  {r}")

    if args.dry_run:
        print("\nDry-run complete. No files written.")
        return

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.reliability_dir.mkdir(parents=True, exist_ok=True)

    stats_dict = {
        "mechanism": stats.mechanism,
        "n_samples": stats.n_samples,
        "sampling_steps": stats.sampling_steps,
        "total_pixels": stats.total_pixels,
        "mean": stats.mean,
        "median": stats.median,
        "std": stats.std,
        "min": stats.min,
        "max": stats.max,
        "p10": stats.p10,
        "p25": stats.p25,
        "p75": stats.p75,
        "p90": stats.p90,
        "p99": stats.p99,
        "low_threshold": stats.low_threshold,
        "high_threshold": stats.high_threshold,
        "low_uncertainty_fraction": stats.low_uncertainty_fraction,
        "medium_uncertainty_fraction": stats.medium_uncertainty_fraction,
        "high_uncertainty_fraction": stats.high_uncertainty_fraction,
    }
    with open(args.output_dir / "uncertainty_statistics.json", "w") as f:
        json.dump(stats_dict, f, indent=2)

    if err_rel:
        rel_dict = {
            "error_type": err_rel.error_type,
            "hr_reference_used": err_rel.hr_reference_used,
            "pearson_correlation": err_rel.pearson_correlation,
            "spearman_correlation": err_rel.spearman_correlation,
            "low_uncertainty_region_error": err_rel.low_uncertainty_region_error,
            "medium_uncertainty_region_error": err_rel.medium_uncertainty_region_error,
            "high_uncertainty_region_error": err_rel.high_uncertainty_region_error,
            "high_vs_low_error_diff_pct_points": err_rel.high_vs_low_error_diff_pct_points,
            "interpretation": err_rel.interpretation,
        }
        with open(args.output_dir / "uncertainty_vs_error.json", "w") as f:
            json.dump(rel_dict, f, indent=2)

    rel_summary = {
        "high_reliability_fraction": rel_res.high_reliability_fraction,
        "medium_reliability_fraction": rel_res.medium_reliability_fraction,
        "low_reliability_fraction": rel_res.low_reliability_fraction,
        "high_reliability_pixels": rel_res.high_reliability_pixels,
        "medium_reliability_pixels": rel_res.medium_reliability_pixels,
        "low_reliability_pixels": rel_res.low_reliability_pixels,
        "total_pixels": rel_res.total_pixels,
        "uncertainty_threshold_low": rel_res.uncertainty_threshold_low,
        "uncertainty_threshold_high": rel_res.uncertainty_threshold_high,
        "limitation_note": rel_res.limitation_note,
    }
    with open(args.reliability_dir / "reliability_summary.json", "w") as f:
        json.dump(rel_summary, f, indent=2)

    np.save(args.reliability_dir / "reliability_map.npy", rel_map.array)
    print(f"\nSaved uncertainty analysis to: {args.output_dir}")
    print(f"Saved reliability analysis to: {args.reliability_dir}")


if __name__ == "__main__":
    run()
