"""
PixelSight Scientific Experiment: Downstream Task Performance Summary
=====================================================================

Aggregates downstream segmentation and classification metrics across Native (10 m)
and PixelSight LDSR-S2 (~2.5 m) representations.

Scientific Conventions:
- Preserves and honestly reports the negative result:
  Downstream segmentation accuracy on 10 m WorldCover proxy labels is lower
  for LDSR-S2 2.5 m (pixel accuracy ~45.1%, mIoU ~0.122) than for Native 10 m
  (pixel accuracy ~76.0%, mIoU ~0.301).
- Documents the scientific reasons:
  1. Proxy label limitation: WorldCover 10 m labels are nearest-neighbor/bilinearly
     upscaled to 2.5 m; they lack genuine sub-pixel ground truth boundaries.
  2. Domain shift: The downstream segmenter was trained on native 10 m reflectance
     distributions, not on diffusion-generated high-frequency spatial patterns.
  3. High-frequency generative texture: Generative diffusion introduces fine-scale
     textural details that can confuse coarse 10 m classifiers.
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

from evaluation.downstream.segmentation import (
    segmentation_metrics,
    SegmentationResult,
)
from evaluation.benchmarking.table_generator import generate_downstream_table, save_tables


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PixelSight downstream performance aggregation"
    )
    parser.add_argument(
        "--comparison-json",
        type=Path,
        default=PROJECT_ROOT / "results/plan4/downstream_comparison.json",
        help="Path to pre-computed downstream comparison JSON",
    )
    parser.add_argument(
        "--native-pred",
        type=Path,
        default=PROJECT_ROOT / "results/plan4/native_prediction.npy",
        help="Path to native segmentation prediction (.npy)",
    )
    parser.add_argument(
        "--sr-pred",
        type=Path,
        default=PROJECT_ROOT / "results/plan4/sr_prediction.npy",
        help="Path to SR segmentation prediction (.npy)",
    )
    parser.add_argument(
        "--labels-native",
        type=Path,
        default=PROJECT_ROOT / "dataset/segmentation/patches/train/labels/r00000_c00000.npy",
        help="Path to native 10 m WorldCover labels (.npy)",
    )
    parser.add_argument(
        "--labels-sr-proxy",
        type=Path,
        default=PROJECT_ROOT / "results/plan4/worldcover_native_512.npy",
        help="Path to 2.5 m proxy WorldCover labels (.npy)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/evaluations/downstream",
        help="Directory to save downstream evaluation outputs",
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
    print("PIXELSIGHT EXPERIMENT: DOWNSTREAM TASK EVALUATION")
    print("=" * 70)
    print(f"Comparison JSON: {args.comparison_json}")
    print(f"Output dir     : {args.output_dir}")
    print(f"Dry run mode   : {args.dry_run}")
    print("-" * 70)

    native_metrics = None
    sr_metrics = None

    # Load precomputed comparison json first (canonical experiment record)
    if args.comparison_json.exists():
        with open(args.comparison_json) as f:
            data = json.load(f)
        native_raw = data.get("native_10m", {})
        sr_raw = data.get("ldsr_s2_2p5m_proxy", {})
        print("Loaded precomputed downstream comparison results from JSON.")
        native_metrics = native_raw
        sr_metrics = sr_raw

    # Array validation if files are present and match shapes
    if args.native_pred.exists() and args.labels_native.exists():
        native_pred = np.load(args.native_pred)
        labels_nat = np.load(args.labels_native)
        if native_pred.shape == labels_nat.shape:
            computed_native = segmentation_metrics(
                prediction=native_pred,
                labels=labels_nat,
                representation="native_10m",
                label_is_proxy=False,
                label_type="worldcover_10m",
            )
            print(f"Verified Native metrics from arrays: acc={computed_native.pixel_accuracy:.4f}, mIoU={computed_native.mean_iou:.4f}")
            if not native_metrics:
                native_metrics = {
                    "pixel_accuracy": computed_native.pixel_accuracy,
                    "mean_iou": computed_native.mean_iou,
                    "mean_dice": computed_native.mean_dice,
                    "mean_precision": computed_native.mean_precision,
                    "mean_recall": computed_native.mean_recall,
                }
        else:
            print(f"Note: Native pred shape {native_pred.shape} != label shape {labels_nat.shape}; using JSON metrics.")

    if args.sr_pred.exists() and args.labels_sr_proxy.exists():
        sr_pred = np.load(args.sr_pred)
        labels_sr = np.load(args.labels_sr_proxy)
        if sr_pred.shape == labels_sr.shape:
            computed_sr = segmentation_metrics(
                prediction=sr_pred,
                labels=labels_sr,
                representation="pixelsight_ldsr_s2_2p5m",
                label_is_proxy=True,
                label_type="worldcover_10m_upscaled_proxy",
                methodological_note="Labels are 10m WorldCover upscaled 4x to 2.5m as proxy. Not genuine 2.5m ground truth.",
            )
            print(f"Verified SR metrics from arrays: acc={computed_sr.pixel_accuracy:.4f}, mIoU={computed_sr.mean_iou:.4f}")
            if not sr_metrics:
                sr_metrics = {
                    "pixel_accuracy": computed_sr.pixel_accuracy,
                    "mean_iou": computed_sr.mean_iou,
                    "mean_dice": computed_sr.mean_dice,
                    "mean_precision": computed_sr.mean_precision,
                    "mean_recall": computed_sr.mean_recall,
                }
        else:
            print(f"Note: SR pred shape {sr_pred.shape} != label shape {labels_sr.shape}; using JSON metrics.")

    print("\n[1] Downstream Performance Summary:")
    print("  Native 10 m (Observed Baseline):")
    print(f"    Pixel Accuracy : {native_metrics.get('pixel_accuracy', 0):.4f}")
    print(f"    Mean IoU       : {native_metrics.get('mean_iou', 0):.4f}")
    print(f"    Mean Dice / F1 : {native_metrics.get('mean_dice', 0):.4f}")
    print(f"    Precision      : {native_metrics.get('mean_precision', 0):.4f}")
    print(f"    Recall         : {native_metrics.get('mean_recall', 0):.4f}")

    print("\n  PixelSight LDSR-S2 (~2.5 m with Proxy Labels):")
    print(f"    Pixel Accuracy : {sr_metrics.get('pixel_accuracy', 0):.4f}")
    print(f"    Mean IoU       : {sr_metrics.get('mean_iou', 0):.4f}")
    print(f"    Mean Dice / F1 : {sr_metrics.get('mean_dice', 0):.4f}")
    print(f"    Precision      : {sr_metrics.get('mean_precision', 0):.4f}")
    print(f"    Recall         : {sr_metrics.get('mean_recall', 0):.4f}")

    print("\n[2] Scientific Findings & Negative Result Reporting:")
    acc_diff = (sr_metrics.get("pixel_accuracy", 0) - native_metrics.get("pixel_accuracy", 0)) * 100
    miou_diff = (sr_metrics.get("mean_iou", 0) - native_metrics.get("mean_iou", 0)) * 100
    print(f"  Pixel Accuracy change: {acc_diff:+.2f} percentage points")
    print(f"  Mean IoU change      : {miou_diff:+.2f} percentage points")
    print("  Analysis: Super-resolution degraded downstream segmentation accuracy on proxy labels.")
    print("  Root causes:")
    print("    1. Proxy-label bias: WorldCover 10 m labels lack genuine 2.5 m sub-pixel boundaries.")
    print("    2. Model domain shift: Segmenter was calibrated on smoother 10 m Sentinel-2 reflectance.")
    print("    3. Generative texture: High-frequency details generated by diffusion introduce boundary noise.")

    # Generate standard downstream table
    table_rows = generate_downstream_table(
        native_seg=native_metrics,
        bicubic_seg=None,  # Bicubic segmentation not computed in P4
        pixelsight_seg=sr_metrics,
        label_is_proxy=True,
    )

    print("\n[3] Standard Downstream Table:")
    for r in table_rows:
        print(f"  {r}")

    if args.dry_run:
        print("\nDry-run complete. No files written.")
        return

    # Save summary
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_data = {
        "native_10m": native_metrics,
        "pixelsight_ldsr_s2_2p5m_proxy": sr_metrics,
        "negative_result_analysis": {
            "accuracy_difference_pct_points": acc_diff,
            "mean_iou_difference_pct_points": miou_diff,
            "finding": "Super-resolution degraded downstream segmentation performance on proxy labels.",
            "explanations": [
                "10m WorldCover proxy labels lack true sub-pixel boundaries.",
                "Downstream model suffers domain shift on diffusion-generated textures.",
                "Diffusion high frequencies introduce edge ambiguity for coarse classifiers."
            ]
        },
        "methodological_note": (
            "The LDSR-S2 2.5m evaluation uses 10m WorldCover labels replicated as a proxy reference. "
            "This is not genuine 2.5m ground truth."
        )
    }

    with open(args.output_dir / "downstream_summary.json", "w") as f:
        json.dump(out_data, f, indent=2)

    save_tables(args.output_dir, downstream_rows=table_rows)
    print(f"\nSaved downstream evaluation to: {args.output_dir}")


if __name__ == "__main__":
    run()
