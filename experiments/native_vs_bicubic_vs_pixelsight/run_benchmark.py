"""
PixelSight Scientific Experiment: Controlled 3-Way Benchmark
============================================================
Native (10 m) vs Bicubic (2.5 m) vs PixelSight LDSR-S2 (~2.5 m)

A rigorous, controlled benchmark evaluating spatial detail, spectral consistency,
downstream task utility, and reconstruction uncertainty.

Scientific Conventions:
- All 3 representations are evaluated on identical spatial bounds.
- Bicubic is a baseline, NEVER an HR reference.
- HR-reference metrics (PSNR, SSIM, SAM) report N/A when genuine HR reference
  is unavailable.
- Downstream negative results are explicitly preserved and documented.
- Full provenance (model, steps=100, scale=4, bands=B02/B03/B04/B08) is recorded.
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

from evaluation.benchmarking.three_way import (
    ThreeWayBenchmark,
    BenchmarkProvenance,
    BenchmarkResult,
)
from evaluation.downstream.segmentation import SegmentationResult
from evaluation.benchmarking.table_generator import (
    generate_image_level_table,
    generate_downstream_table,
    generate_spectral_table,
    generate_uncertainty_table,
    save_tables,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PixelSight controlled 3-way benchmark"
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
        "--downstream-json",
        type=Path,
        default=PROJECT_ROOT / "results/plan4/downstream_comparison.json",
        help="Path to downstream segmentation comparison JSON",
    )
    parser.add_argument(
        "--uncertainty-stats",
        type=Path,
        default=PROJECT_ROOT / "results/uncertainty/uncertainty_statistics.json",
        help="Path to uncertainty statistics JSON",
    )
    parser.add_argument(
        "--uncertainty-error",
        type=Path,
        default=PROJECT_ROOT / "results/uncertainty/uncertainty_vs_error.json",
        help="Path to uncertainty vs error JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/comparisons",
        help="Directory to save 3-way benchmark comparison tables and results",
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
    if arr.ndim == 3 and arr.shape[2] in (3, 4) and arr.shape[0] not in (3, 4):
        arr = np.transpose(arr, (2, 0, 1))
    return arr


def run():
    args = parse_args()
    print("=" * 70)
    print("PIXELSIGHT: CONTROLLED 3-WAY BENCHMARK")
    print("Native (10 m) vs Bicubic (2.5 m) vs PixelSight LDSR-S2 (~2.5 m)")
    print("=" * 70)
    print(f"Native input  : {args.native_patch}")
    print(f"SR input      : {args.sr_image}")
    print(f"HR reference  : {args.hr_reference} (Available: {args.hr_reference is not None})")
    print(f"Output dir    : {args.output_dir}")
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
    print(f"Native shape: {native.shape}, SR shape: {sr.shape}")

    hr_ref = None
    hr_available = False
    if args.hr_reference and args.hr_reference.exists():
        hr_ref = load_array(args.hr_reference)
        hr_available = True
        print(f"Loaded genuine HR reference: {hr_ref.shape}")
    else:
        print("Genuine HR reference unavailable. PSNR/SSIM/SAM vs HR will report N/A.")

    # Downstream segmentation ingestion
    native_seg = None
    sr_seg = None
    if args.downstream_json.exists():
        with open(args.downstream_json) as f:
            d_data = json.load(f)
        nat_d = d_data.get("native_10m", {})
        sr_d = d_data.get("ldsr_s2_2p5m_proxy", {})
        native_seg = SegmentationResult(
            representation="native_10m",
            pixel_accuracy=nat_d.get("pixel_accuracy"),
            mean_iou=nat_d.get("mean_iou"),
            mean_dice=nat_d.get("mean_dice"),
            mean_precision=nat_d.get("mean_precision"),
            mean_recall=nat_d.get("mean_recall"),
            label_is_proxy=False,
            label_type="worldcover_10m",
        )
        sr_seg = SegmentationResult(
            representation="pixelsight_ldsr_s2_2p5m",
            pixel_accuracy=sr_d.get("pixel_accuracy"),
            mean_iou=sr_d.get("mean_iou"),
            mean_dice=sr_d.get("mean_dice"),
            mean_precision=sr_d.get("mean_precision"),
            mean_recall=sr_d.get("mean_recall"),
            label_is_proxy=True,
            label_type="worldcover_10m_proxy",
            methodological_note="10m WorldCover proxy replicated to 2.5m.",
        )
        print("Ingested downstream segmentation evaluation.")

    provenance = BenchmarkProvenance(
        model="LDSR-S2",
        checkpoint="opensr-ldsrs2_v1_0_0.ckpt",
        sampling_steps=100,
        scale=4,
        bands=["B02", "B03", "B04", "B08"],
        input_scene="patch_00000",
        crs="EPSG:32643",
        input_resolution_m=10.0,
        output_resolution_m=2.5,
        n_uncertainty_samples=5,
        random_seed="0,1,2,3,4",
        hr_reference_available=hr_available,
        limitations=[
            "No genuine HR reference available (PSNR/SSIM/SAM vs HR are N/A)",
            "Downstream WorldCover labels are 10m proxy replicated to 2.5m",
            "Super-resolution degraded downstream segmentation accuracy on proxy labels",
            "Uncertainty represents stochastic diffusion variation across seeds, not MC-dropout",
        ],
    )

    benchmark = ThreeWayBenchmark(
        native_array=native,
        sr_array=sr,
        hr_reference=hr_ref,
        hr_reference_available=hr_available,
        provenance=provenance,
    )

    result = benchmark.run(
        native_segmentation=native_seg,
        bicubic_segmentation=None,
        sr_segmentation=sr_seg,
    )

    # Uncertainty data for table
    unc_stats = None
    unc_err = None
    if args.uncertainty_stats.exists():
        with open(args.uncertainty_stats) as f:
            unc_stats = json.load(f)
    if args.uncertainty_error.exists():
        with open(args.uncertainty_error) as f:
            unc_err = json.load(f)

    # 1. Image-level table
    img_table = generate_image_level_table(
        benchmark_result=result,
        hr_reference_available=hr_available,
    )

    # 2. Downstream table
    down_table = generate_downstream_table(
        native_seg=native_seg,
        bicubic_seg=None,
        pixelsight_seg=sr_seg,
        label_is_proxy=True,
    )

    # 3. Spectral table
    spec_table = generate_spectral_table(
        bicubic_spectral={
            "ndvi_mae_vs_native": result.bicubic.ndvi_mae_vs_native,
            "ndvi_rmse_vs_native": result.bicubic.ndvi_rmse_vs_native,
            "ndvi_mean": result.bicubic.ndvi_mean,
            "sam_degrees": result.bicubic.sam_degrees,
        },
        pixelsight_spectral={
            "ndvi_mae_vs_native": result.pixelsight.ndvi_mae_vs_native,
            "ndvi_rmse_vs_native": result.pixelsight.ndvi_rmse_vs_native,
            "ndvi_mean": result.pixelsight.ndvi_mean,
            "sam_degrees": result.pixelsight.sam_degrees,
        },
        hr_reference_available=hr_available,
    )

    # 4. Uncertainty table
    unc_table = generate_uncertainty_table(
        uncertainty_stats=unc_stats,
        uncertainty_error_relation=unc_err,
    )

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)

    print("\n--- [Table 1: Image-Level Fidelity] ---")
    for r in img_table:
        print(f"  {r}")

    print("\n--- [Table 2: Downstream Task Performance] ---")
    for r in down_table:
        print(f"  {r}")

    print("\n--- [Table 3: Spectral Consistency (NDVI)] ---")
    for r in spec_table:
        print(f"  {r}")

    print("\n--- [Table 4: Reconstruction Uncertainty & Reliability] ---")
    for r in unc_table:
        print(f"  {r}")

    print("\n--- [Negative Findings & Limitations] ---")
    for nf in result.negative_findings:
        print(f"  * {nf}")
    for lim in provenance.limitations:
        print(f"  - Limitation: {lim}")

    if args.dry_run:
        print("\nDry-run complete. No files written.")
        return

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_tables(
        args.output_dir,
        image_level_rows=img_table,
        downstream_rows=down_table,
        spectral_rows=spec_table,
        uncertainty_rows=unc_table,
    )

    bench_dict = {
        "provenance": {
            "model": provenance.model,
            "checkpoint": provenance.checkpoint,
            "sampling_steps": provenance.sampling_steps,
            "scale": provenance.scale,
            "bands": provenance.bands,
            "input_scene": provenance.input_scene,
            "crs": provenance.crs,
            "hr_reference_available": provenance.hr_reference_available,
            "limitations": provenance.limitations,
        },
        "negative_findings": result.negative_findings,
        "spatial": {
            "native_gradient_energy": result.native.gradient_energy,
            "bicubic_gradient_energy": result.bicubic.gradient_energy,
            "pixelsight_gradient_energy": result.pixelsight.gradient_energy,
        },
        "ndvi": {
            "native_mean": result.native.ndvi_mean,
            "bicubic_mae_vs_native": result.bicubic.ndvi_mae_vs_native,
            "pixelsight_mae_vs_native": result.pixelsight.ndvi_mae_vs_native,
        },
    }
    with open(args.output_dir / "benchmark_summary.json", "w") as f:
        json.dump(bench_dict, f, indent=2)

    print(f"\nSaved 3-way benchmark comparison tables and summary to: {args.output_dir}")


if __name__ == "__main__":
    run()
