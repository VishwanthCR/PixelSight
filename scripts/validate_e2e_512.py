"""
PixelSight E2E 512x512 Validation Script
========================================
Validates the complete end-to-end pipeline on `dataset/plan2/geotiff/train_test_512_10m.tif`
across all four supported domains:
1. Research Core Engine
2. Crop Monitoring
3. Urban Analysis
4. Disaster Management
"""

import json
import sys
from pathlib import Path
import shutil
import time
import numpy as np
import rasterio
import torch

# Ensure the project root is in sys.path regardless of invocation directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.services.core_engine import PixelSightEngine
from backend.app.services.crop import run_crop_analysis
from backend.app.services.disaster import align_temporal_pair, run_disaster_analysis
from backend.app.services.reporting import write_manifest, write_report
from backend.app.services.urban import run_urban_pipeline


INPUT_512 = Path("dataset/plan2/geotiff/train_test_512_10m.tif")
OUTPUT_ROOT = Path("results/e2e_validation_512")


def run_validation():
    print(f"============================================================")
    print(f"PIXELSIGHT E2E 512x512 VALIDATION")
    print(f"============================================================")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Input : {INPUT_512} (exists: {INPUT_512.exists()})")
    print(f"Compute Device : {device}")

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Inspect
    print("\n[Stage 1] Inspecting 512x512 raster...")
    inspection = PixelSightEngine.inspect(INPUT_512)
    print(f"  Dimensions: {inspection.width} x {inspection.height}")
    print(f"  Bands: {inspection.band_names}")
    print(f"  CRS: {inspection.crs}")
    print(f"  Compatible: {inspection.compatible}")
    assert inspection.compatible, f"Input rejected: {inspection.errors}"

    # 2. Preprocessing
    print("\n[Stage 2] Preprocessing to float32 reflectance...")
    norm_path = OUTPUT_ROOT / "preprocessing" / "normalized.tif"
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    _, pre_ops = PixelSightEngine.preprocess(INPUT_512, norm_path)
    print(f"  Preprocessed raster saved at {norm_path}")

    # 3. Super-Resolution via Core Engine (100 steps, 4x)
    print("\n[Stage 3] Running PixelSight Core Engine Enhancement (4x, 100 diffusion steps)...")
    sr_path = OUTPUT_ROOT / "super_resolution" / "sr.tif"
    sr_preview = OUTPUT_ROOT / "super_resolution" / "sr_preview.png"
    t0 = time.perf_counter()
    sr_meta, dev_used = PixelSightEngine.enhance(
        normalized_path=norm_path,
        output_sr_path=sr_path,
        preview_path=sr_preview,
        device=device,
    )
    sr_duration = time.perf_counter() - t0
    print(f"  Enhancement completed in {sr_duration:.2f}s ({sr_meta['model']}, device: {dev_used})")

    with rasterio.open(sr_path) as src:
        print(f"  SR Dimensions: {src.width} x {src.height} (Scale factor 4x)")
        print(f"  SR CRS: {src.crs}")
        assert src.width == 512 * 4 and src.height == 512 * 4

    # 4. Uncertainty
    print("\n[Stage 4] Generating stochastic diffusion uncertainty map...")
    unc_tif = OUTPUT_ROOT / "uncertainty" / "uncertainty_map.tif"
    unc_png = OUTPUT_ROOT / "uncertainty" / "uncertainty_map.png"
    unc_meta = PixelSightEngine.uncertainty(norm_path, sr_path, unc_tif, unc_png)
    print(f"  Mean Uncertainty: {unc_meta.get('mean', '—')}")

    # 5. Crop Application Pipeline
    print("\n[Stage 5] Running Crop Monitoring Pipeline...")
    crop_results = run_crop_analysis(norm_path, sr_path, OUTPUT_ROOT)
    print(f"  Native Mean NDVI: {crop_results['statistics']['native']['mean']:.4f}")
    print(f"  SR Mean NDVI: {crop_results['statistics']['super_resolution']['mean']:.4f}")
    print(f"  Consistency MAE: {crop_results['consistency_metrics']['mae']:.4f}")

    # 6. Urban Application Pipeline
    print("\n[Stage 6] Running Urban Analysis Pipeline...")
    urban_results = run_urban_pipeline(norm_path, sr_path, OUTPUT_ROOT, device=device)
    print(f"  Built-up Area: {urban_results['indicators']['built_up_fraction']*100:.1f}%")
    print(f"  Vegetation Area: {urban_results['indicators']['vegetation_fraction']*100:.1f}%")

    # 7. Disaster Application Pipeline
    print("\n[Stage 7] Running Disaster Management Pipeline (Temporal Change)...")
    # For temporal pair, simulate post-event acquisition with localized radiometric change
    post_norm = OUTPUT_ROOT / "preprocessing" / "post_simulated.tif"
    with rasterio.open(norm_path) as src:
        prof = src.profile.copy()
        post_data = src.read().astype(np.float32)
        # Introduce change in central quadrant
        h, w = post_data.shape[1], post_data.shape[2]
        post_data[:, h//4:3*h//4, w//4:3*w//4] *= 0.65
    with rasterio.open(post_norm, "w", **prof) as dst:
        dst.write(post_data)

    sr_post = OUTPUT_ROOT / "application" / "disaster" / "post_event" / "sr_post.tif"
    sr_post_prev = OUTPUT_ROOT / "application" / "disaster" / "previews" / "post_sr_preview.png"
    sr_post_prev.parent.mkdir(parents=True, exist_ok=True)
    PixelSightEngine.enhance(post_norm, sr_post, preview_path=sr_post_prev, device=device)

    post_unc_tif = OUTPUT_ROOT / "application" / "disaster" / "post_event" / "unc_post.tif"
    post_unc_png = OUTPUT_ROOT / "application" / "disaster" / "post_event" / "unc_post.png"
    PixelSightEngine.uncertainty(post_norm, sr_post, post_unc_tif, post_unc_png)

    disaster_results = run_disaster_analysis(sr_path, sr_post, unc_tif, post_unc_tif, OUTPUT_ROOT)
    print(f"  Changed Area: {disaster_results['statistics']['change_percentage']:.2f}%")
    print(f"  Lower Uncertainty Fraction: {disaster_results['statistics']['reliability_breakdown']['lower_uncertainty_fraction_of_change']*100:.1f}%")

    # 8. Reports & Manifest
    print("\n[Stage 8] Generating multi-format reports and manifest...")
    output_files = {}
    output_files.update(crop_results["outputs"])
    output_files.update(urban_results["outputs"])
    output_files.update(disaster_results["outputs"])

    report = write_report(
        OUTPUT_ROOT / "report" / "report.json",
        job_id="e2e_512_validation",
        input_metadata=inspection.model_dump(),
        preprocessing=pre_ops,
        runtime_seconds=sr_duration,
        device=dev_used,
        output_files=output_files,
        application="research",
        application_data={"crop": crop_results, "urban": urban_results, "disaster": disaster_results},
    )
    manifest = write_manifest(OUTPUT_ROOT, "e2e_512_validation", report)
    print(f"  Manifest written with {len(manifest['artifacts'])} catalogued artifacts.")

    print("\n============================================================")
    print("ALL 512x512 E2E MODULES VALIDATED SUCCESSFULLY!")
    print("============================================================")


if __name__ == "__main__":
    run_validation()
