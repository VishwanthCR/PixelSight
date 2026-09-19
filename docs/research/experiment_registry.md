# PixelSight Scientific Experiment Registry & Provenance

This registry records all experimental runs, parameters, datasets, and generated artifacts to ensure complete reproducibility.

---

## 1. Global Benchmark Configuration

| Parameter | Value | Scientific Description |
|-----------|-------|------------------------|
| **Base Model** | `LDSR-S2` | Latent Diffusion Super-Resolution adapted for Sentinel-2 MSI |
| **Model Weights** | `opensr-ldsrs2_v1_0_0.ckpt` | Official pre-trained checkpoint (1.13 GB) |
| **Scale Factor** | `4×` | 10.0 m Ground Sampling Distance (GSD) → 2.5 m GSD |
| **Sampling Steps** | `100` | DDPM/LDM reverse diffusion steps per sample |
| **Input Bands** | `B02, B03, B04, B08` | Blue, Green, Red, NIR (Sentinel-2 10 m native stack) |
| **Input Tile Dimensions** | `(128, 128, 4)` | Equivalent to 1.28 km × 1.28 km spatial footprint |
| **Output SR Dimensions**| `(512, 512, 4)` | 4× super-resolved spatial footprint |
| **Uncertainty Samples** | `N = 5` | Random seeds: `0, 1, 2, 3, 4` |
| **Coordinate Reference System** | `EPSG:32643` | WGS 84 / UTM Zone 43N |

---

## 2. Executed Experiments

### Exp-1: Stochastic Uncertainty & Spatial Reliability Analysis
- **Script**: [`experiments/uncertainty/run_uncertainty_analysis.py`](file:///c:/Users/Karth/Downloads/PixelSight/experiments/uncertainty/run_uncertainty_analysis.py)
- **Primary Inputs**:
  - Uncertainty map: `results/plan3/uncertainty_map.npy`
  - Mean SR output: `results/plan3/mean_sr.npy`
  - Downstream error map: `results/plan4/error_map.npy`
- **Generated Outputs**:
  - `results/uncertainty/uncertainty_statistics.json`
  - `results/uncertainty/uncertainty_vs_error.json`
  - `results/reliability/reliability_summary.json`
  - `results/reliability/reliability_map.npy`
- **Key Metrics**:
  - Mean uncertainty: `0.003319` (std: `0.001754`)
  - Low reliability region error rate: `63.61%` vs High reliability error rate: `46.87%` (+16.74 percentage points)
  - Pearson correlation ($U$ vs Error): `0.1481`

---

### Exp-2: Spectral Fidelity & NDVI Consistency Analysis
- **Script**: [`experiments/spectral/run_spectral_analysis.py`](file:///c:/Users/Karth/Downloads/PixelSight/experiments/spectral/run_spectral_analysis.py)
- **Primary Inputs**:
  - Native Sentinel-2 patch: `dataset/plan2/patches/train/patch_00000.npz`
  - Mean SR output: `results/plan3/mean_sr.npy`
- **Generated Outputs**:
  - `results/evaluations/spectral/spectral_fidelity.json`
  - `results/evaluations/spectral/spectral_comparison_table.csv`
  - `results/evaluations/spectral/spectral_comparison_table.json`
- **Key Metrics**:
  - Native 10 m NDVI: Mean = `0.1732`, Std = `0.0592`
  - Bicubic 2.5 m NDVI: Mean = `0.1731`, MAE vs Native = `0.0003`
  - LDSR-S2 2.5 m NDVI: Mean = `0.1732`, MAE vs Native = `0.0141`
  - SAM vs HR Reference: `N/A` (genuine HR reference unavailable)

---

### Exp-3: Downstream Task Evaluation (ESA WorldCover Segmentation)
- **Script**: [`experiments/downstream/run_downstream_summary.py`](file:///c:/Users/Karth/Downloads/PixelSight/experiments/downstream/run_downstream_summary.py)
- **Primary Inputs**:
  - Pre-computed comparison: `results/plan4/downstream_comparison.json`
  - Native prediction: `results/plan4/native_prediction.npy`
  - Native labels: `dataset/segmentation/patches/train/labels/r00000_c00000.npy`
  - SR prediction: `results/plan4/sr_prediction.npy`
  - SR proxy labels: `results/plan4/worldcover_native_512.npy`
- **Generated Outputs**:
  - `results/evaluations/downstream/downstream_summary.json`
  - `results/evaluations/downstream/downstream_comparison_table.csv`
  - `results/evaluations/downstream/downstream_comparison_table.json`
- **Key Findings**:
  - Native 10 m: Pixel Accuracy = `75.98%`, mIoU = `0.3012`
  - LDSR-S2 2.5 m (Proxy): Pixel Accuracy = `45.11%`, mIoU = `0.1224`
  - Documented negative result attributed to proxy label resolution mismatch and classifier domain shift.

---

### Exp-4: Controlled 3-Way Benchmark
- **Script**: [`experiments/native_vs_bicubic_vs_pixelsight/run_benchmark.py`](file:///c:/Users/Karth/Downloads/PixelSight/experiments/native_vs_bicubic_vs_pixelsight/run_benchmark.py)
- **Outputs Generated**:
  - `results/comparisons/benchmark_summary.json`
  - `results/comparisons/image_level_comparison_table.csv` & `.json`
  - `results/comparisons/downstream_comparison_table.csv` & `.json`
  - `results/comparisons/spectral_comparison_table.csv` & `.json`
  - `results/comparisons/uncertainty_breakdown_table.csv` & `.json`

---

### Exp-5: Master Pipeline & Novel Research Diagnostics (SRM Conservation, FRC/MTF, Decision Support)
- **Script**: [`experiments/master_pipeline/run_master_experiment.py`](file:///c:/Users/Karth/Downloads/PixelSight/experiments/master_pipeline/run_master_experiment.py)
- **Primary Inputs**:
  - Native Sentinel-2 patch: `dataset/plan2/patches/train/patch_00000.npz`
  - Mean SR output: `results/plan3/mean_sr.npy`
  - Uncertainty map: `results/plan3/uncertainty_map.npy`
  - Error map: `results/plan4/error_map.npy`
- **Generated Visual & Scientific Reports**:
  - Flagship 10-Panel Figure: `results/figures/master_research_figure.png` (4.6 MB)
  - Quintile Error Calibration Curve: `results/figures/quintile_calibration.png`
  - Master Scientific JSON Report: `results/reports/master_evaluation_report.json`
  - Master Scientific Markdown Report: `results/reports/master_evaluation_report.md`
  - Master Scientific HTML Report: `results/reports/master_evaluation_report.html`
- **Novel Research Findings**:
  - **SRM Sub-Pixel Abundance Preservation**: Sub-Pixel Fraction Error (SPFE) = `0.2509`, Sub-pixel Preservation Score (SPS) = `0.621`, Hallucinated Class Rate (HCR) = `0.01%`.
  - **Frequency-Domain Resolution (FRC/MTF)**: Effective Physical Resolving Limit = `2.50 m`, Radial Power Roll-off Slope = `-3.89`, High-Frequency Hallucination Index = `2108.91`.
  - **Uncertainty-Gated Hybrid Decision Support**: 75.0% confident high-resolution routing, 25.0% high-uncertainty fallback, 89.1% hallucination suppression rate, Decision Safety Score = `0.678`.

---


## 3. Benchmark Comparison Tables

### Table 1: Image-Level Fidelity

| Method | PSNR (dB) | SSIM | SAM (°) | MAE | RMSE |
|:-------|:----------|:-----|:--------|:----|:-----|
| **Bicubic 2.5 m** | N/A | N/A | N/A | N/A | N/A |
| **PixelSight LDSR-S2 (~2.5 m)** | N/A | N/A | N/A | N/A | N/A |
| *Note* | *N/A = metric requires genuine HR reference which is unavailable.* | | | | |

### Table 2: Downstream Task Performance

| Method | mIoU | mF1 / Dice | Accuracy | Precision | Recall |
|:-------|:-----|:-----------|:---------|:----------|:-------|
| **Native 10 m (observed)** | 0.3012 | 0.3961 | 0.7598 | 0.6119 | 0.4397 |
| **PixelSight LDSR-S2 (~2.5 m)** | 0.1224 | 0.1758 | 0.4511 | 0.3118 | 0.4438 |
| *Note* | *Labels are 10 m WorldCover proxy replicated to 2.5 m. The SR evaluation does not use genuine 2.5 m labels.* | | | | |

### Table 3: Spectral Consistency (NDVI)

| Method | NDVI MAE vs Native | NDVI RMSE vs Native | SAM (°) | NDVI Mean |
|:-------|:-------------------|:--------------------|:--------|:----------|
| **Bicubic 2.5 m** | 0.0003 | 0.0005 | N/A | 0.1731 |
| **PixelSight LDSR-S2 (~2.5 m)** | 0.0141 | 0.0198 | N/A | 0.1732 |
| *Note* | *NDVI comparisons use native 10 m NDVI as reference baseline (consistency diagnostic). SAM N/A — HR reference unavailable.* | | | |

### Table 4: Reconstruction Uncertainty & Reliability

| Uncertainty Level | Region Error Rate | Pearson Corr (unc-err) |
|:------------------|:------------------|:-----------------------|
| **Low** | 0.4687 | 0.1481 |
| **Medium** | 0.5422 | |
| **High** | 0.6361 | |
| *Mean Uncertainty* | *0.003319* | |
| *Median Uncertainty* | *0.002963* | |
| *P90 Uncertainty* | *0.005219* | |
| *Max Uncertainty* | *0.048180* | |
| *Note* | *Uncertainty = stochastic diffusion variation (std across 5 seeds). This is NOT MC-dropout or Bayesian predictive uncertainty.* | |
