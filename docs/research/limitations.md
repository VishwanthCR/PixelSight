# PixelSight Scientific Limitations & Methodological Constraints

Scientific rigor requires complete transparency regarding experimental boundaries, assumptions, and negative findings. This document formally registers the known limitations of the PixelSight framework.

---

## 1. Ground Truth & HR Reference Availability

### 1.1 Absence of Paired Independent High-Resolution Imagery
- **Status**: No paired genuine high-resolution satellite imagery (such as NAIP 0.6 m or WorldView 0.3-0.5 m) is available in the current dataset split.
- **Scientific Impact**:
  - Full-reference image quality metrics (PSNR, SSIM, SAM, and reconstruction MAE/RMSE against HR truth) **cannot be legitimately computed**.
  - Conventional literature often circumvents this by using bicubic or nearest-neighbor upscaling of the low-resolution input as a pseudo-reference. PixelSight **strictly prohibits** this practice, as it artificially rewards models that smooth rather than resolve.
  - All HR-referenced fields in PixelSight benchmark tables report `N/A` or `null`.

---

## 2. Downstream Task Evaluation: Negative Finding Documentation

### 2.1 The Negative Result
In controlled experiments comparing native Sentinel-2 10 m input against LDSR-S2 2.5 m super-resolved imagery for U-Net land cover segmentation:
- **Native 10 m**: Pixel Accuracy = **75.98%**, Mean IoU = **0.3012**
- **LDSR-S2 2.5 m (Proxy)**: Pixel Accuracy = **45.11%**, Mean IoU = **0.1224**
- **Net Degradation**: -30.87 percentage points in accuracy, -17.88 percentage points in mIoU.

### 2.2 Root Cause Analysis
1. **Proxy Label Resolution Mismatch**:
   - The reference segmentation labels originate from ESA WorldCover 10 m data, which are upscaled 4× to 2.5 m via nearest-neighbor replication.
   - These proxy labels contain stepped, blocky 10 m boundaries rather than genuine sub-pixel land cover transitions.
   - When LDSR-S2 generates plausible high-frequency boundaries, they diverge from the artificial blocky proxy labels, resulting in severe metric penalties (false positives/false negatives along edges).
2. **Classifier Domain Shift**:
   - The downstream U-Net classifier was trained on native Sentinel-2 10 m spectral distributions and spatial smoothness.
   - Latent diffusion introduces fine-scale generative textures and high-frequency patterns that fall outside the classifier's training manifold.
3. **Generative Artifacts**:
   - Stochastic diffusion can produce micro-patterns that do not correspond to semantic land cover classes known to the classifier.

> [!NOTE]
> This negative finding is a **valid and valuable scientific result**. It demonstrates that super-resolution cannot simply be applied out-of-the-box to downstream tasks without either fine-tuning the downstream model on super-resolved data or evaluating against genuine sub-pixel ground truth labels.

---

## 3. Nature of the Uncertainty Metric

### 3.1 Empirical Stochastic Variation vs. Bayesian Epistemic Uncertainty
- **Mechanism**: The uncertainty metric is computed as the standard deviation across $N=5$ independent LDSR-S2 diffusion trajectories driven by different pseudorandom seeds.
- **Limitation**:
  - This captures **generative trajectory sensitivity** to initial Gaussian noise.
  - It does **not** quantify model weight uncertainty (epistemic uncertainty), nor does it guarantee calibrated Bayesian coverage.
  - A low uncertainty value indicates high consensus among diffusion trajectories, but does not guarantee correctness if the underlying model is systematically biased.

---

## 4. Spectral Index Diagnostics vs. Reconstruction Accuracy

### 4.1 Native NDVI as a Consistency Baseline
- When evaluating NDVI fidelity without an HR reference, native 10 m NDVI is used as a baseline.
- Bicubic interpolation achieves lower MAE against native NDVI ($0.0003$) compared to LDSR-S2 ($0.0141$).
- **Explanation**: Bicubic interpolation performs local linear smoothing, which naturally preserves the smoothed low-frequency NDVI distribution of native 10 m pixels. LDSR-S2 reconstructs sharp localized variations in reflectance, slightly shifting localized pixel ratios away from the 10 m spatial average. This is an expected mathematical consequence of super-resolution and is documented as a diagnostic rather than an error.

---

## 5. Band Ingestion & NIR Restrictions

- Synthesizing NIR from visible bands ($\text{NIR} \approx (R+G+B)/3$) is **physically invalid** in remote sensing.
- The scientific evaluation pipeline strictly accepts 4-band stacks where Band 3 is genuine Sentinel-2 B08. Any RGB-only inference must be treated strictly as an uncalibrated visualization.
