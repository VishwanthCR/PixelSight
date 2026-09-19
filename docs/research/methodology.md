# PixelSight Scientific Methodology

## 1. Executive Summary & Problem Formulation

PixelSight addresses the challenge of **Super Resolution Mapping (SRM)** from medium-resolution satellite imagery (Sentinel-2 MSI 10 m) to high-resolution representations (~2.5 m equivalent). Rather than treating super-resolution as a pure computer-vision perceptual upscaling task, PixelSight establishes an **Uncertainty-Aware, Spectral-Spatial-Geospatial and Downstream-Task-Aware Satellite Super-Resolution Research Framework**.

```
Native Sentinel-2 10 m (B02, B03, B04, B08)
                  │
                  ├──► [Baseline] Bicubic Interpolation (2.5 m)
                  │
                  └──► [PixelSight] LDSR-S2 Latent Diffusion (4×, 100 steps)
                             │
                             ├──► Stochastic Diffusion Uncertainty (N=5 seeds)
                             ├──► Spatial Reliability / Reconstruction-Risk Map
                             ├──► Multi-Dimensional Fidelity (Spatial, Spectral, Geospatial)
                             └──► Downstream Task Evaluation (Land Cover Segmentation)
```

---

## 2. Core Model Architecture: LDSR-S2

PixelSight employs **LDSR-S2** (Latent Diffusion Super-Resolution for Sentinel-2):
- **Bands**: 4-band multi-spectral stack matching Sentinel-2 10 m spatial resolution:
  - Band 0: `B02` (Blue, ~490 nm)
  - Band 1: `B03` (Green, ~560 nm)
  - Band 2: `B04` (Red, ~665 nm)
  - Band 3: `B08` (Near-Infrared / NIR, ~842 nm)
- **Scale Factor**: 4× spatial resolution enhancement (10 m → 2.5 m ground sampling distance).
- **Inference Parameterization**:
  - Diffusion sampling steps: `100` steps (strictly maintained for research reproducibility).
  - Checkpoint: `opensr-ldsrs2_v1_0_0.ckpt`.
  - Conditioned on low-resolution latent representations via a pre-trained VAE encoder/decoder.

> [!IMPORTANT]
> **Strict NIR Band Preservation**: In multi-spectral remote sensing, NIR (`B08`) carries critical physical reflectance information about vegetation cellular structure and biomass. PixelSight **never** synthesizes NIR from RGB channels (`(R+G+B)/3`). Genuine Sentinel-2 `B08` data is strictly ingested and processed.

---

## 3. Stochastic Diffusion Uncertainty Estimation

### 3.1 Mechanism

Standard deep-learning super-resolution models produce deterministic point estimates. Diffusion models, however, are generative processes driven by stochastic differential equations or Markovian reverse sampling steps:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t, c) \right) + \sigma_t z$$

where $z \sim \mathcal{N}(0, I)$ is stochastic Gaussian noise injected at each step.

PixelSight estimates **stochastic diffusion variation uncertainty** by running LDSR-S2 inference $N$ times (default $N=5$) with different random initial seeds $\{s_1, s_2, \dots, s_N\}$ while holding the conditioning input $c$ fixed:

$$\mu_{\text{SR}}(i, j, b) = \frac{1}{N} \sum_{k=1}^{N} x_{\text{SR}}^{(k)}(i, j, b)$$

$$U(i, j) = \frac{1}{B} \sum_{b=1}^{B} \sqrt{\frac{1}{N-1} \sum_{k=1}^{N} \left( x_{\text{SR}}^{(k)}(i, j, b) - \mu_{\text{SR}}(i, j, b) \right)^2}$$

### 3.2 Scientific Distinction: Stochastic Variation vs MC-Dropout

PixelSight maintains rigorous scientific terminology:
- **What it is**: Empirical stochastic variation across diffusion generative trajectories. High values indicate spatial features whose fine-scale reconstruction is sensitive to the random noise path.
- **What it is NOT**:
  - It is **not** Monte Carlo Dropout (no dropout masks are sampled at test time).
  - It is **not** calibrated Bayesian posterior predictive uncertainty.
  - It does **not** imply that pixels are "hallucinated" or "fabricated".

---

## 4. Spatial Reconstruction Reliability Mapping

To translate raw pixel-wise uncertainty maps into actionable insights for remote sensing analysts, PixelSight defines a **Spatial Reliability / Reconstruction-Risk Map**:

$$R(i, j) = \begin{cases}
\text{HIGH\_RELIABILITY} (0) & \text{if } U(i, j) < \tau_{\text{low}} \\
\text{MEDIUM\_RELIABILITY} (1) & \text{if } \tau_{\text{low}} \le U(i, j) \le \tau_{\text{high}} \\
\text{LOW\_RELIABILITY} (2) & \text{if } U(i, j) > \tau_{\text{high}}
\end{cases}$$

- **Thresholds**: $\tau_{\text{low}}$ and $\tau_{\text{high}}$ default to the 33rd ($p_{33}$) and 67th ($p_{67}$) percentiles of empirical uncertainty.
- **Interpretation**: Regions labeled `LOW_RELIABILITY` (reconstruction-risk zones) exhibit high diffusion variance—frequently corresponding to complex structural transitions, building edges, and sub-pixel texture boundaries. Users are advised to exercise caution when basing quantitative decisions on these zones.

---

## 5. Multi-Fidelity Evaluation Framework

PixelSight evaluates super-resolution across three complementary dimensions:

### 5.1 Spatial Fidelity
- **Gradient Energy**: Measures edge sharpness and high-frequency structural content:
  $$E_{\text{grad}} = \frac{1}{B} \sum_{b=1}^{B} \frac{1}{HW} \sum_{i, j} \left( (\nabla_x I_b(i, j))^2 + (\nabla_y I_b(i, j))^2 \right)$$
- **Reference-Gated Metrics**: PSNR, SSIM, MAE, and RMSE against a high-resolution reference are computed **only** when a genuine independent HR image is present. When absent, metrics are reported as `null / N/A` rather than fabricating synthetic scores.

### 5.2 Spectral Fidelity
- **Normalized Difference Vegetation Index (NDVI)**:
  $$\text{NDVI} = \frac{\text{B08} - \text{B04}}{\text{B08} + \text{B04}}$$
- **Native 10 m NDVI as Reference Baseline**: Native 10 m NDVI is used as the physical baseline for inter-method consistency diagnostics (MAE and RMSE vs native).
- **Spectral Angle Mapper (SAM)**: Measures angular deviation between multi-spectral vectors:
  $$\text{SAM}(x, y) = \arccos \left( \frac{x \cdot y}{\|x\|_2 \|y\|_2} \right)$$
  (HR-reference gated).

### 5.3 Geospatial Integrity
Verifies complete preservation of:
- Coordinate Reference System (CRS, e.g., EPSG:32643)
- Affine transformation matrix (origin and pixel spacing: 10.0 m input → 2.5 m output)
- Spatial bounding box extent
- Channel order and nodata consistency

---

## 6. Downstream Task Evaluation: Land Cover Segmentation

To evaluate whether super-resolved imagery benefits real-world remote sensing tasks, PixelSight assesses downstream semantic segmentation using European Space Agency (ESA) WorldCover classes (Tree, Shrubland, Grassland, Cropland, Built-up, Bare, Water):
- **Metrics**: Per-class Intersection over Union (IoU), F1/Dice score, precision, recall, and overall pixel accuracy.
- **Reporting Integrity**: Super-resolution results are benchmarked directly against native observed inputs, preserving negative and positive outcomes without bias.

---

## 7. Super-Resolution Mapping (SRM) Sub-Pixel Abundance Conservation

In traditional geospatial science, **Super-Resolution Mapping (SRM)** (Atkinson 1997, Verhoeye & De Wulf 2002) is defined as sub-pixel land-cover allocation where coarse mixed pixels are decomposed into fine sub-pixels under the physical constraint:

$$\sum_{j \in \text{subpixels}} A_{j, c} = F_c \times A_{\text{coarse}}$$

PixelSight validates whether deep diffusion SR satisfies physical conservation laws or produces unphysical generative hallucinations:
- **Sub-Pixel Fraction Error (SPFE)**: Mean absolute difference between coarse spectral unmixing fractions and 4×4 sub-pixel classified abundance:
  $$\text{SPFE}_c = \frac{1}{M} \sum_{i=1}^{M} \left| \hat{f}_{i, c}^{\text{SR}} - f_{i, c}^{\text{coarse}} \right|$$
- **Area Conservation Index (ACI)**: Ratio of total sub-pixel land-cover area to unmixed coarse area.
- **Hallucinated Class Rate (HCR)**: Frequency with which the diffusion model generates class $c$ in sub-pixels when the native 10 m pixel contains $f_{i, c}^{\text{coarse}} < 0.05$.
- **Sub-Pixel Preservation Score (SPS)**: Composite index $\in [0, 1]$ summarizing physical adherence to native spectral endmember mixtures.

---

## 8. Frequency-Domain Modulation Transfer Function (MTF) & Fourier Ring Correlation (FRC)

In optical remote sensing, a 2.5 m pixel grid does **not** guarantee 2.5 m optical resolving power. PixelSight introduces rigorous frequency-domain diagnostics:
- **Radial Power Spectral Density (RPSD)**: Azimuthally integrated 2D Fourier power spectrum $P(f)$ to analyze high-frequency energy distribution and spectral roll-off slope ($\alpha$).
- **Modulation Transfer Function (MTF50)**: Spatial frequency (cycles/pixel) where optical contrast drops to 50% of the low-frequency plateau.
- **Fourier Ring Correlation (FRC)**: Normalized cross-correlation across frequency shells computed between independent stochastic diffusion realizations ($N=5$ seeds):
  $$\text{FRC}(r) = \frac{\sum_{r_i \in r} F_1(r_i) F_2^*(r_i)}{\sqrt{\sum_{r_i \in r} |F_1(r_i)|^2 \sum_{r_i \in r} |F_2(r_i)|^2}}$$
  The intersection with the $\frac{1}{2}$-bit (0.143) threshold determines the true effective physical ground sampling distance ($\text{GSD}_{\text{eff}} = \frac{1}{2 f_{\text{cutoff}}}$).
- **High-Frequency Hallucination Index (HFHI)**: Ratio of high-frequency power exceeding the native sensor's optical modulation envelope relative to smoothed bicubic baselines.

---

## 9. Uncertainty-Gated Hybrid Fusion for Decision Support

Generative diffusion models provide crisp visual details but can hallucinate boundaries in high-variance regions. PixelSight implements an **Uncertainty-Gated Hybrid Fusion Engine**:

$$I_{\text{hybrid}}(x, y) = w(x, y) I_{\text{SR}}(x, y) + \left(1 - w(x, y)\right) I_{\text{native\_up}}(x, y)$$

where $w(x, y) = \frac{1}{1 + \exp\left(\frac{U(x, y) - \tau}{s}\right)}$.
- **High-Confidence Regions ($U < \tau$)**: Preserves 2.5 m super-resolved micro-boundaries for precision agricultural parceling and urban infrastructure mapping.
- **High-Uncertainty Regions ($U \ge \tau$)**: Smooths back to the radiometrically conserved native observation, suppressing false-positive disaster damage claims and unmixing artifacts.
- **Decision Safety Score (DSS)**: Quantifies risk-controlled utility for operational Earth Observation deployments.

---

## 10. Hierarchical Reference Benchmark Subsystem

When evaluating super-resolution, PixelSight prioritizes references hierarchically:
1. **`REAL_HR_REFERENCE`**: Genuine independent high-resolution satellite imagery (e.g. WorldView, Pleiades, NAIP) with validated geographic co-registration.
2. **`EXTERNAL_HR_REFERENCE`**: High-resolution imagery requiring CRS reprojection, bounding-box alignment, and nearest/bilinear resampling.
3. **`SYNTHETIC_HR_REFERENCE`**: Scientifically controlled synthetic degradation benchmark modeling Gaussian Point-Spread Function (PSF) blur, 4× decimation, and calibrated sensor noise on known ground-truth targets.
4. **`CONSISTENCY_DIAGNOSTIC`**: Strict self-referential diagnostics when no HR reference exists; reference-based metrics (PSNR, SSIM, SAM) are reported strictly as `N/A`.

