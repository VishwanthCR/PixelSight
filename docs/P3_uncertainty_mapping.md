# PixelSight P3 — Uncertainty Mapping Guide

## 1. Purpose of P3

P3 is the **uncertainty-mapping stage** of PixelSight.

The current PixelSight pipeline has two important stages that P3 must build on:

```text
Plan 1
  Controlled 2× SR benchmark
  LR 32×32×4 → SR 64×64×4

        ↓

Plan 2
  Real Sentinel-2 L2A multispectral SR
  10 m B02/B03/B04/B08 → approximately 2.5 m
  using pretrained LDSR-S2

        ↓

P3
  Uncertainty estimation / mapping
  SR output → uncertainty map
```

The goal is **not** to produce another SR model. The goal is to quantify and visualize **where the LDSR-S2 prediction is reliable and where it is uncertain**.

P3 should therefore consume the existing SR predictions and produce spatially aligned uncertainty products plus figures that can later be used alongside the SR results and downstream remote-sensing analysis.

---

## 2. Read This Before Starting

The repository is organized as a research codebase with separate areas for source code, scripts, experiments, documentation, and results. Large local datasets and model checkpoints are intentionally excluded from GitHub. The current repository root contains `configs/`, `data/`, `docs/`, `experiments/`, `results/`, `scripts/`, and `src/`. citehttps://github.com/VishwanthCR/PixelSight

Plan 2 is a real-world 4× multispectral super-resolution experiment using Sentinel-2 Level-2A data and four 10 m bands:

- B02 — Blue
- B03 — Green
- B04 — Red
- B08 — Near Infrared

The intended output resolution is approximately **2.5 m**. citehttps://github.com/VishwanthCR/PixelSight/blob/main/docs/plan2_experiment_design.md

### Critical ground-truth rule

Sentinel-2 does **not** provide native 2.5 m imagery. Therefore, a bicubic 2.5 m upsampled Sentinel-2 image must **not** be treated as genuine high-resolution ground truth. The existing Plan 2 experiment design explicitly separates controlled evaluation from evaluation against a genuine external high-resolution reference dataset. citehttps://github.com/VishwanthCR/PixelSight/blob/main/docs/plan2_experiment_design.md

This constraint is important for P3 too: uncertainty maps should describe **model confidence / prediction variability**, not be presented as a direct measurement of pixel-wise SR error against a nonexistent native 2.5 m truth.

---

## 3. What P3 Should Answer

The uncertainty component should help answer questions such as:

1. **Where is LDSR-S2 confident?**
2. **Where is LDSR-S2 uncertain?**
3. **Are uncertainty patterns spatially meaningful?**
4. **Do edges, boundaries, clouds, masked areas, or spectrally unusual regions show higher uncertainty?**
5. **Does uncertainty differ across B02/B03/B04/B08?**
6. **Does uncertainty correlate with observable reconstruction instability or disagreement among stochastic SR samples?**

The final product should make uncertainty interpretable as a spatial map, not just as a single number.

---

## 4. Recommended P3 Strategy

Because LDSR-S2 is a diffusion-based SR model, the most useful starting point is **sampling-based uncertainty estimation**.

Conceptually:

```text
Same 10 m input
      │
      ├── stochastic LDSR sample 1 ──┐
      ├── stochastic LDSR sample 2 ──┤
      ├── stochastic LDSR sample 3 ──┤
      ├── ...                        ├──→ mean SR
      └── stochastic LDSR sample N ──┘
                                      │
                                      └──→ pixel-wise variance / uncertainty
```

For each spatial location and spectral band, collect multiple stochastic predictions.

Let the predictions be:

\[
Y_1, Y_2, \ldots, Y_N
\]

Then the pixel-wise predictive mean is:

\[
\mu = \frac{1}{N}\sum_{i=1}^{N}Y_i
\]

and the predictive variance is:

\[
\sigma^2 = \frac{1}{N-1}\sum_{i=1}^{N}(Y_i-\mu)^2
\]

A simple uncertainty map can use:

- per-band standard deviation
- mean standard deviation across the four bands
- band-weighted uncertainty
- optionally a normalized uncertainty score

Do **not** start by inventing a complex uncertainty formulation. First establish a correct stochastic baseline.

---

## 5. Why Multiple Samples Matter

A single deterministic LDSR prediction gives only one SR image.

A single prediction cannot directly reveal how sensitive the output is to the model's stochastic generation process.

Multiple stochastic predictions let us measure:

```text
high agreement among samples
        → lower predictive uncertainty

large disagreement among samples
        → higher predictive uncertainty
```

This is particularly appropriate for a diffusion-based model because repeated sampling can expose generation variability.

P3 should verify that sampling is actually stochastic before interpreting variance as uncertainty. If repeated runs are identical, first investigate seeds, sampler behavior, inference settings, and model implementation.

---

## 6. Start With a Small Experiment

Do **not** immediately run uncertainty estimation over the entire Sentinel-2 dataset.

Start with the existing Plan 2 inference samples already available locally.

A practical first experiment is:

- 1–3 existing Sentinel-2 test/sample tiles
- the existing 4 bands
- a small number of stochastic runs first, e.g. 8–16 samples
- store all predictions temporarily
- calculate mean and variance
- generate visual maps

Once the pipeline is verified, increase the number of samples if the uncertainty estimate is unstable.

This avoids unnecessary GPU usage and lets the team validate the methodology before scaling up.

---

## 7. P3 Inputs

The implementation should be able to work from:

### Required

- original 10 m Sentinel-2 four-band input
- corresponding LDSR-S2 inference configuration
- pretrained LDSR-S2 checkpoint, kept outside Git
- GPU/CPU device

### Existing Plan 2 conventions

The four channels are ordered:

```text
[B02, B03, B04, B08]
```

The model's operational input is a 128×128 four-channel tensor and the model outputs a 512×512 four-channel SR tensor for 4× SR.

Keep this channel order unchanged throughout P3.

---

## 8. Suggested Output Products

P3 should eventually generate at least these outputs for every evaluated sample.

### A. Mean SR

```text
mean_sr.tif
```

The pixel-wise mean across stochastic LDSR predictions.

### B. Per-band uncertainty

```text
uncertainty_B02.tif
uncertainty_B03.tif
uncertainty_B04.tif
uncertainty_B08.tif
```

or a single four-band uncertainty GeoTIFF.

### C. Aggregate uncertainty

```text
uncertainty_mean.tif
```

For example, the mean per-pixel standard deviation across the four bands.

### D. Visualization

At minimum:

```text
uncertainty_map_rgb.png
uncertainty_map_false_color.png
uncertainty_band_comparison.png
mean_vs_uncertainty.png
```

The exact filenames may change as implementation evolves.

---

## 9. Recommended Repository Location

Keep P3 code separate from the existing Plan 2 inference scripts.

Recommended structure:

```text
scripts/
└── plan3/
    ├── run_uncertainty.py
    ├── compute_uncertainty.py
    ├── visualize_uncertainty.py
    └── validate_uncertainty.py

src/
└── pixelsight/
    └── uncertainty/
        ├── __init__.py
        ├── sampling.py
        ├── metrics.py
        └── visualization.py

results/
└── plan3/
    ├── samples/
    ├── maps/
    ├── figures/
    └── tables/

docs/
└── P3_uncertainty_mapping.md
```

Large imagery, checkpoints, raw stochastic samples, and temporary inference files should remain outside Git according to the existing repository data-management approach.

---

## 10. Spatial Alignment Requirements

This is one of the most important technical requirements.

Every uncertainty product must remain spatially aligned with the SR output.

Preserve:

- CRS
- affine transform
- pixel size
- raster width/height
- bounds
- band order

For 4× SR:

```text
10 m input
↓
2.5 m SR
```

The output transform must reflect the 2.5 m pixel size while retaining the same geographic bounds for a correctly tiled/stiched output.

Do not create uncertainty PNGs first and then attempt to reconstruct geospatial alignment later. Compute uncertainty in the native SR raster coordinates, then export visualizations from that aligned product.

---

## 11. Suggested Uncertainty Quantities

### 11.1 Per-band standard deviation

For band \(b\):

\[
U_b(x,y)=\sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(Y_{i,b}(x,y)-\mu_b(x,y))^2}
\]

This is the primary quantity to implement first.

### 11.2 Aggregate uncertainty

A simple aggregate map can be:

\[
U_{mean}(x,y)=\frac{1}{4}\sum_b U_b(x,y)
\]

This provides a single uncertainty map while preserving the per-band maps for analysis.

### 11.3 Relative uncertainty

If the absolute scale makes interpretation difficult, investigate a relative form such as:

\[
U_{rel}(x,y)=\frac{\sigma(x,y)}{|\mu(x,y)|+\epsilon}
\]

Use this only after validating the absolute uncertainty map because division by low-intensity pixels can exaggerate noise.

---

## 12. Quality-Control Checks

Every P3 experiment should perform these checks automatically.

### Shape check

All stochastic outputs must have identical shape:

```text
(4, H, W)
```

or the equivalent batch-first form.

### Finite-value check

Verify there are no NaNs or infinities.

### Range check

The reflectance-like SR values should remain within the expected model/data range. Do not silently clip values before uncertainty calculation because clipping can artificially reduce measured variance.

### Deterministic-vs-stochastic check

Run the same input multiple times.

If every result is exactly identical, investigate why before declaring uncertainty to be zero.

### Non-zero uncertainty check

On a normal textured sample, the aggregate uncertainty map should not be identically zero.

### Alignment check

Verify that the uncertainty raster has the same spatial metadata as the corresponding SR product.

---

## 13. Visualization Guidance

The uncertainty visualization should be easy to interpret.

Recommended figure layout:

```text
Original 10m RGB
        │
        ├── LDSR mean SR RGB
        │
        └── uncertainty map
```

For multispectral analysis, also include:

```text
B02 uncertainty
B03 uncertainty
B04 uncertainty
B08 uncertainty
```

Use percentile-based display normalization for uncertainty when needed, but preserve the original numerical raster for quantitative analysis.

Do not confuse display normalization with actual uncertainty values.

---

## 14. Important Distinction: Uncertainty vs Error

P3 must keep these concepts separate.

### Error

Requires a reference/target:

```text
prediction − ground truth
```

Examples:

- MAE
- RMSE
- PSNR
- SSIM
- SAM

### Uncertainty

Describes prediction variability / confidence without requiring a true high-resolution target.

For P3:

```text
multiple stochastic predictions
        ↓
variability
        ↓
uncertainty estimate
```

A high uncertainty value does **not automatically mean the SR prediction is wrong**.

Likewise, a low uncertainty value does **not prove correctness**.

This distinction should be preserved in the paper, README, figures, and presentations.

---

## 15. Relationship to Plan 2 Evaluation

Plan 2's current experiment design explicitly defines a controlled benchmark:

```text
Native 10 m reference
        ↓
Controlled degradation
        ↓
Lower-resolution LR
        ↓
SR model
        ↓
Reconstructed 10 m
        ↓
Compare with native 10 m reference
```

This provides a route for quantitative reconstruction metrics at a known reference resolution. citehttps://github.com/VishwanthCR/PixelSight/blob/main/docs/plan2_experiment_design.md

P3 uncertainty should be reported **alongside** these evaluation results, not used as a replacement for them.

A useful eventual analysis is:

```text
SR quality metrics
        +
uncertainty map
        ↓
interpretability of where the model is reliable / unstable
```

---

## 16. Recommended Experiments

### Experiment P3-A — Sampling sanity test

Run one input repeatedly and verify that outputs differ when stochastic sampling is enabled.

Deliverable:

```text
sample_01_run_01.tif
sample_01_run_02.tif
...
```

### Experiment P3-B — Initial uncertainty map

Use a small number of stochastic predictions and calculate per-band standard deviation.

Deliverable:

```text
mean_sr.tif
uncertainty_B02.tif
uncertainty_B03.tif
uncertainty_B04.tif
uncertainty_B08.tif
uncertainty_mean.tif
```

### Experiment P3-C — Sample-count sensitivity

Compare uncertainty maps using increasing numbers of stochastic samples, for example:

```text
N = 4
N = 8
N = 16
N = 32
```

Check whether the uncertainty statistics stabilize.

### Experiment P3-D — Spatial interpretation

Investigate whether high uncertainty occurs around:

- strong boundaries
- heterogeneous land cover
- unusual spectra
- difficult textures
- image edges / tile boundaries
- potentially cloud/shadow affected areas

Do not assume these relationships beforehand; validate them from the actual maps.

---

## 17. What Not to Do

Do **not**:

- treat bicubic 2.5 m Sentinel-2 imagery as real ground truth
- call uncertainty a direct accuracy/error map
- mix B02/B03/B04/B08 channel ordering
- clip stochastic predictions before calculating variance
- destroy geospatial metadata when saving uncertainty products
- immediately run the entire dataset before validating one sample
- commit raw Sentinel-2 imagery or large model checkpoints to GitHub
- compare uncertainty magnitudes across experiments without recording the sampling configuration

---

## 18. What P3 Should Commit

The final P3 contribution should primarily contain:

```text
scripts/plan3/
src/pixelsight/uncertainty/
docs/P3_uncertainty_mapping.md
results/plan3/  (only appropriately sized tracked research figures/tables)
```

A good commit sequence is:

```text
1. Add P3 uncertainty module
2. Add sampling experiment
3. Add uncertainty computation
4. Add visualization
5. Add validation / QC
6. Add documented experiment results
```

Do not commit:

- raw Sentinel-2 datasets
- large `.tif` collections
- LDSR checkpoints
- huge arrays of every stochastic prediction
- temporary caches

---

## 19. P3 Deliverable Definition

A successful first version of P3 should make it possible to run something conceptually similar to:

```bash
python scripts/plan3/run_uncertainty.py \
    --input <10m_sentinel2_tile> \
    --samples 16 \
    --sampling-steps 100
```

and obtain:

```text
results/plan3/<sample>/
├── mean_sr.tif
├── uncertainty_B02.tif
├── uncertainty_B03.tif
├── uncertainty_B04.tif
├── uncertainty_B08.tif
├── uncertainty_mean.tif
└── figures/
    ├── rgb_comparison.png
    ├── uncertainty_map.png
    └── band_uncertainty.png
```

The exact CLI and filenames can change, but the functionality should remain equivalent.

---

## 20. Definition of Done

P3 is ready for integration when:

- [ ] repeated stochastic LDSR runs are confirmed to produce meaningful variation
- [ ] per-band uncertainty is computed correctly
- [ ] aggregate uncertainty is computed correctly
- [ ] uncertainty rasters are geospatially aligned with the SR output
- [ ] visualizations are generated from the numerical uncertainty products
- [ ] a small-sample experiment has been completed before large-scale execution
- [ ] uncertainty is clearly distinguished from prediction error
- [ ] the methodology and sampling configuration are documented
- [ ] no raw datasets/checkpoints are committed
- [ ] results can be reproduced from documented commands/configuration

---

## 21. Current Context for the Team

The current PixelSight repository already contains Plan 2 documentation describing the Sentinel-2 four-band setup and the ground-truth limitation. citehttps://github.com/VishwanthCR/PixelSight/blob/main/docs/plan2_experiment_design.md

P3 should therefore **extend the existing Plan 2 pipeline rather than create a separate data/model pipeline**.

The key architectural relationship is:

```text
Plan 2 data + LDSR-S2 inference
                │
                ▼
        Existing SR products
                │
                ▼
       P3 stochastic sampling
                │
        ┌───────┴────────┐
        ▼                ▼
   mean SR         uncertainty maps
        │                │
        └───────┬────────┘
                ▼
       joint SR + uncertainty
              analysis
```

Start small, verify the stochastic behavior, preserve geospatial metadata, and only then scale the uncertainty analysis.