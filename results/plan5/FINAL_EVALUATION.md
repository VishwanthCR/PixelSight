# PixelSight — Final Quantitative Evaluation

## P2 — Super-resolution

- Input: 10 m Sentinel-2
- Output: approximately 2.5 m super-resolved representation
- Scale factor: 4x
- Spectral bands: B02, B03, B04, B08
- Diffusion sampling: 100 steps
- Demonstrated output: 512x512 -> 2048x2048

## P3 — Uncertainty

The diffusion model was sampled five times using different random seeds.

The pixel-wise mean provides the final SR estimate and the pixel-wise
standard deviation provides the uncertainty estimate.

### Uncertainty statistics

- Mean: 0.003319
- Median: 0.002963
- 90th percentile: 0.005219
- Maximum: 0.048180

High-uncertainty pixels had an error rate of approximately
70.62%,
compared with 53.15%
for low-uncertainty pixels.

This is a useful indication that model uncertainty contains information
about potentially unreliable downstream predictions.

## P4 — Urban analysis

Gradient energy:

- Native 10 m: 0.007364
- Bicubic 2.5 m: 0.001941
- LDSR 2.5 m: 0.003489

LDSR has approximately 1.80x the
gradient energy of bicubic upsampling.

This supports the conclusion that LDSR produces substantially more
spatial detail than bicubic interpolation in the evaluated scene.

This does NOT by itself establish improved building-detection accuracy.

## P4 — Crop / vegetation analysis

NDVI consistency:

- Bicubic MAE: 0.00765141
- LDSR MAE: 0.00734301
- MAE improvement: 4.03%

- Bicubic RMSE: 0.01130433
- LDSR RMSE: 0.00976343
- RMSE improvement: 13.63%

This supports better preservation of vegetation-related spectral
consistency relative to bicubic in the evaluated scene.

It does NOT establish crop classification, yield prediction, or crop
monitoring accuracy.

## P4 — Segmentation

### Native 10 m

- Pixel accuracy: 0.759766
- mIoU: 0.301203
- Dice: 0.396126
- Precision: 0.611905
- Recall: 0.439686

### LDSR 2.5 m

- Pixel accuracy: 0.451069
- mIoU: 0.122368
- Dice: 0.175830
- Precision: 0.311810
- Recall: 0.443772

The direct-transfer segmentation experiment did not improve with LDSR.
This should be reported as a limitation rather than hidden.

The main methodological issue is that the U-Net was trained on 10 m
imagery and then directly applied to 2.5 m SR imagery, while the
WorldCover labels are 10 m-scale proxy labels.

## Disaster management

No disaster-specific reference dataset was found in the current local
dataset tree.

Therefore PixelSight should describe disaster assessment as a target
application rather than claim an experimentally validated disaster
result.

## Overall conclusion

PixelSight demonstrates:

1. 10 m -> approximately 2.5 m multispectral super-resolution.
2. Explicit per-pixel uncertainty estimation.
3. Greater spatial-detail energy than bicubic interpolation.
4. Better NDVI consistency than bicubic on the evaluated test scene.
5. A measurable relationship between uncertainty and downstream error.

The project does not yet establish universal downstream task improvement
or true 2.5 m ground-truth reconstruction.
