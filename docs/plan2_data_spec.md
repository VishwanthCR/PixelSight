\# PixelSight Plan 2 — Sentinel-2 Data Specification



\## Objective



Build a real-world 4× multispectral super-resolution pipeline using

Sentinel-2 imagery.



\## Input Data



Use Sentinel-2 Level-2A surface reflectance imagery.



Required bands:



\- B02 — Blue

\- B03 — Green

\- B04 — Red

\- B08 — Near Infrared



All four bands have native 10 m spatial resolution.



\## Preprocessing



1\. Select B02, B03, B04, and B08.

2\. Convert the bands to a common spatial grid.

3\. Apply cloud / invalid-pixel masking.

4\. Normalize reflectance consistently.

5\. Generate spatially aligned multispectral tiles.

6\. Preserve geographic metadata for each tile.



\## Super-Resolution Task



Input:



\- 10 m resolution

\- 4 spectral channels



Target:



\- 4× spatial enlargement

\- approximately 2.5 m output resolution

\- same four spectral channels



Tile design:



\- LR: 128 × 128 × 4

\- HR: 512 × 512 × 4



\## Dataset Splitting



Splits must be spatially separated to avoid leakage.



Recommended:



\- Training regions

\- Validation regions

\- Final test regions



The final test regions must remain untouched during model development.



\## Evaluation



\### Super-Resolution Metrics



\- PSNR

\- SSIM

\- SAM



\### Downstream Evaluation



Where a suitable segmentation/classification task and labels are available:



\- IoU

\- F1

\- Dice



The downstream evaluation measures whether super-resolution improves

practical remote-sensing usefulness rather than only pixel-level similarity.



\## Reproducibility



Record:



\- Sentinel-2 product identifiers

\- acquisition dates

\- geographic regions

\- preprocessing parameters

\- cloud-mask method

\- normalization method

\- tile dimensions

\- train/validation/test regions

\- model checkpoint identifiers

\- random seeds



Raw Sentinel-2 datasets must remain outside Git.

