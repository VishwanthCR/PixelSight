# Data

This directory contains documentation and dataset access instructions only.

## Plan 1

Paired multispectral patches:

- LR: 32×32×4
- HR: 64×64×4
- Region 1: training
- Region 2: validation
- Region 3: untouched final test

The local dataset is intentionally excluded from Git.

## Plan 2

The planned real-world pipeline will use Sentinel-2 L2A imagery with B04/B03/B02/B08 at 10 m resolution, subject to final dataset and preprocessing verification.

## Reproducibility

Dataset acquisition, preprocessing, splits and quality-control decisions should be documented under `docs/` and experiment configurations under `configs/`.
