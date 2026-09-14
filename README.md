# PixelSight

Multispectral super-resolution research project for improving spatial resolution while preserving spectral fidelity in remote-sensing imagery.

## Research Plans

### Plan 1 — 2× Super-Resolution Experiment

A controlled experiment using paired 32×32×4 low-resolution and 64×64×4 high-resolution patches.

- Region 1: training
- Region 2: validation
- Region 3: final test
- Bicubic baseline
- Lightweight 4-channel SwinIR
- Primary metrics: PSNR, SSIM, SAM
- Explicit handling and evaluation of empty/background patches

### Plan 2 — Real Sentinel-2 Super-Resolution

A real-world pipeline using Sentinel-2 L2A imagery with 10 m RGB-NIR bands and modern pretrained super-resolution methods. The pipeline will investigate 4× upscaling and downstream remote-sensing usefulness.

Downstream evaluation will include segmentation/classification metrics such as IoU and F1/Dice where appropriate.

## Repository Structure

```text
configs/       Experiment configurations
src/           Main PixelSight Python package
scripts/       Command-line experiment scripts
experiments/   Experiment-specific metadata and runs
results/       Metrics, figures and qualitative outputs
notebooks/     Exploratory and analysis notebooks
docs/          Methodology and research documentation
tests/         Unit tests
assets/        Architecture and presentation assets
data/          Dataset documentation (raw datasets are not committed)
```

## Data and Checkpoints

Raw datasets, generated datasets, model checkpoints and large experiment artifacts are intentionally excluded from Git. See `.gitignore` and `data/README.md`.

## Development Workflow

Use feature branches and pull requests. The intended workflow is:

```text
feature branch → Pull Request → review → develop → main
```

Do not commit secrets, raw datasets, or large model checkpoints.

## Status

Plan 1 baseline and initial SwinIR experiment completed. SwinIR v2 is planned to address empty-patch behavior using all training patches and a bicubic-residual design. Plan 2 is the next major research stage.
