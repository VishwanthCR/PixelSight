# urban_satellite_super_resolution

This framework is integrated into PixelSight. It provides a research baseline for uncertainty-aware urban analysis from georeferenced Sentinel-2 RGB+NIR imagery.

## Scientific contract

The 4x output is a model-derived estimate at approximately 2.5 m target pixel spacing. It is not observed 2.5 m ground truth. Every inference writes a predictive uncertainty raster and confidence classes; low-confidence regions require review. PSNR and SSIM are descriptive image metrics and are not sufficient evidence of scientific validity.

## Dataset layout

```text
data/
  train/{lr,hr,labels}/
  validation/{lr,hr,labels}/
  test/{lr,hr,labels}/
```

Files pair by stem. LR and HR must have matching CRS, bounds, band order, and geographic alignment. Labels are categorical GeoTIFFs with `255` as ignore. Keep train, validation, and test scenes geographically separated.

## Commands

From the repository root with the virtual environment active:

```powershell
$env:PYTHONPATH='src;.'
python scripts/urban_satellite_super_resolution.py prepare-data --output data/synthetic
python scripts/urban_satellite_super_resolution.py infer --input data/synthetic/scene_lr.tif --output results/urban_demo --passes 20
python scripts/urban_satellite_super_resolution.py evaluate --prediction path/to/prediction.npy --target path/to/target.npy
```

Training expects paired files under `data/train` and uses `configs/urban_satellite_super_resolution.yaml`:

```powershell
python scripts/urban_satellite_super_resolution.py train --config configs/urban_satellite_super_resolution.yaml
```

The inference output directory contains `super_resolved.tif`, `uncertainty_std.tif`, `confidence_classes.tif`, five urban probability GeoTIFFs, an optional `urban_change_probability.tif`, `quicklook.png`, `report.json`, and an interpretable `report.md`.

## Dependencies and data

Install the project in editable mode with `pip install -e .`. The project does not download data automatically. Suitable sources include Sentinel-2 L2A products from Copernicus Data Space or equivalent research datasets, paired with a defensible high-resolution reference and urban labels. Record source, date, CRS, preprocessing, and geographic split in experiment metadata.
