# PixelSight

PixelSight is a multispectral satellite-imagery super-resolution and geospatial-analysis framework.

It combines research pipelines for Sentinel-2 imagery with a FastAPI backend and React frontend.

> **Scientific note:** PixelSight produces a model-derived 4× super-resolved representation (~2.5 m equivalent) from compatible 10 m Sentinel-2 RGB-NIR imagery. It must not be described as observed or ground-truth 2.5 m imagery.

---

## What PixelSight Does

The current application provides an end-to-end workflow for compatible satellite imagery:

```text
Satellite Input
      ↓
Input Inspection
      ↓
Preprocessing
      ↓
LDSR-S2 4× Super Resolution
      ↓
Uncertainty Estimation
      ↓
Urban Land-Cover Analysis
      ↓
Reports / GeoTIFF Outputs
      ↓
React Results Dashboard
