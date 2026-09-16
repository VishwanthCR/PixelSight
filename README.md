# PixelSight

PixelSight is a multispectral satellite-imagery super-resolution and geospatial-analysis framework.

It combines research pipelines for Sentinel-2 imagery with a FastAPI backend and React frontend.

> Scientific note: PixelSight produces a model-derived 4x super-resolved representation (~2.5 m equivalent) from compatible 10 m Sentinel-2 RGB-NIR imagery. It must not be described as observed or ground-truth 2.5 m imagery.

## Current Application Flow

```text
Satellite GeoTIFF
       |
       v
Input Inspection
       |
       v
Preprocessing
       |
       v
Model Preparation
       |
       v
LDSR-S2 4x Super Resolution
       |
       v
Result GeoTIFF + Previews
       |
       v
JSON Report