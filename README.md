# PixelSight

PixelSight is a multispectral satellite-imagery super-resolution and geospatial-analysis framework.

It combines research pipelines for Sentinel-2 imagery with a FastAPI backend and React frontend.

> **Scientific note:** PixelSight produces a model-derived 4× super-resolved representation (~2.5 m equivalent) from compatible 10 m Sentinel-2 RGB-NIR imagery. It must not be described as observed or ground-truth 2.5 m imagery.

## Project Positioning

PixelSight is designed as an uncertainty-aware satellite-analysis framework rather than only an image-sharpening tool.

The system combines super-resolution, uncertainty estimation, and downstream geospatial analysis so that the usefulness of a super-resolved representation can be evaluated beyond visual appearance.

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
LDSR-S2 4× Super Resolution
       |
       +--------------------+
       |                    |
       v                    v
Mean SR Output       Uncertainty Estimation
       |                    |
       +----------+---------+
                  |
                  v
           Urban Analysis
                  |
                  v
          JSON / GeoTIFF Outputs
                  |
                  v
             Report
                  |
                  v
          React Results Dashboard