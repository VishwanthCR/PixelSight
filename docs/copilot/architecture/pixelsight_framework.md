# PixelSight Framework Architecture

## 1. Overview

PixelSight is a modular satellite-image processing framework for
deep-learning-based super-resolution and downstream geospatial analysis.

The framework is designed to accept supported satellite imagery as
input and provide a standardized processing pipeline consisting of:

- Input inspection
- Input validation
- Preprocessing
- Individual image processing
- Batch processing
- Tiled super-resolution
- Uncertainty estimation
- Land-cover segmentation
- Urban planning analysis
- Crop monitoring analysis
- Disaster management analysis
- Evaluation
- Report generation
- Result export and download

The framework is not limited to a single pre-generated experiment.

The frontend is a client of the framework APIs, while all scientific
processing remains in the backend/services layer.

---

# 2. High-Level Architecture

```text
                         ┌──────────────────────┐
                         │      User / UI       │
                         │ React / TypeScript   │
                         └──────────┬───────────┘
                                    │
                                    │ REST API
                                    ▼
                    ┌──────────────────────────────┐
                    │        API Gateway           │
                    │        FastAPI / v1          │
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
       Input Services        Processing Services    Job Services
              │                    │                    │
       ┌──────┴──────┐      ┌──────┴─────────┐          │
       │             │      │                │          │
       ▼             ▼      ▼                ▼          ▼
    Inspect      Preprocess  Single        Batch     Job Manager
                              Processing   Processing
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Tile Processing │
                         └────────┬────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
                 LDSR-S2      Uncertainty   Segmentation
                    │             │             │
                    └─────────────┼─────────────┘
                                  ▼
                         Geospatial Analysis
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
                  Urban         Crop         Disaster
                 Planning     Monitoring     Management
                    │             │             │
                    └─────────────┼─────────────┘
                                  ▼
                         Evaluation Engine
                                  │
                                  ▼
                         Report Generator
                                  │
                                  ▼
                         Result / File Store
                                  │
                                  ▼
                         REST API Response
                                  │
                                  ▼
                              Frontend