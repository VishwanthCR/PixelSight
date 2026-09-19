# PixelSight

**Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imagery**

PixelSight is a research-grade, uncertainty-aware satellite intelligence platform developed for SIH 2026 Problem Statement **SIH26142**. It combines LDSR-S2 diffusion-based 4× super-resolution with downstream geospatial analysis across four application domains: research/core, crop monitoring, urban analysis, and disaster management.

> **Scientific Integrity Notice:** PixelSight produces a model-derived 4× super-resolved representation (~2.5 m equivalent spatial resolution) from compatible 10 m Sentinel-2 RGB-NIR imagery. It must never be described as observed or ground-truth 2.5 m imagery. All metrics are computed neutrally without fabrication.

---

## Platform Overview

```text
Sentinel-2 GeoTIFF (10 m, B02/B03/B04/B08)
         │
         ▼
   Input Inspection
         │
         ▼
   Preprocessing (float32, [0,1] reflectance)
         │
         ▼
   LDSR-S2 4× Super-Resolution (100 diffusion steps)
         │
    ┌────┴────────────────────────┐
    ▼                             ▼
 Mean SR Output         Uncertainty Estimation
    │                             │
    └──────────┬──────────────────┘
               │
    ┌──────────┼──────────────────┐
    ▼          ▼                  ▼
  Crop       Urban            Disaster
Monitoring  Analysis         Management
    │          │                  │
    └──────────┴──────────────────┘
               │
    Multi-format Reports (JSON / Markdown / HTML)
               │
               ▼
    React Results Dashboard
```

---

## Architecture

### Core Engine — `PixelSightEngine`

The canonical entry point (`backend/app/services/core_engine.py`) provides:

| Method | Description |
|---|---|
| `inspect(path)` | Validate dimensions, CRS, band compatibility |
| `preprocess(input, output)` | Normalize to float32 reflectance `[0, 1]` |
| `enhance(norm, sr_out, ...)` | LDSR-S2 4× SR, 100 diffusion steps, tile/stitch |
| `uncertainty(norm, sr, tif, png)` | Pixel-level stochastic uncertainty map |

All four application pipelines share this engine. It includes a test/mock mode (`PIXELSIGHT_TEST_MODE=1`) for automated CI without GPU weights.

### Application Services

| Service | File | Description |
|---|---|---|
| Research Core | `core_engine.py` | SR + uncertainty, neutral reporting |
| Crop Monitoring | `services/crop.py` | NDVI (B04/B08), consistency metrics, stress signals |
| Urban Analysis | `services/urban.py` | Built-up mask, vegetation fraction, segmentation |
| Disaster Management | `services/disaster.py` | Temporal alignment, spectral change detection, reliability breakdown |

---

## API Reference

### Core Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health check |
| `GET` | `/api/v1/capabilities` | Model and platform capabilities |
| `GET` | `/api/v1/applications` | List available application domains |
| `POST` | `/api/v1/inspect` | Inspect uploaded GeoTIFF |
| `POST` | `/api/v1/preprocess` | Preprocess and normalize raster |
| `POST` | `/api/v1/process` | Full research pipeline (SR + uncertainty + urban + report) |
| `GET` | `/api/v1/jobs/{job_id}` | Poll research job status |
| `GET` | `/api/v1/results/{job_id}` | Get research job results |
| `GET` | `/api/v1/reports/{job_id}` | Retrieve JSON report |
| `GET` | `/api/v1/manifest/{job_id}` | Retrieve artifact manifest |
| `GET` | `/api/v1/results/{job_id}/files/{path}` | Download any job artifact |

### Application Pipelines

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/v1/applications/crop` | Crop monitoring pipeline (single image) |
| `POST` | `/api/v1/applications/urban` | Urban analysis pipeline (single image) |
| `POST` | `/api/v1/applications/disaster` | Disaster management pipeline (pre + post event) |
| `GET` | `/api/v1/applications/{job_id}` | Poll application job status |
| `GET` | `/api/v1/applications/{job_id}/results` | Get application results |
| `GET` | `/api/v1/applications/{job_id}/report` | Get application report |

### Copernicus Data Space Ecosystem (CDSE) & Sentinel Hub

| Method | Route | Description |
|---|---|---|
| `GET` | `/api/v1/copernicus/capabilities` | Check OAuth configuration and supported parameters |
| `POST` | `/api/v1/copernicus/estimate` | AOI dimensions, geodesic area, native/SR pixel counts |
| `POST` | `/api/v1/copernicus/search` | Search Sentinel-2 L2A scenes via CDSE Catalog API |
| `POST` | `/api/v1/copernicus/acquire` | Acquire 4-band BOA GeoTIFF (B02/B03/B04/B08) via Processing API |
| `POST` | `/api/v1/applications/{app}/from-copernicus` | Complete async pipeline from scene/AOI to application report |

---

## Project Structure

```
PixelSight/
├── backend/
│   └── app/
│       ├── main.py                    # FastAPI application + all routes
│       ├── schemas.py                 # Pydantic models
│       ├── config.py                  # Environment configuration
│       └── services/
│           ├── core_engine.py         # PixelSightEngine (shared)
│           ├── crop.py                # Crop monitoring pipeline
│           ├── urban.py               # Urban analysis pipeline
│           ├── disaster.py            # Disaster management pipeline
│           ├── reporting.py           # Multi-format report generation
│           ├── app_registry.py        # Application registry (discovery)
│           ├── raster.py              # GeoTIFF inspection + preprocessing
│           ├── uncertainty.py         # Stochastic uncertainty estimation
│           ├── visualization.py       # RGB preview generation
│           └── jobs.py                # Job store + async submission
├── configs/
│   └── applications/
│       ├── research.yaml
│       ├── crop.yaml
│       ├── urban.yaml
│       └── disaster.yaml
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── LandingPage.jsx         # Application selector
│       │   ├── ProcessingPage.jsx      # Domain-aware processing progress
│       │   ├── ResultsDashboard.jsx    # Results and report viewer
│       │   ├── CropMonitoringPage.jsx  # Crop-specific UI
│       │   ├── UrbanAnalysisPage.jsx   # Urban-specific UI
│       │   └── DisasterManagementPage.jsx  # Disaster-specific UI
│       └── api/
│           └── srmApi.js               # API client for all domains
├── scripts/
│   ├── validate_e2e_512.py            # End-to-end validation on 512×512
│   ├── plan2/                         # LDSR-S2 inference scripts
│   ├── evaluate_bicubic.py            # Bicubic baseline evaluation
│   └── segmentation/                  # Segmentation model scripts
├── tests/
│   ├── test_applications/
│   │   ├── test_crop.py
│   │   ├── test_urban.py
│   │   ├── test_disaster.py
│   │   └── test_api_applications.py
│   ├── test_api/
│   ├── test_segmentation/
│   └── test_scientific_evaluation.py
├── dataset/
│   └── plan2/geotiff/                 # Training and validation GeoTIFFs
└── checkpoints/                       # Model weight files
```

---

## Setup

### Requirements

- Python ≥ 3.10
- CUDA-capable GPU recommended (RTX 3050 6 GB tested)
- Node.js ≥ 18 (frontend)

### Backend
 
```bash
pip install -r requirements.txt
```

#### Copernicus OAuth Setup (Optional for Live Satellite Search & Acquisition)
Create `backend/.env` with your CDSE credentials:
```env
COPERNICUS_CLIENT_ID=your_client_id_here
COPERNICUS_CLIENT_SECRET=your_client_secret_here
```
> **Security Notice:** Credentials remain strictly backend-only and are never sent to the browser or logged.

To run with test/mock mode (no GPU weights needed):

```bash
$env:PIXELSIGHT_TEST_MODE = "1"
uvicorn backend.app.main:app --reload
```

For real LDSR-S2 inference (requires model weights):

```bash
uvicorn backend.app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Model Weights

Place the LDSR-S2 checkpoint in `checkpoints/plan2/ldsr_s2_best.pth` and the segmentation checkpoint in `checkpoints/segmentation/unet_worldcover_best.pth`.

---

## Running Tests

```bash
# All tests (must pass 100%)
python -m pytest tests/ -v

# Application pipeline tests only
python -m pytest tests/test_applications/ -v

# API tests
python -m pytest tests/test_api/ -v
```

Test environment automatically uses mock mode (`PIXELSIGHT_TEST_MODE=1`) — no GPU weights required.

---

## End-to-End Validation

Validates all four domains on a real 512×512 Sentinel-2 tile:

```bash
python scripts/validate_e2e_512.py
```

This script:
1. Inspects and preprocesses `dataset/plan2/geotiff/train_test_512_10m.tif`
2. Runs 4× LDSR-S2 super-resolution
3. Generates uncertainty maps
4. Runs crop, urban, and disaster analysis pipelines
5. Writes multi-format reports and artifact manifests

---

## Output Artifacts

Each job produces:

| Artifact | Path |
|---|---|
| Super-resolved GeoTIFF | `super_resolution/sr.tif` |
| SR Preview (PNG) | `super_resolution/sr_preview.png` |
| Uncertainty Map (GeoTIFF) | `uncertainty/uncertainty_map.tif` |
| Uncertainty Map (PNG) | `uncertainty/uncertainty_map.png` |
| Application outputs | `application/{domain}/` |
| Report (JSON) | `report/report.json` |
| Report (Markdown) | `report/report.md` |
| Report (HTML) | `report/report.html` |
| Manifest | `manifest.json` |

---

## Scientific Limitations

- SR output represents a model-derived prediction, not an observed acquisition.
- NDVI and segmentation-based metrics are derived from model output and may diverge from independently acquired high-resolution reference data.
- Uncertainty maps are stochastic estimates, not calibrated confidence intervals.
- Change detection in the disaster domain relies on spectral difference; geometric misalignment between acquisitions may inflate false positives.

---

## License

Research and academic use. SIH 2026 submission by the PixelSight team.