# Copernicus Data Space Ecosystem (CDSE) Integration Guide

PixelSight integrates directly with the official **Copernicus Data Space Ecosystem (CDSE)** and **Sentinel Hub APIs** to provide native search, interactive AOI bounding box selection, multi-spectral band acquisition (B02, B03, B04, B08), and downstream LDSR-S2 super-resolution.

---

## Architecture Overview

```text
[ React Interactive Satellite Map ]
              │ (Draw AOI bbox / date range / cloud cover)
              ▼
[ POST /api/v1/copernicus/search ]
              │
              ▼ (Sentinel Hub Catalog 1.0.0 API)
[ CDSE Sentinel-2 L2A Scenes ]
              │
              ▼ (User selects scene candidate)
[ POST /api/v1/applications/{app}/from-copernicus ]
              │
              ▼ (Sentinel Hub Processing API)
[ 4-Band GeoTIFF (B02, B03, B04, B08 at 10 m BOA Float32) ]
              │
              ▼
[ PixelSight Core Engine (LDSR-S2, 100 diffusion steps) ]
              │
       ┌──────┴──────┐
       ▼             ▼
[ 4× SR Output ] [ Proxy Uncertainty ]
       │             │
       └──────┬──────┘
              ▼
[ Downstream Application (Crop / Urban / Disaster) ]
```

---

## Authentication & Security

CDSE uses OAuth2 OpenID Connect authentication via Keycloak.

### Credentials Setup
Configure the following in `backend/.env` (never commit real secrets):
```env
COPERNICUS_CLIENT_ID=your_cdse_oauth_client_id
COPERNICUS_CLIENT_SECRET=your_cdse_oauth_client_secret
```

### Security Guarantees
- **Backend-Only**: OAuth credentials and Bearer tokens are strictly isolated in `backend/app/services/copernicus_auth.py`.
- **Zero Frontend Exposure**: No credentials or tokens are ever sent to React or returned in API responses.
- **In-Memory Caching with Expiry Buffer**: OAuth access tokens are cached thread-safely in Python memory and automatically refreshed 60 seconds before expiration.
- **No Token Logging**: Secrets and Authorization headers are masked and never logged to stdout or log files.

---

## API Reference

### 1. Capabilities
`GET /api/v1/copernicus/capabilities`
Returns authentication status, supported collections (`sentinel-2-l2a`), required bands (`B02, B03, B04, B08`), and AOI engineering limits.

### 2. AOI Estimation
`POST /api/v1/copernicus/estimate`
Calculates geodesic dimensions (km), area (km²), native 10 m pixel dimensions, 4× SR pixel dimensions, and 128×128 tile counts. Labels execution mode (`interactive` vs `background`).

### 3. Scene Search
`POST /api/v1/copernicus/search`
Queries the CDSE Catalog API for Sentinel-2 L2A tiles matching the specified AOI, date range, and maximum cloud cover percentage.

### 4. Direct Acquisition
`POST /api/v1/copernicus/acquire`
Constructs an evalscript requesting BOA reflectance for bands B02, B03, B04, and B08, issues a POST request to Sentinel Hub Processing API (`/api/v1/process`), and saves an authoritative GeoTIFF with full CRS and affine transform.

### 5. Application Launch from Copernicus
`POST /api/v1/applications/{application}/from-copernicus`
Asynchronously triggers:
- Single-scene acquisition for `crop` and `urban`.
- Dual-scene acquisition (pre-event + post-event) for `disaster`.
- 100 diffusion step LDSR-S2 super-resolution.
- Uncertainty mapping.
- Downstream domain analysis.
- Multi-format reports and manifest generation.

---

## Scientific Integrity & Limitations

1. **Super-Resolution Attribution**: Output is strictly labeled as **"4× super-resolved representation (~2.5 m equivalent)"** and never described as "true 2.5 m ground-truth imagery".
2. **Deterministic Uncertainty**: When multi-pass sampling is not configured, uncertainty maps are explicitly labeled as **"Proxy uncertainty"** (local spectral variance / reconstruction residual proxy).
3. **Multispectral Input Integrity**: PixelSight rejects RGB/JPEG/PNG uploads that lack authentic Near-Infrared (B08) data. Fake NIR synthesis is strictly forbidden.
4. **Conservative Domain Language**:
   - Crop: Stress indicators are derived from NDVI differences; no unilateral claims of disease diagnosis.
   - Urban: Built-up fraction and 7-class land cover distributions are reported neutrally.
   - Disaster: Areas exhibiting significant spectral change are identified as **"potential affected regions"** rather than unverified destruction claims.
