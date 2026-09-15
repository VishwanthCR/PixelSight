# PixelSight Frontend

This React/Vite application is an API client for the PixelSight FastAPI backend. It does not run model inference in the browser.

## Run

From this directory:

```powershell
npm install
npm run dev
```

Open `http://localhost:5173`.

Start the backend separately with the Python environment that contains PyTorch and OpenSR:

```powershell
python -m uvicorn backend.app.main:app --reload
```

The Vite proxy forwards `/api` to `http://127.0.0.1:8000`. Set `VITE_BACKEND_URL` in `.env` when the backend runs elsewhere.

## Workflow

1. Upload a `.tif` or `.tiff` GeoTIFF.
2. Inspect explicit `B02`, `B03`, `B04`, and `B08` metadata.
3. Start processing.
4. Poll the real job stages.
5. Display the returned original and LDSR-S2 previews.
6. Download the SR GeoTIFF and generated JSON report.

The backend runs the real LDSR-S2 checkpoint with 100 sampling steps and returns a 4x super-resolved representation (~2.5m equivalent). PSNR, SSIM, and SAM remain unavailable unless a valid high-resolution reference is supplied.

## API routes used

- `GET /api/v1/health`
- `POST /api/v1/inspect`
- `POST /api/v1/process`
- `GET /api/v1/jobs/{job_id}`
- `GET /api/v1/results/{job_id}`
- `GET /api/v1/reports/{job_id}`
- `GET /api/v1/results/{job_id}/files/{relative_path}`
