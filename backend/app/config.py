from pathlib import Path
import os

# Automatically search for .env in current working dir or backend/
from dotenv import load_dotenv

# Try backend/.env first, then root .env
backend_env = Path(__file__).resolve().parent.parent / ".env"
root_env = Path(__file__).resolve().parent.parent.parent / ".env"
if backend_env.exists():
    load_dotenv(backend_env)
elif root_env.exists():
    load_dotenv(root_env)
else:
    load_dotenv()

RESULTS_ROOT = Path(
    os.getenv("PIXELSIGHT_RESULTS_DIR", "results/framework")
).resolve()

CACHE_ROOT = Path(
    os.getenv("PIXELSIGHT_CACHE_DIR", "results/cache")
).resolve()
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = int(
    os.getenv("PIXELSIGHT_MAX_UPLOAD_BYTES", str(512 * 1024 * 1024))
)

ALLOWED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# Copernicus Data Space Ecosystem (CDSE) / Sentinel Hub Configuration
COPERNICUS_TOKEN_URL = os.getenv(
    "COPERNICUS_TOKEN_URL",
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
)
COPERNICUS_CATALOG_URL = os.getenv(
    "COPERNICUS_CATALOG_URL",
    "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search",
)
COPERNICUS_PROCESS_URL = os.getenv(
    "COPERNICUS_PROCESS_URL",
    "https://sh.dataspace.copernicus.eu/api/v1/process",
)

# Configurable AOI and hardware limits
AOI_MAX_AREA_SQKM = float(os.getenv("PIXELSIGHT_MAX_AOI_KM2", os.getenv("PIXELSIGHT_AOI_MAX_AREA_SQKM", "25.0")))
AOI_INTERACTIVE_MAX_AREA_SQKM = float(os.getenv("PIXELSIGHT_AOI_INTERACTIVE_MAX_SQKM", "9.0"))
PIXELSIGHT_LDSR_BATCH_SIZE = int(os.getenv("PIXELSIGHT_LDSR_BATCH_SIZE", "1"))

# Geographic scope and model constants
PIXELSIGHT_SUPPORTED_COUNTRY = os.getenv("PIXELSIGHT_SUPPORTED_COUNTRY", "INDIA")
PIXELSIGHT_MAX_AOI_KM2 = AOI_MAX_AREA_SQKM
PIXELSIGHT_DEFAULT_AOI_KM = float(os.getenv("PIXELSIGHT_DEFAULT_AOI_KM", "1.28"))
PIXELSIGHT_INPUT_RESOLUTION_M = float(os.getenv("PIXELSIGHT_INPUT_RESOLUTION_M", "10.0"))
PIXELSIGHT_SR_SCALE = int(os.getenv("PIXELSIGHT_SR_SCALE", "4"))
PIXELSIGHT_LDSR_STEPS = int(os.getenv("PIXELSIGHT_LDSR_STEPS", "100"))
