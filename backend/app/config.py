from pathlib import Path
import os


RESULTS_ROOT = Path(
    os.getenv("PIXELSIGHT_RESULTS_DIR", "results/framework")
).resolve()

MAX_UPLOAD_BYTES = int(
    os.getenv("PIXELSIGHT_MAX_UPLOAD_BYTES", str(512 * 1024 * 1024))
)

ALLOWED_SUFFIXES = {".tif", ".tiff"}
