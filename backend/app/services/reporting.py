from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def write_report(
    path: Path,
    *,
    job_id: str,
    input_metadata: dict[str, Any],
    preprocessing: list[dict[str, Any]],
    runtime_seconds: float,
    device: str,
    output_files: dict[str, str],
    status: str = "completed",
) -> dict[str, Any]:
    report = {
        "job_id": job_id,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": input_metadata,
        "preprocessing": {"operations": preprocessing},
        "super_resolution": {
            "model": "LDSR-S2",
            "scale": 4,
            "input_tile_size": [128, 128],
            "overlap": 12,
            "sampling_steps": 100,
            "description": "4x super-resolved representation (~2.5m equivalent)",
        },
        "runtime": {"device": device, "seconds": runtime_seconds},
        "outputs": output_files,
        "evaluation": {
            "status": "reference_unavailable",
            "psnr": None,
            "ssim": None,
            "sam": None,
        },
        "scientific_limitations": [
            "The SR product is a super-resolved representation, not observed 2.5m imagery.",
            "Reference-dependent metrics are null because no valid high-resolution reference was supplied.",
            "Uncertainty and segmentation are not yet part of this processing job.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
