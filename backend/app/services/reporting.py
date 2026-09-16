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
    evaluation: dict[str, Any] | None = None,
    urban_analysis: dict[str, Any] | None = None,
    uncertainty: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation = evaluation or {
        "status": "reference_unavailable",
        "psnr": None,
        "ssim": None,
        "sam": None,
        "reason": "No valid high-resolution reference was supplied.",
    }
    scientific_limitations = [
        "The SR product is a super-resolved representation, not observed 2.5m imagery.",
    ]
    if evaluation["status"] == "reference_available":
        scientific_limitations.append("Reference-based metrics were computed against the high-resolution reference supplied by the user.")
    else:
        scientific_limitations.append("Reference-dependent metrics are null because no valid high-resolution reference was supplied.")
    if urban_analysis:
        scientific_limitations.extend(urban_analysis.get("limitations", []))
    else:
        scientific_limitations.append("Urban segmentation was not available for this processing job.")
    if uncertainty and uncertainty.get("limitation"):
        scientific_limitations.append(uncertainty["limitation"])

    if evaluation.get("status") == "reference_available":
        evaluation_interpretation = (
            "The generated SR output was compared with the uploaded input after the input was resampled to the SR grid. "
            f"PSNR was {evaluation['psnr']:.3f} dB, SSIM was {evaluation['ssim']:.4f}, and SAM was {evaluation['sam']:.3f} degrees. "
            "These values describe input-to-output consistency; they are not evidence that the SR details are observed high-resolution ground truth."
        )
    else:
        evaluation_interpretation = (
            "A valid input-to-output evaluation could not be produced for this job. "
            f"{evaluation.get('reason', 'No evaluation values were returned.')}"
        )

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
        "evaluation": evaluation,
        "evaluation_interpretation": evaluation_interpretation,
        "urban_analysis": urban_analysis,
        "uncertainty": uncertainty,
        "scientific_limitations": scientific_limitations,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
