"""Standard output manifest and artifact record generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ArtifactManifest:
    """Canonical artifact manifest describing a super-resolution inference run."""

    job_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model: dict[str, Any] = field(default_factory=dict)
    input_metadata: dict[str, Any] = field(default_factory=dict)
    preprocessing: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    urban_analysis: dict[str, Any] = field(default_factory=dict)
    scientific_disclaimer: str = (
        "Inferred model products are mathematical approximations (~2.5m equivalent), "
        "not physically observed ground truth. Metrics without independent reference "
        "represent consistency checks."
    )
    status_flags: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return target


def hash_file_or_config(content: str | bytes | Path) -> str:
    """Compute sha256 hash for provenance and reproducibility."""
    hasher = hashlib.sha256()
    if isinstance(content, Path):
        hasher.update(content.read_bytes())
    elif isinstance(content, str):
        hasher.update(content.encode("utf-8"))
    else:
        hasher.update(content)
    return hasher.hexdigest()[:16]


def create_manifest(
    *,
    job_id: str,
    model_name: str,
    model_version: str = "0.1.0",
    checkpoint: str | None = None,
    input_path: str | Path,
    input_metadata: dict[str, Any] | None = None,
    preprocessing: list[dict[str, Any]] | None = None,
    artifacts: dict[str, str] | None = None,
    metrics: dict[str, Any] | None = None,
    uncertainty: dict[str, Any] | None = None,
    urban_analysis: dict[str, Any] | None = None,
    independent_reference_available: bool = False,
    is_calibrated_uncertainty: bool = False,
) -> ArtifactManifest:
    """Construct an ArtifactManifest with canonical scientific flags."""
    return ArtifactManifest(
        job_id=job_id,
        model={
            "name": model_name,
            "version": model_version,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "hash": hash_file_or_config(str(checkpoint)) if checkpoint and Path(checkpoint).exists() else None,
        },
        input_metadata={"source": str(input_path), **(input_metadata or {})},
        preprocessing=preprocessing or [],
        artifacts=artifacts or {},
        metrics=metrics or {},
        uncertainty=uncertainty or {},
        urban_analysis=urban_analysis or {},
        status_flags={
            "independent_reference_available": independent_reference_available,
            "consistency_only": not independent_reference_available,
            "proxy_uncertainty": not is_calibrated_uncertainty,
            "calibrated_uncertainty": is_calibrated_uncertainty,
            "urban_semantic_estimate": True,
            "validated_object_count": False,
        },
    )
