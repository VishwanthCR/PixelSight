from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import threading
import uuid
from typing import Any, Callable


class JobStore:
    def __init__(self, results_root: Path):
        self.results_root = results_root
        self.results_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pixelsight")

    def create(self, application: str = "research") -> tuple[str, Path]:
        job_id = f"ps_{uuid.uuid4().hex[:8]}"
        job_dir = self.results_root / job_id
        for name in (
            "input",
            "preprocessing",
            "super_resolution",
            "uncertainty",
            "segmentation",
            "analysis",
            "application",
            "report",
        ):
            (job_dir / name).mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "application": application,
                "status": "queued",
                "stage": "queued",
                "progress": 0.0,
                "error": None,
                "outputs": {},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        return job_id, job_dir

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(values)

    def submit(self, job_id: str, function: Callable[[], None]) -> None:
        self.update(job_id, status="processing", stage="queued", progress=0.0)
        self._executor.submit(self._run, job_id, function)

    def _run(self, job_id: str, function: Callable[[], None]) -> None:
        self.update(job_id, status="processing", stage="queued", progress=0.0)
        try:
            function()
        except Exception as exc:
            self.update(job_id, status="failed", stage="failed", error=str(exc))
