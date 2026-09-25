"""
PixelSight Adaptive Compute Planner

Translates a HardwareProfile into a concrete ExecutionPlan that governs:
  - LDSR-S2 tile processing order and parallelism
  - Batch sizes for segmentation inference
  - Memory guard limits
  - Recommended parallelism for multi-application batch jobs
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from backend.app.services.compute.hardware import HardwareProfile, get_hardware_profile


@dataclass
class ExecutionPlan:
    """
    The concrete plan for executing a single PixelSight workload on
    the detected hardware.

    All numerical values are safe upper bounds — the runtime may use
    fewer resources if the input is small.
    """
    # Inference device
    device: str  # passed directly to torch
    ldsr_device: str

    # Tile processing
    tile_size: int = 128          # fixed by LDSR-S2 architecture
    tile_overlap: int = 12        # fixed by LDSR-S2 architecture
    tile_batch_size: int = 1      # how many tiles to process per CUDA call
    max_parallel_tiles: int = 1   # thread-level concurrency for tile I/O

    # Segmentation
    seg_batch_size: int = 1       # UNet batch size
    seg_patch_size: int = 128     # UNet input patch

    # Uncertainty
    uncertainty_monte_carlo_runs: int = 8  # dropout forward passes

    # Memory
    fp16_inference: bool = False
    bf16_inference: bool = False
    max_active_jobs: int = 1      # concurrent jobs for batch processing

    # Diagnostics
    strategy_label: str = "cpu_safe"
    notes: list[str] = field(default_factory=list)


class ComputePlanner:
    """
    Translates a HardwareProfile into an ExecutionPlan.

    Usage::

        planner = ComputePlanner()
        plan = planner.plan_for_single_image()
        plan = planner.plan_for_batch(n_images=10)
    """

    def __init__(self, profile: Optional[HardwareProfile] = None) -> None:
        self.profile = profile or get_hardware_profile()

    # ── Public API ──────────────────────────────────────────────────────────

    def plan_for_single_image(self) -> ExecutionPlan:
        """
        Returns an ExecutionPlan optimised for a single-image pipeline
        (Research / Crop / Urban / Disaster per-image).
        """
        p = self.profile
        notes: list[str] = []

        if p.environment == "cuda_capable" and p.primary_gpu:
            vram = p.primary_gpu.vram_gb
            return self._cuda_plan(vram, notes, batch_job=False)

        if p.environment == "mps":
            notes.append("Apple MPS backend — FP16 enabled, batch size 1.")
            return ExecutionPlan(
                device=p.device,
                ldsr_device=p.ldsr_device,
                tile_batch_size=1,
                max_parallel_tiles=2,
                seg_batch_size=1,
                fp16_inference=True,
                max_active_jobs=1,
                strategy_label="mps_optimised",
                notes=notes,
            )

        # CPU fallback
        workers = max(1, min(p.cpu_count_logical // 2, 2))
        notes.append(
            f"CPU-only mode: {p.cpu_count_logical} logical cores, "
            f"{workers} tile workers. Inference will be slow."
        )
        return ExecutionPlan(
            device="cpu",
            ldsr_device="cpu",
            tile_batch_size=1,
            max_parallel_tiles=workers,
            seg_batch_size=1,
            fp16_inference=False,
            max_active_jobs=1,
            strategy_label="cpu_safe",
            notes=notes,
        )

    def plan_for_batch(self, n_images: int = 1) -> ExecutionPlan:
        """
        Returns an ExecutionPlan tuned for a multi-image batch workload.

        The planner reserves headroom to avoid OOM errors when multiple
        tiles may be resident simultaneously.
        """
        p = self.profile
        notes: list[str] = [f"Batch workload: {n_images} images requested."]

        if p.environment == "cuda_capable" and p.primary_gpu:
            vram = p.primary_gpu.vram_gb
            plan = self._cuda_plan(vram, notes, batch_job=True)
            # Conservative active job limit — one per 4 GB headroom
            plan.max_active_jobs = max(1, int(vram // 4))
            plan.notes.append(
                f"Max concurrent jobs: {plan.max_active_jobs} "
                f"(VRAM={vram:.1f}GB / 4GB per job heuristic)."
            )
            return plan

        # CPU batch — pure sequential
        notes.append("CPU batch mode: sequential processing, 1 active job.")
        return ExecutionPlan(
            device="cpu",
            ldsr_device="cpu",
            tile_batch_size=1,
            max_parallel_tiles=max(1, p.cpu_count_logical // 2),
            seg_batch_size=1,
            fp16_inference=False,
            max_active_jobs=1,
            strategy_label="cpu_batch_sequential",
            notes=notes,
        )

    def plan_for_workload(
        self,
        width: int = 128,
        height: int = 128,
        batch_size: int = 1,
        n_images: int = 1,
    ) -> dict:
        """
        Compute a workload-specific execution plan taking input dimensions,
        tile overlap geometry, and hardware constraints into account.
        """
        p = self.profile
        step = 128 - 12  # 116 px stride for 128x128 tiles with 12px overlap
        w_eff = max(128, width)
        h_eff = max(128, height)
        tiles_x = max(1, math.ceil((w_eff - 128) / step) + 1) if w_eff > 128 else 1
        tiles_y = max(1, math.ceil((h_eff - 128) / step) + 1) if h_eff > 128 else 1
        tiles_per_image = tiles_x * tiles_y
        total_tiles = tiles_per_image * max(1, n_images)

        if p.device_type == "cuda" and p.primary_gpu:
            vram = p.primary_gpu.vram_gb
            gpu_count = len(p.gpus)
            if gpu_count > 1:
                backend = "multi_gpu"
                workers = min(gpu_count, 4)
                parallelism = "multi_gpu"
                reason = f"{gpu_count} GPUs detected; parallel multi-device execution enabled."
            else:
                backend = "single_gpu"
                workers = 1
                parallelism = "gpu_serial"
                reason = f"Single {vram:.1f} GB GPU detected; LDSR-S2 inference is VRAM constrained."
            gpu_devices = [g.index for g in p.gpus]
            tile_batch = 1 if vram < 8 else 2
        elif p.device_type == "mps":
            backend = "single_gpu"
            workers = 1
            gpu_devices = [0]
            tile_batch = 1
            parallelism = "mps_serial"
            reason = "Apple Silicon unified memory detected; sequential inference planned."
        else:
            backend = "cpu"
            workers = max(1, min(p.cpu_count_logical // 2, 4))
            gpu_devices = []
            tile_batch = 1
            parallelism = "cpu_multithread"
            reason = f"No GPU available; executing across {workers} CPU worker threads."

        return {
            "backend": backend,
            "execution_environment": p.execution_environment,
            "workers": workers,
            "gpu_devices": gpu_devices,
            "tile_batch_size": tile_batch,
            "parallelism": parallelism,
            "reason": reason,
            "input_dimensions": {"width": width, "height": height},
            "tile_dimensions": {"tile_size": 128, "overlap": 12, "stride": step},
            "estimated_tiles_per_scene": tiles_per_image,
            "total_estimated_tiles": total_tiles,
            "diffusion_steps": 100,
            "scale_factor": 4,
            "fp16_enabled": p.supports_fp16,
            "max_active_jobs": max(1, int(p.primary_gpu.vram_gb // 4)) if (p.primary_gpu and p.device_type == "cuda") else 1,
        }

    def as_dict(self) -> dict:
        """Return the hardware profile as a JSON-serialisable dict for API responses."""
        p = self.profile
        plan = self.plan_for_single_image()
        workload_plan = self.plan_for_workload(128, 128, 1, 1)

        # Base prompt structure
        res = {
            "environment": "hpc" if p.is_hpc else "local",
            "execution_environment": p.execution_environment,
            "execution_backend": p.execution_backend,
            "cpu": {
                "model": p.cpu_model,
                "cores": p.cpu_count,
                "logical_processors": p.cpu_count_logical,
                "ram_gb": round(p.system_ram_gb, 2),
            },
            "gpu": {
                "available": p.is_gpu_available,
                "count": len(p.gpus),
                "devices": [
                    {
                        "index": g.index,
                        "name": g.name,
                        "vram_gb": round(g.vram_gb, 2),
                        "compute_capability": g.compute_capability,
                    }
                    for g in p.gpus
                ],
                "primary": (
                    {
                        "name": p.primary_gpu.name,
                        "vram_gb": round(p.primary_gpu.vram_gb, 2),
                        "compute_capability": p.primary_gpu.compute_capability,
                    }
                    if p.primary_gpu
                    else None
                ),
            },
            "cuda": {
                "available": p.is_gpu_available,
                "version": p.cuda_version or ("12.0" if p.is_gpu_available else "none"),
                "supports_fp16": p.supports_fp16,
                "supports_bf16": p.supports_bf16,
            },
            # Backward-compatible blocks for existing tests & UI
            "hardware": {
                "device": p.device,
                "device_type": p.device_type,
                "environment": p.environment,
                "execution_environment": p.execution_environment,
                "execution_backend": p.execution_backend,
                "gpus": [
                    {
                        "index": g.index,
                        "name": g.name,
                        "vram_gb": round(g.vram_gb, 2),
                        "compute_capability": g.compute_capability,
                    }
                    for g in p.gpus
                ],
                "primary_gpu": (
                    {
                        "name": p.primary_gpu.name,
                        "vram_gb": round(p.primary_gpu.vram_gb, 2),
                        "compute_capability": p.primary_gpu.compute_capability,
                    }
                    if p.primary_gpu
                    else None
                ),
                "cpu_model": p.cpu_model,
                "cpu_physical_cores": p.cpu_count,
                "cpu_logical_cores": p.cpu_count_logical,
                "system_ram_gb": round(p.system_ram_gb, 2),
                "supports_fp16": p.supports_fp16,
                "supports_bf16": p.supports_bf16,
                "torch_version": p.torch_version,
                "platform": p.platform_info,
            },
            "execution_plan": {
                "strategy": plan.strategy_label,
                "ldsr_device": plan.ldsr_device,
                "tile_batch_size": plan.tile_batch_size,
                "tile_workers": plan.max_parallel_tiles,
                "segmentation_batch_size": plan.seg_batch_size,
                "fp16_inference": plan.fp16_inference,
                "max_active_jobs": plan.max_active_jobs,
                "uncertainty_mc_runs": plan.uncertainty_monte_carlo_runs,
                "notes": plan.notes,
            },
            "plan": workload_plan,
        }
        return res

    # ── Private helpers ─────────────────────────────────────────────────────

    def _cuda_plan(
        self,
        vram_gb: float,
        notes: list[str],
        batch_job: bool,
    ) -> ExecutionPlan:
        p = self.profile

        # Scale everything from VRAM
        if vram_gb >= 16:
            tile_batch = 4
            tile_workers = 4
            seg_batch = 4
            strategy = "cuda_high_end"
        elif vram_gb >= 8:
            tile_batch = 2
            tile_workers = 2
            seg_batch = 2
            strategy = "cuda_mid_range"
        elif vram_gb >= 4:
            tile_batch = 1
            tile_workers = 2
            seg_batch = 2
            strategy = "cuda_low_vram"
        else:
            # ≤4 GB (integrated / laptop GPU)
            tile_batch = 1
            tile_workers = 1
            seg_batch = 1
            strategy = "cuda_minimal"

        # Batch jobs are more conservative
        if batch_job:
            seg_batch = max(1, seg_batch // 2)
            tile_workers = max(1, tile_workers - 1)

        notes.append(
            f"CUDA device: {p.primary_gpu.name} ({vram_gb:.1f} GB VRAM), "
            f"strategy: {strategy}."
        )

        return ExecutionPlan(
            device=p.device,
            ldsr_device="cuda",
            tile_batch_size=tile_batch,
            max_parallel_tiles=tile_workers,
            seg_batch_size=seg_batch,
            fp16_inference=p.supports_fp16,
            bf16_inference=p.supports_bf16,
            max_active_jobs=1,
            strategy_label=strategy,
            notes=notes,
        )
