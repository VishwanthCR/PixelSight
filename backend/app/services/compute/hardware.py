"""
PixelSight Hardware Detection Module

Detects and profiles the local compute environment: GPU/CPU capabilities,
available VRAM, core counts, and memory. Used by the planner to determine
the optimal execution strategy for inference workloads.
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPUInfo:
    """Information about a single GPU device."""
    index: int
    name: str
    vram_gb: float
    compute_capability: Optional[str] = None
    is_cuda: bool = True


@dataclass
class HardwareProfile:
    """
    Full hardware profile of the current machine.

    This is the central data structure consumed by the ComputePlanner
    to determine optimal batch sizes, tile counts, and parallelism.
    """
    # Device selection
    device: str  # "cuda:0", "cpu", etc.
    device_type: str  # "cuda", "mps", "cpu"

    # GPU info (None if CPU-only)
    gpus: list[GPUInfo] = field(default_factory=list)
    primary_gpu: Optional[GPUInfo] = None

    # CPU info
    cpu_count: int = 1
    cpu_count_logical: int = 1
    cpu_model: str = "Unknown"

    # Memory
    system_ram_gb: float = 0.0

    # Inferred capabilities
    supports_fp16: bool = False
    supports_bf16: bool = False
    recommended_batch_size: int = 1
    recommended_tile_workers: int = 1
    max_tiles_in_memory: int = 4

    # Context flags
    is_gpu_available: bool = False
    is_mps_available: bool = False  # Apple Silicon
    ldsr_device: str = "cpu"

    # Diagnostics
    environment: str = "unknown"  # "cuda_capable", "cpu_only", "mps"
    execution_environment: str = "UNKNOWN"  # "LOCAL_CPU", "LOCAL_SINGLE_GPU", "LOCAL_MULTI_GPU", "HPC", "UNKNOWN"
    execution_backend: str = "cpu"  # "single_gpu", "multi_gpu", "cpu"
    torch_version: str = ""
    cuda_version: str = ""
    python_version: str = ""
    platform_info: str = ""
    is_hpc: bool = False


def detect_hardware() -> HardwareProfile:
    """
    Probe the compute environment and return a HardwareProfile.

    This function is the single source of truth for hardware discovery
    in the PixelSight framework. It is safe to call at startup and
    results are deterministic within a single process run.
    """
    try:
        import torch
    except ImportError:
        return HardwareProfile(
            device="cpu",
            device_type="cpu",
            environment="cpu_only",
            platform_info=platform.platform(),
        )

    torch_version = torch.__version__
    plat = platform.platform()

    # ── CPU info ────────────────────────────────────────────────────────────
    try:
        cpu_logical = os.cpu_count() or 1
        cpu_physical = cpu_logical  # psutil not required
        try:
            import psutil
            cpu_physical = psutil.cpu_count(logical=False) or cpu_logical
        except ImportError:
            pass
    except Exception:
        cpu_logical = 1
        cpu_physical = 1

    # ── System RAM ──────────────────────────────────────────────────────────
    system_ram_gb = 0.0
    try:
        import psutil
        system_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass

    # ── CUDA GPUs ───────────────────────────────────────────────────────────
    gpus: list[GPUInfo] = []
    primary_gpu: Optional[GPUInfo] = None

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024 ** 3)
            cc = f"{props.major}.{props.minor}"
            gpu = GPUInfo(
                index=i,
                name=props.name,
                vram_gb=vram_gb,
                compute_capability=cc,
                is_cuda=True,
            )
            gpus.append(gpu)

        primary_gpu = gpus[0] if gpus else None
        device = "cuda:0"
        device_type = "cuda"
        is_gpu = True
        is_mps = False
        environment = "cuda_capable"

        # FP16/BF16 support (compute capability ≥ 7.0)
        supports_fp16 = False
        supports_bf16 = False
        if primary_gpu and primary_gpu.compute_capability:
            major = int(primary_gpu.compute_capability.split(".")[0])
            supports_fp16 = major >= 7
            supports_bf16 = major >= 8

        # Adaptive batch size based on VRAM
        vram = primary_gpu.vram_gb if primary_gpu else 0.0
        if vram >= 16:
            batch_size = 4
            tile_workers = 4
            max_tiles = 16
        elif vram >= 8:
            batch_size = 2
            tile_workers = 2
            max_tiles = 8
        elif vram >= 4:
            batch_size = 1
            tile_workers = 2
            max_tiles = 4
        else:
            # Very low VRAM (e.g., 4 GB integrated or 6 GB laptop)
            batch_size = 1
            tile_workers = 1
            max_tiles = 2

        ldsr_device = "cuda"

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        device_type = "mps"
        is_gpu = False
        is_mps = True
        environment = "mps"
        supports_fp16 = True
        supports_bf16 = False
        batch_size = 1
        tile_workers = 2
        max_tiles = 4
        ldsr_device = "mps"

    else:
        device = "cpu"
        device_type = "cpu"
        is_gpu = False
        is_mps = False
        environment = "cpu_only"
        supports_fp16 = False
        supports_bf16 = False
        # Scale CPU tile workers from available logical cores
        tile_workers = max(1, min(cpu_logical // 2, 4))
        batch_size = 1
        max_tiles = 2
        ldsr_device = "cpu"

    # ── HPC & Environment Taxonomy ───────────────────────────────────────────
    is_hpc = bool(
        os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_NAME")
        or os.environ.get("PBS_JOBID")
        or os.environ.get("LSB_JOBID")
        or os.environ.get("SGE_CELL")
    )

    if is_hpc:
        exec_env = "HPC"
        exec_backend = "multi_gpu" if len(gpus) > 1 else ("single_gpu" if gpus else "cpu")
    elif len(gpus) > 1:
        exec_env = "LOCAL_MULTI_GPU"
        exec_backend = "multi_gpu"
    elif len(gpus) == 1:
        exec_env = "LOCAL_SINGLE_GPU"
        exec_backend = "single_gpu"
    elif device_type == "mps":
        exec_env = "LOCAL_SINGLE_GPU"
        exec_backend = "single_gpu"
    elif device_type == "cpu":
        exec_env = "LOCAL_CPU"
        exec_backend = "cpu"
    else:
        exec_env = "UNKNOWN"
        exec_backend = "cpu"

    cuda_ver = getattr(torch.version, "cuda", "") or ""
    cpu_model = _detect_cpu_model()

    return HardwareProfile(
        device=device,
        device_type=device_type,
        gpus=gpus,
        primary_gpu=primary_gpu,
        cpu_count=cpu_physical,
        cpu_count_logical=cpu_logical,
        cpu_model=cpu_model,
        system_ram_gb=system_ram_gb,
        supports_fp16=supports_fp16,
        supports_bf16=supports_bf16,
        recommended_batch_size=batch_size,
        recommended_tile_workers=tile_workers,
        max_tiles_in_memory=max_tiles,
        is_gpu_available=is_gpu,
        is_mps_available=is_mps,
        ldsr_device=ldsr_device,
        environment=environment,
        execution_environment=exec_env,
        execution_backend=exec_backend,
        torch_version=torch_version,
        cuda_version=cuda_ver,
        python_version=platform.python_version(),
        platform_info=plat,
        is_hpc=is_hpc,
    )


def _detect_cpu_model() -> str:
    """Best-effort CPU model name detection across platforms."""
    try:
        if platform.system() == "Windows":
            import subprocess
            result = subprocess.run(
                ["wmic", "cpu", "get", "Name", "/format:value"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            for line in result.stdout.splitlines():
                if "=" in line:
                    return line.split("=", 1)[1].strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return result.stdout.strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


# Module-level singleton — detected once at import time
_profile: Optional[HardwareProfile] = None


def get_hardware_profile() -> HardwareProfile:
    """
    Returns the cached hardware profile, detecting it on first call.

    Thread-safe for read-only access after first call.
    """
    global _profile
    if _profile is None:
        _profile = detect_hardware()
    return _profile
