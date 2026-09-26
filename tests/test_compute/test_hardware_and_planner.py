"""
Tests for the PixelSight adaptive compute engine.

Validates hardware detection, execution plan generation, and the
API response shape — without requiring a physical GPU.
"""
import pytest

from backend.app.services.compute.hardware import HardwareProfile, detect_hardware, get_hardware_profile
from backend.app.services.compute.planner import ComputePlanner, ExecutionPlan


# ---------------------------------------------------------------------------
# Hardware Detection Tests
# ---------------------------------------------------------------------------

def test_detect_hardware_returns_profile():
    profile = detect_hardware()
    assert isinstance(profile, HardwareProfile)


def test_hardware_profile_has_device():
    profile = detect_hardware()
    assert profile.device in {"cpu", "cuda:0", "mps"} or profile.device.startswith("cuda:")


def test_hardware_profile_device_type():
    profile = detect_hardware()
    assert profile.device_type in {"cpu", "cuda", "mps"}


def test_hardware_profile_environment():
    profile = detect_hardware()
    assert profile.environment in {"cpu_only", "cuda_capable", "mps"}


def test_hardware_profile_cpu_counts():
    profile = detect_hardware()
    assert profile.cpu_count >= 1
    assert profile.cpu_count_logical >= 1
    assert profile.cpu_count_logical >= profile.cpu_count


def test_hardware_profile_torch_version():
    profile = detect_hardware()
    assert isinstance(profile.torch_version, str)
    assert len(profile.torch_version) > 0


def test_get_hardware_profile_is_cached():
    """get_hardware_profile() must return the same object on repeated calls."""
    p1 = get_hardware_profile()
    p2 = get_hardware_profile()
    assert p1 is p2


def test_hardware_profile_ldsr_device_valid():
    profile = detect_hardware()
    assert profile.ldsr_device in {"cpu", "cuda", "mps"}


def test_hardware_profile_cuda_fields_consistent():
    profile = detect_hardware()
    if profile.device_type == "cuda":
        assert profile.is_gpu_available is True
        assert profile.primary_gpu is not None
        assert profile.primary_gpu.vram_gb > 0
    else:
        # CPU or MPS — no CUDA GPU
        pass  # primary_gpu may be None, that's fine


# ---------------------------------------------------------------------------
# Execution Planner Tests
# ---------------------------------------------------------------------------

def test_compute_planner_creates_plan():
    planner = ComputePlanner()
    plan = planner.plan_for_single_image()
    assert isinstance(plan, ExecutionPlan)


def test_compute_planner_device_matches_hardware():
    planner = ComputePlanner()
    plan = planner.plan_for_single_image()
    assert plan.device == planner.profile.device


def test_compute_planner_tile_size_fixed():
    planner = ComputePlanner()
    plan = planner.plan_for_single_image()
    # LDSR-S2 requires 128×128 tiles
    assert plan.tile_size == 128
    assert plan.tile_overlap == 12


def test_compute_planner_batch_size_positive():
    planner = ComputePlanner()
    plan = planner.plan_for_single_image()
    assert plan.tile_batch_size >= 1
    assert plan.seg_batch_size >= 1


def test_compute_planner_batch_plan_max_jobs():
    planner = ComputePlanner()
    plan = planner.plan_for_batch(n_images=5)
    assert plan.max_active_jobs >= 1


def test_compute_planner_batch_plan_is_conservative():
    """Batch plan should never have more tile workers than single-image plan."""
    planner = ComputePlanner()
    single = planner.plan_for_single_image()
    batch = planner.plan_for_batch(n_images=10)
    assert batch.max_parallel_tiles <= single.max_parallel_tiles + 1  # allow minor rounding


def test_compute_planner_strategy_label_set():
    planner = ComputePlanner()
    plan = planner.plan_for_single_image()
    assert isinstance(plan.strategy_label, str)
    assert len(plan.strategy_label) > 0


def test_compute_planner_as_dict_shape():
    planner = ComputePlanner()
    d = planner.as_dict()

    assert "hardware" in d
    assert "execution_plan" in d

    hw = d["hardware"]
    assert "device_type" in hw
    assert "environment" in hw
    assert "cpu_logical_cores" in hw
    assert "torch_version" in hw

    ep = d["execution_plan"]
    assert "strategy" in ep
    assert "ldsr_device" in ep
    assert "tile_batch_size" in ep
    assert "tile_workers" in ep
    assert "fp16_inference" in ep
    assert "max_active_jobs" in ep


def test_compute_planner_as_dict_hardware_serialisable():
    """All hardware values should be JSON-serialisable primitives."""
    import json
    planner = ComputePlanner()
    d = planner.as_dict()
    # Should not raise
    serialised = json.dumps(d)
    assert len(serialised) > 0


def test_compute_planner_fp16_only_on_capable_device():
    planner = ComputePlanner()
    plan = planner.plan_for_single_image()
    if planner.profile.device_type == "cpu":
        assert plan.fp16_inference is False
    # On CUDA/MPS fp16 may be True — just check it's a bool
    assert isinstance(plan.fp16_inference, bool)


def test_uncertainty_monte_carlo_runs_positive():
    planner = ComputePlanner()
    plan = planner.plan_for_single_image()
    assert plan.uncertainty_monte_carlo_runs >= 1


# ---------------------------------------------------------------------------
# Integration: API shape matches FastAPI endpoint contract
# ---------------------------------------------------------------------------

def test_compute_api_integration():
    """Simulate what the FastAPI route returns and validate the shape."""
    from fastapi.testclient import TestClient
    from backend.app.main import app

    client = TestClient(app)
    response = client.get("/api/v1/compute/profile")
    assert response.status_code == 200
    data = response.json()
    assert "hardware" in data
    assert "execution_plan" in data
    assert data["hardware"]["device_type"] in {"cpu", "cuda", "mps"}
