"""
PixelSight Adaptive Compute Engine

Provides hardware detection, resource profiling, and adaptive execution planning
so the framework automatically scales its inference workload to the available hardware.
"""
from backend.app.services.compute.hardware import HardwareProfile, detect_hardware
from backend.app.services.compute.planner import ComputePlanner, ExecutionPlan

__all__ = [
    "HardwareProfile",
    "ComputePlanner",
    "ExecutionPlan",
    "detect_hardware",
]
