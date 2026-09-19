"""
PixelSight Scientific Reporting Subsystem
=========================================
Exports publication-quality visual diagnostics and multi-format reports.
"""

from evaluation.reporting.visualizer import (
    plot_master_ten_panel,
    plot_quintile_error_curve,
)
from evaluation.reporting.reporter import MasterReporter

__all__ = [
    "plot_master_ten_panel",
    "plot_quintile_error_curve",
    "MasterReporter",
]
