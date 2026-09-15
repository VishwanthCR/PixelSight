"""
Prepare Sentinel-2 data for PixelSight Plan 2.

This script is intentionally a scaffold.

It does NOT download Sentinel-2 data.
Raw imagery must be acquired separately and kept outside Git.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src")
)

from pixelsight.data.plan2.sentinel2 import (
    REQUIRED_BANDS,
    validate_bands,
)


def main():

    print("=" * 60)
    print("PixelSight Plan 2 — Sentinel-2 Preparation")
    print("=" * 60)

    print()
    print("Required bands:")

    for band in REQUIRED_BANDS:
        print(f"  - {band}")

    print()
    print("Expected native resolution: 10 m")
    print("Proposed LR tile: 128 × 128 × 4")
    print("Proposed SR tile: 512 × 512 × 4")

    print()
    print("No data processing performed yet.")
    print("Acquire and verify Sentinel-2 L2A scenes first.")


if __name__ == "__main__":
    main()