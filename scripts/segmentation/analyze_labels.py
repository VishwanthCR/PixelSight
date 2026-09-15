"""
PixelSight - Segmentation Label Analysis
=========================================

Analyze WorldCover class distributions for the Sentinel-2 scenes
prepared for segmentation.

This script does NOT modify the labels.

It reports:
    - pixel counts
    - percentages
    - classes present
    - classes absent
    - nodata pixels

This allows us to decide the final segmentation class mapping
using the actual dataset rather than assumptions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio


WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


def analyze_label(label_path: Path) -> None:

    if not label_path.exists():
        raise FileNotFoundError(
            f"Label file not found:\n{label_path}"
        )

    print("=" * 70)
    print("PixelSight Segmentation")
    print("WorldCover Label Analysis")
    print("=" * 70)

    print(f"\nInput:")
    print(f"  {label_path}")

    with rasterio.open(label_path) as src:

        labels = src.read(1)

        print("\nRaster metadata")
        print("----------------")
        print(f"Size       : {src.width} × {src.height}")
        print(f"CRS        : {src.crs}")
        print(f"Resolution : {src.res}")
        print(f"Bounds     : {src.bounds}")
        print(f"NoData     : {src.nodata}")

    total_pixels = labels.size

    print("\nClass distribution")
    print("------------------")

    print(
        f"{'ID':>5}  "
        f"{'Class':<28}  "
        f"{'Pixels':>15}  "
        f"{'Percent':>10}"
    )

    print("-" * 70)

    for class_id, class_name in WORLDCOVER_CLASSES.items():

        count = int(
            np.count_nonzero(
                labels == class_id
            )
        )

        percentage = (
            count / total_pixels * 100
        )

        print(
            f"{class_id:>5}  "
            f"{class_name:<28}  "
            f"{count:>15,}  "
            f"{percentage:>9.4f}%"
        )

    nodata_count = int(
        np.count_nonzero(labels == 0)
    )

    print("-" * 70)

    print(
        f"{'0':>5}  "
        f"{'NoData / outside coverage':<28}  "
        f"{nodata_count:>15,}  "
        f"{nodata_count / total_pixels * 100:>9.4f}%"
    )

    unique_values = np.unique(labels)

    print("\nUnique values")
    print("-------------")
    print(unique_values.tolist())

    unexpected = [
        value
        for value in unique_values
        if value != 0
        and value not in WORLDCOVER_CLASSES
    ]

    if unexpected:
        print("\nWARNING")
        print("-------")
        print(
            "Unexpected class values found:"
        )
        print(unexpected)

        raise ValueError(
            "Unexpected label values detected."
        )

    present_classes = [
        class_id
        for class_id in WORLDCOVER_CLASSES
        if np.any(labels == class_id)
    ]

    absent_classes = [
        class_id
        for class_id in WORLDCOVER_CLASSES
        if not np.any(labels == class_id)
    ]

    print("\nClasses present")
    print("----------------")

    for class_id in present_classes:
        print(
            f"{class_id:>3} : "
            f"{WORLDCOVER_CLASSES[class_id]}"
        )

    print("\nClasses absent")
    print("--------------")

    for class_id in absent_classes:
        print(
            f"{class_id:>3} : "
            f"{WORLDCOVER_CLASSES[class_id]}"
        )

    print("\nAnalysis completed successfully.")


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze a PixelSight aligned "
            "WorldCover label GeoTIFF."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Aligned WorldCover label GeoTIFF.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    analyze_label(
        args.input
    )


if __name__ == "__main__":
    main()