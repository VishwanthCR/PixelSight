"""
PixelSight Segmentation
Create aligned Sentinel-2 image/label patches.

Input:
    Sentinel-2 4-band GeoTIFF
    Aligned WorldCover label GeoTIFF

Output:
    image patches: float32, shape (4, patch_size, patch_size)
    label patches: uint8, shape (patch_size, patch_size)

WorldCover → PixelSight mapping:
    10  -> 0  Tree
    20  -> 1  Shrubland
    30  -> 2  Grassland
    40  -> 3  Cropland
    50  -> 4  Built-up
    60  -> 5  Bare
    80  -> 6  Water
    90  -> 255 Ignore
    95  -> 255 Ignore
    70  -> 255 Ignore
    100 -> 255 Ignore
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm


WORLD_COVER_TO_PIXEL_SIGHT = {
    10: 0,    # Tree
    20: 1,    # Shrubland
    30: 2,    # Grassland
    40: 3,    # Cropland
    50: 4,    # Built-up
    60: 5,    # Bare
    80: 6,    # Water
    70: 255,  # Snow/ice
    90: 255,  # Wetland
    95: 255,  # Mangroves
    100: 255, # Moss/lichen
}


CLASS_NAMES = {
    0: "Tree",
    1: "Shrubland",
    2: "Grassland",
    3: "Cropland",
    4: "Built-up",
    5: "Bare",
    6: "Water",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create aligned PixelSight segmentation patches."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Input Sentinel-2 4-band GeoTIFF.",
    )

    parser.add_argument(
        "--label",
        required=True,
        help="Aligned WorldCover label GeoTIFF.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for patches.",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Patch size in pixels. Default: 128.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Patch stride in pixels. Default: 128.",
    )

    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.90,
        help="Minimum fraction of non-ignore pixels required. Default: 0.90.",
    )

    return parser.parse_args()


def validate_inputs(image_path: Path, label_path: Path):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not label_path.exists():
        raise FileNotFoundError(f"Label not found: {label_path}")

    with rasterio.open(image_path) as image_src:
        if image_src.count != 4:
            raise ValueError(
                f"Expected 4 image bands, found {image_src.count}."
            )

        image_shape = (image_src.height, image_src.width)
        image_crs = image_src.crs
        image_transform = image_src.transform

    with rasterio.open(label_path) as label_src:
        label_shape = (label_src.height, label_src.width)
        label_crs = label_src.crs
        label_transform = label_src.transform

    if image_shape != label_shape:
        raise ValueError(
            f"Image/label shape mismatch: "
            f"{image_shape} vs {label_shape}"
        )

    if image_crs != label_crs:
        raise ValueError(
            f"Image/label CRS mismatch: "
            f"{image_crs} vs {label_crs}"
        )

    if image_transform != label_transform:
        raise ValueError("Image/label transform mismatch.")

    print("Input validation: PASS")
    print(f"  Image size : {image_shape[1]} × {image_shape[0]}")
    print(f"  Bands      : 4")
    print(f"  CRS        : {image_crs}")
    print(f"  Resolution : {image_transform.a}, {abs(image_transform.e)}")


def map_labels(label_patch: np.ndarray) -> np.ndarray:
    mapped = np.full(
        label_patch.shape,
        255,
        dtype=np.uint8,
    )

    for worldcover_id, pixelsight_id in WORLD_COVER_TO_PIXEL_SIGHT.items():
        mapped[label_patch == worldcover_id] = pixelsight_id

    return mapped


def normalize_image(image_patch: np.ndarray) -> np.ndarray:
    """
    Convert Sentinel-2 reflectance values to approximately [0, 1].

    Sentinel-2 L2A imagery is commonly stored using a scale factor
    around 10000. Values are clipped after scaling.
    """

    image_patch = image_patch.astype(np.float32)

    # Handle integer Sentinel-2 reflectance representation.
    if np.nanmax(image_patch) > 1.5:
        image_patch /= 10000.0

    image_patch = np.clip(image_patch, 0.0, 1.0)

    return image_patch.astype(np.float32)


def main():
    args = parse_args()

    image_path = Path(args.image)
    label_path = Path(args.label)
    output_dir = Path(args.output)

    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive.")

    if args.stride <= 0:
        raise ValueError("--stride must be positive.")

    if not 0.0 <= args.min_valid_fraction <= 1.0:
        raise ValueError("--min-valid-fraction must be between 0 and 1.")

    print("=" * 70)
    print("PixelSight Segmentation")
    print("Patch Creation")
    print("=" * 70)

    print("\nImage:")
    print(f"  {image_path}")

    print("\nLabel:")
    print(f"  {label_path}")

    print("\nPatch configuration:")
    print(f"  Patch size         : {args.patch_size}")
    print(f"  Stride             : {args.stride}")
    print(f"  Minimum valid frac : {args.min_valid_fraction}")

    validate_inputs(image_path, label_path)

    output_images = output_dir / "images"
    output_labels = output_dir / "labels"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    class_pixel_counts = {
        class_id: 0
        for class_id in CLASS_NAMES
    }

    total_patches = 0
    saved_patches = 0
    rejected_patches = 0

    with rasterio.open(image_path) as image_src, \
            rasterio.open(label_path) as label_src:

        height = image_src.height
        width = image_src.width

        row_positions = range(
            0,
            height - args.patch_size + 1,
            args.stride,
        )

        col_positions = range(
            0,
            width - args.patch_size + 1,
            args.stride,
        )

        positions = [
            (row, col)
            for row in row_positions
            for col in col_positions
        ]

        print(f"\nCandidate patches: {len(positions):,}")

        for row, col in tqdm(
            positions,
            desc="Creating patches",
            unit="patch",
        ):
            total_patches += 1

            window = rasterio.windows.Window(
                col_off=col,
                row_off=row,
                width=args.patch_size,
                height=args.patch_size,
            )

            image_patch = image_src.read(
                indexes=[1, 2, 3, 4],
                window=window,
            )

            label_patch = label_src.read(
                1,
                window=window,
            )

            mapped_label = map_labels(label_patch)

            valid_fraction = np.mean(mapped_label != 255)

            if valid_fraction < args.min_valid_fraction:
                rejected_patches += 1
                continue

            image_patch = normalize_image(image_patch)

            if not np.isfinite(image_patch).all():
                rejected_patches += 1
                continue

            patch_id = f"r{row:05d}_c{col:05d}"

            image_file = output_images / f"{patch_id}.npy"
            label_file = output_labels / f"{patch_id}.npy"

            np.save(image_file, image_patch)
            np.save(label_file, mapped_label)

            saved_patches += 1

            for class_id in CLASS_NAMES:
                class_pixel_counts[class_id] += int(
                    np.sum(mapped_label == class_id)
                )

    print("\n" + "=" * 70)
    print("Patch creation summary")
    print("=" * 70)

    print(f"Total candidate patches : {total_patches:,}")
    print(f"Saved patches            : {saved_patches:,}")
    print(f"Rejected patches         : {rejected_patches:,}")

    if total_patches > 0:
        print(
            f"Retention rate           : "
            f"{saved_patches / total_patches * 100:.2f}%"
        )

    print("\nSaved class distribution:")
    print("-" * 55)

    total_valid_pixels = sum(class_pixel_counts.values())

    for class_id, class_name in CLASS_NAMES.items():
        count = class_pixel_counts[class_id]

        if total_valid_pixels > 0:
            percentage = (
                count / total_valid_pixels * 100
            )
        else:
            percentage = 0.0

        print(
            f"{class_id:>3}  "
            f"{class_name:<15} "
            f"{count:>12,} "
            f"{percentage:>8.4f}%"
        )

    print("\nOutput:")
    print(f"  Images: {output_images}")
    print(f"  Labels: {output_labels}")

    print("\nPatch creation completed successfully.")


if __name__ == "__main__":
    main()