"""
PixelSight - Segmentation Dataset Preparation
==============================================

Phase 2:
    Download the required ESA WorldCover 2021 v200 tiles,
    mosaic them when necessary, reproject the land-cover labels
    onto the Sentinel-2 grid, and validate exact alignment.

The actual segmentation patches will be created in a later phase.

WorldCover 2021 v200:
    - 10 m global land-cover product
    - 11 original classes
    - 3° x 3° COG tiles
    - EPSG:4326
    - Public AWS Open Data bucket

Official source:
    https://esa-worldcover.org/en/data-access
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import boto3
import numpy as np
import rasterio
from botocore import UNSIGNED
from botocore.client import Config
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, transform_bounds


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

WORLDCOVER_YEAR = 2021
WORLDCOVER_VERSION = "v200"

WORLDCOVER_BUCKET = "esa-worldcover"
WORLDCOVER_PREFIX = "v200/2021/map"

TILE_SIZE_DEG = 3

# WorldCover classes
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


# ---------------------------------------------------------------------
# Tile helpers
# ---------------------------------------------------------------------

def floor_tile_coordinate(value: float) -> int:
    """Return the lower 3-degree grid coordinate."""

    return math.floor(value / TILE_SIZE_DEG) * TILE_SIZE_DEG


def format_latitude(latitude: int) -> str:
    if latitude >= 0:
        return f"N{latitude:02d}"

    return f"S{abs(latitude):02d}"


def format_longitude(longitude: int) -> str:
    if longitude >= 0:
        return f"E{longitude:03d}"

    return f"W{abs(longitude):03d}"


def worldcover_tile_name(
    latitude: int,
    longitude: int,
) -> str:
    return (
        f"{format_latitude(latitude)}"
        f"{format_longitude(longitude)}"
    )


def get_required_tiles(
    west: float,
    south: float,
    east: float,
    north: float,
) -> list[str]:
    """
    Determine every 3° x 3° WorldCover tile intersecting
    a WGS84 bounding box.
    """

    min_lon = floor_tile_coordinate(west)
    max_lon = floor_tile_coordinate(east - 1e-10)

    min_lat = floor_tile_coordinate(south)
    max_lat = floor_tile_coordinate(north - 1e-10)

    tiles = []

    latitude = min_lat

    while latitude <= max_lat:
        longitude = min_lon

        while longitude <= max_lon:
            tiles.append(
                worldcover_tile_name(
                    latitude,
                    longitude,
                )
            )

            longitude += TILE_SIZE_DEG

        latitude += TILE_SIZE_DEG

    return tiles


# ---------------------------------------------------------------------
# AWS WorldCover access
# ---------------------------------------------------------------------

def create_unsigned_s3_client():
    """
    Create an anonymous S3 client.

    WorldCover is hosted in a public AWS Open Data bucket.
    """

    return boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED),
        region_name="eu-central-1",
    )


def find_worldcover_object(
    s3,
    tile_name: str,
) -> str:
    """
    Find the exact WorldCover Map.tif object for a tile.

    We use prefix discovery instead of assuming the exact filename,
    making the downloader more robust to filename formatting.
    """

    prefix = (
        f"{WORLDCOVER_PREFIX}/"
        f"ESA_WorldCover_10m_2021_v200_{tile_name}"
    )

    print(f"\nSearching AWS for:")
    print(f"  {prefix}")

    response = s3.list_objects_v2(
        Bucket=WORLDCOVER_BUCKET,
        Prefix=prefix,
    )

    contents = response.get("Contents", [])

    candidates = [
        obj["Key"]
        for obj in contents
        if obj["Key"].endswith("_Map.tif")
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Could not find WorldCover Map.tif for tile "
            f"{tile_name}.\n"
            f"Searched prefix:\n{prefix}"
        )

    if len(candidates) > 1:
        print("Multiple candidates found:")

        for candidate in candidates:
            print(f"  {candidate}")

        # Prefer the exact expected naming pattern.
        exact = [
            key
            for key in candidates
            if key.endswith(
                f"{tile_name}_Map.tif"
            )
        ]

        if exact:
            return exact[0]

    return candidates[0]


def download_worldcover_tile(
    s3,
    tile_name: str,
    output_dir: Path,
) -> Path:
    """
    Download one WorldCover tile if it is not already present.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir /
        f"ESA_WorldCover_10m_2021_v200_{tile_name}_Map.tif"
    )

    if output_path.exists():
        print(
            f"\nAlready downloaded:"
            f"\n  {output_path}"
        )

        return output_path

    key = find_worldcover_object(
        s3,
        tile_name,
    )

    print("\nDownloading WorldCover tile")
    print("---------------------------")
    print(f"Tile : {tile_name}")
    print(f"S3   : s3://{WORLDCOVER_BUCKET}/{key}")
    print(f"To   : {output_path}")

    s3.download_file(
        WORLDCOVER_BUCKET,
        key,
        str(output_path),
    )

    if not output_path.exists():
        raise RuntimeError(
            f"Download reported success but file does not exist:\n"
            f"{output_path}"
        )

    size_mb = output_path.stat().st_size / (1024 ** 2)

    print(
        f"Downloaded successfully "
        f"({size_mb:.2f} MB)."
    )

    return output_path


# ---------------------------------------------------------------------
# Sentinel-2 inspection
# ---------------------------------------------------------------------

def get_scene_info(
    input_path: Path,
) -> dict:
    """
    Read Sentinel-2 GeoTIFF metadata and convert its bounds to WGS84.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input GeoTIFF does not exist:\n{input_path}"
        )

    with rasterio.open(input_path) as src:

        if src.crs is None:
            raise ValueError(
                "Input GeoTIFF has no CRS."
            )

        west, south, east, north = transform_bounds(
            src.crs,
            "EPSG:4326",
            src.bounds.left,
            src.bounds.bottom,
            src.bounds.right,
            src.bounds.top,
            densify_pts=21,
        )

        return {
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "resolution": src.res,
            "wgs84_bounds": (
                west,
                south,
                east,
                north,
            ),
        }


# ---------------------------------------------------------------------
# WorldCover mosaic
# ---------------------------------------------------------------------

def create_worldcover_mosaic(
    tile_paths: list[Path],
    output_path: Path,
) -> Path:
    """
    Mosaic WorldCover tiles into one temporary/reference raster.

    The source tiles are categorical labels, so no interpolation
    occurs during the mosaic operation.
    """

    print("\nMosaicking WorldCover tiles")
    print("---------------------------")

    datasets = []

    try:
        for tile_path in tile_paths:
            print(f"Opening: {tile_path}")

            datasets.append(
                rasterio.open(tile_path)
            )

        mosaic, transform = merge(
            datasets,
            method="first",
        )

        profile = datasets[0].profile.copy()

        profile.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "count": 1,
                "dtype": mosaic.dtype,
                "compress": "deflate",
                "predictor": 2,
            }
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as dst:
            dst.write(
                mosaic[0],
                1,
            )

    finally:
        for dataset in datasets:
            dataset.close()

    print(f"Mosaic saved:")
    print(f"  {output_path}")

    return output_path


# ---------------------------------------------------------------------
# Reprojection / alignment
# ---------------------------------------------------------------------

def align_worldcover_to_sentinel2(
    sentinel_path: Path,
    worldcover_path: Path,
    output_path: Path,
) -> None:
    """
    Reproject WorldCover onto the exact Sentinel-2 raster grid.

    IMPORTANT:
        WorldCover is categorical data.
        Therefore nearest-neighbour resampling is mandatory.
    """

    print("\nAligning WorldCover to Sentinel-2")
    print("---------------------------------")

    with rasterio.open(sentinel_path) as s2:

        target_crs = s2.crs
        target_transform = s2.transform
        target_width = s2.width
        target_height = s2.height

        with rasterio.open(worldcover_path) as wc:

            print(f"Source CRS : {wc.crs}")
            print(f"Target CRS : {target_crs}")

            print(
                f"Target size: "
                f"{target_width} × {target_height}"
            )

            aligned = np.zeros(
                (
                    target_height,
                    target_width,
                ),
                dtype=np.uint8,
            )

            reproject(
                source=rasterio.band(wc, 1),
                destination=aligned,
                src_transform=wc.transform,
                src_crs=wc.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest,
                src_nodata=wc.nodata,
                dst_nodata=0,
            )

            profile = s2.profile.copy()

            profile.update(
                {
                    "driver": "GTiff",
                    "height": target_height,
                    "width": target_width,
                    "count": 1,
                    "dtype": "uint8",
                    "crs": target_crs,
                    "transform": target_transform,
                    "nodata": 0,
                    "compress": "deflate",
                    "predictor": 2,
                    "BIGTIFF": "IF_SAFER",
                }
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with rasterio.open(
                output_path,
                "w",
                **profile,
            ) as dst:

                dst.write(
                    aligned,
                    1,
                )

                dst.set_band_description(
                    1,
                    "WorldCover 2021 v200",
                )


# ---------------------------------------------------------------------
# Alignment validation
# ---------------------------------------------------------------------

def validate_alignment(
    sentinel_path: Path,
    label_path: Path,
) -> None:
    """
    Verify that the aligned label and Sentinel-2 image have exactly
    matching spatial dimensions, CRS, transform, and bounds.
    """

    print("\nAlignment validation")
    print("--------------------")

    with rasterio.open(sentinel_path) as s2:
        with rasterio.open(label_path) as label:

            checks = {}

            checks["width"] = (
                s2.width == label.width
            )

            checks["height"] = (
                s2.height == label.height
            )

            checks["crs"] = (
                s2.crs == label.crs
            )

            checks["transform"] = (
                s2.transform == label.transform
            )

            checks["bounds"] = (
                s2.bounds == label.bounds
            )

            checks["resolution"] = (
                s2.res == label.res
            )

            for name, passed in checks.items():
                print(
                    f"{name.capitalize():<12}: "
                    f"{'PASS' if passed else 'FAIL'}"
                )

            if not all(checks.values()):
                raise RuntimeError(
                    "WorldCover/Sentinel-2 alignment "
                    "validation FAILED."
                )

            labels = label.read(1)

            unique_values = np.unique(
                labels
            )

            print(
                "\nUnique WorldCover values "
                "in aligned label:"
            )

            print(
                "  ",
                unique_values.tolist(),
            )

            invalid_values = [
                value
                for value in unique_values
                if value != 0
                and value not in WORLDCOVER_CLASSES
            ]

            if invalid_values:
                raise RuntimeError(
                    "Unexpected WorldCover class values found: "
                    f"{invalid_values}"
                )

            print(
                "\nAlignment validation: PASS"
            )


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def prepare_scene(
    input_path: Path,
) -> None:

    print("=" * 60)
    print("PixelSight Segmentation")
    print("WorldCover 2021 v200 Preparation")
    print("=" * 60)

    print(f"\nInput:")
    print(f"  {input_path}")

    # -------------------------------------------------------------
    # 1. Inspect Sentinel-2
    # -------------------------------------------------------------

    scene = get_scene_info(
        input_path
    )

    print("\nSentinel-2 metadata")
    print("-------------------")
    print(
        f"Size       : "
        f"{scene['width']} × {scene['height']}"
    )
    print(
        f"Bands      : "
        f"{scene['count']}"
    )
    print(
        f"CRS        : "
        f"{scene['crs']}"
    )
    print(
        f"Resolution : "
        f"{scene['resolution']}"
    )
    print(
        f"Bounds     : "
        f"{scene['bounds']}"
    )

    west, south, east, north = scene[
        "wgs84_bounds"
    ]

    print("\nWGS84 bounds")
    print("------------")
    print(f"West  : {west:.8f}")
    print(f"South : {south:.8f}")
    print(f"East  : {east:.8f}")
    print(f"North : {north:.8f}")

    # -------------------------------------------------------------
    # 2. Determine WorldCover tiles
    # -------------------------------------------------------------

    tiles = get_required_tiles(
        west,
        south,
        east,
        north,
    )

    print("\nRequired WorldCover tiles")
    print("-------------------------")

    for tile in tiles:
        print(f"  {tile}")

    print(
        f"\nTotal tiles: {len(tiles)}"
    )

    # -------------------------------------------------------------
    # 3. Local directories
    # -------------------------------------------------------------

    root = Path(
        "dataset/segmentation"
    )

    worldcover_dir = (
        root / "worldcover"
    )

    aligned_dir = (
        root / "aligned_labels"
    )

    mosaic_dir = (
        root / "mosaics"
    )

    # -------------------------------------------------------------
    # 4. Download WorldCover
    # -------------------------------------------------------------

    print("\nConnecting to public WorldCover AWS bucket...")

    s3 = create_unsigned_s3_client()

    tile_paths = []

    for tile in tiles:

        tile_path = download_worldcover_tile(
            s3=s3,
            tile_name=tile,
            output_dir=worldcover_dir,
        )

        tile_paths.append(
            tile_path
        )

    # -------------------------------------------------------------
    # 5. Mosaic
    # -------------------------------------------------------------

    scene_name = input_path.stem

    mosaic_path = (
        mosaic_dir /
        f"{scene_name}_worldcover_mosaic.tif"
    )

    create_worldcover_mosaic(
        tile_paths=tile_paths,
        output_path=mosaic_path,
    )

    # -------------------------------------------------------------
    # 6. Align to Sentinel-2
    # -------------------------------------------------------------

    aligned_path = (
        aligned_dir /
        f"{scene_name}_worldcover_aligned.tif"
    )

    align_worldcover_to_sentinel2(
        sentinel_path=input_path,
        worldcover_path=mosaic_path,
        output_path=aligned_path,
    )

    # -------------------------------------------------------------
    # 7. Validate
    # -------------------------------------------------------------

    validate_alignment(
        sentinel_path=input_path,
        label_path=aligned_path,
    )

    # -------------------------------------------------------------
    # 8. Final summary
    # -------------------------------------------------------------

    print("\n" + "=" * 60)
    print("WorldCover preparation completed successfully.")
    print("=" * 60)

    print("\nInput:")
    print(f"  {input_path}")

    print("\nWorldCover tiles:")
    for tile_path in tile_paths:
        print(f"  {tile_path}")

    print("\nMosaic:")
    print(f"  {mosaic_path}")

    print("\nAligned label:")
    print(f"  {aligned_path}")

    print(
        "\nThe aligned label is now on the exact same "
        "pixel grid as the Sentinel-2 image."
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Prepare ESA WorldCover 2021 v200 labels "
            "for PixelSight segmentation."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Input 4-band Sentinel-2 GeoTIFF."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    prepare_scene(
        args.input
    )


if __name__ == "__main__":
    main()