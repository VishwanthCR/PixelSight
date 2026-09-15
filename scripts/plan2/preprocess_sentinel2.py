from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


ROOT = Path("dataset/plan2/sentinel2")
OUTPUT = Path("dataset/plan2/patches")

PATCH_SIZE = 128
STRIDE = 128

BANDS = ("B02", "B03", "B04", "B08")

# Sentinel-2 SCL classes retained as valid for this experiment:
#
# 2  = Dark Area Pixels
# 4  = Vegetation
# 5  = Bare Soils
# 6  = Water
#
# Excluded:
# 0  = No data
# 1  = Saturated / defective
# 3  = Cloud shadow
# 7  = Cloud low probability / Unclassified
# 8  = Cloud medium probability
# 9  = Cloud high probability
# 10 = Thin cirrus
# 11 = Snow / ice
VALID_SCL = {2, 4, 5, 6}

# A patch must contain at least this fraction of valid pixels.
MIN_VALID_FRACTION = 0.90


def find_band(split, band):
    matches = list(
        ROOT.joinpath(split).glob(f"*_{band}_10m.jp2")
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {band} file for {split}, "
            f"found {len(matches)}"
        )

    return matches[0]


def find_scl(split):
    matches = list(
        ROOT.joinpath(split).glob("*_SCL_20m.jp2")
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one SCL file for {split}, "
            f"found {len(matches)}"
        )

    return matches[0]


def create_mask(scl):
    """
    Convert the native 20 m SCL image to a 10 m validity mask.

    Each 20 m SCL pixel corresponds to a 2x2 block of 10 m pixels.
    Nearest-neighbour replication is therefore used.
    """

    scl_10m = np.repeat(
        np.repeat(scl, 2, axis=0),
        2,
        axis=1,
    )

    valid = np.isin(
        scl_10m,
        list(VALID_SCL),
    )

    return valid


def read_scene(split):
    print(f"\n=== {split.upper()} ===")

    band_paths = {
        band: find_band(split, band)
        for band in BANDS
    }

    scl_path = find_scl(split)

    # B02 is used as the spatial reference.
    with rasterio.open(band_paths["B02"]) as ref:
        height = ref.height
        width = ref.width
        transform = ref.transform
        crs = ref.crs

    image = np.empty(
        (height, width, len(BANDS)),
        dtype=np.float32,
    )

    # Read and normalize the four 10 m bands.
    for i, band in enumerate(BANDS):

        with rasterio.open(band_paths[band]) as src:
            data = src.read(1).astype(np.float32)

            # Sentinel-2 L2A reflectance scaling.
            data /= 10000.0

            # Keep reflectance within [0, 1].
            data = np.clip(data, 0.0, 1.0)

            image[:, :, i] = data

    # Read SCL at native 20 m resolution.
    with rasterio.open(scl_path) as src:
        scl = src.read(1)

    # Convert SCL to a 10 m validity mask.
    valid_mask = create_mask(scl)

    if valid_mask.shape != (height, width):
        raise RuntimeError(
            f"SCL mask shape {valid_mask.shape} does not match "
            f"10 m image shape {(height, width)}"
        )

    valid_percentage = valid_mask.mean() * 100.0

    print(f"Image shape: {image.shape}")
    print(f"CRS: {crs}")
    print(f"Valid pixels: {valid_percentage:.2f}%")

    return image, valid_mask, transform, crs


def save_patches(split):
    image, mask, transform, crs = read_scene(split)

    output_dir = OUTPUT / split
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    height, width, _ = image.shape

    total_candidates = 0
    saved = 0
    skipped_quality = 0

    for y in range(
        0,
        height - PATCH_SIZE + 1,
        STRIDE,
    ):
        for x in range(
            0,
            width - PATCH_SIZE + 1,
            STRIDE,
        ):

            total_candidates += 1

            patch = image[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE,
                :
            ]

            patch_mask = mask[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE,
            ]

            valid_fraction = float(
                patch_mask.mean()
            )

            # Reject patches with too many invalid pixels.
            if valid_fraction < MIN_VALID_FRACTION:
                skipped_quality += 1
                continue

            patch_transform = rasterio.windows.transform(
                Window(
                    x,
                    y,
                    PATCH_SIZE,
                    PATCH_SIZE,
                ),
                transform,
            )

            output_path = (
                output_dir /
                f"patch_{saved:05d}.npz"
            )

            np.savez_compressed(
                output_path,
                image=patch.astype(np.float32),
                mask=patch_mask,
                valid_fraction=np.float32(valid_fraction),
                transform=np.array(patch_transform),
                crs=str(crs),
            )

            saved += 1

    print(f"Candidate patches : {total_candidates}")
    print(f"Saved patches     : {saved}")
    print(f"Rejected (<90%)   : {skipped_quality}")


def main():
    print("PixelSight Plan 2 preprocessing")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Stride: {STRIDE}")
    print(f"Minimum valid fraction: {MIN_VALID_FRACTION:.0%}")
    print(f"Valid SCL classes: {sorted(VALID_SCL)}")

    for split in (
        "train",
        "validation",
        "test",
    ):
        save_patches(split)

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()