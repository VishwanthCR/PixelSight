from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine


PATCH_FILE = Path(
    "dataset/plan2/patches/train/patch_00000.npz"
)

SR_FILE = Path(
    "dataset/plan2/ldsr_test/patch_00000_sr.npy"
)

OUTPUT_DIR = Path(
    "dataset/plan2/ldsr_output"
)


def normalize_rgb(rgb):
    rgb = rgb.astype(np.float32)

    low = np.percentile(rgb, 2)
    high = np.percentile(rgb, 98)

    rgb = (rgb - low) / (high - low + 1e-8)

    return np.clip(rgb, 0, 1)


def main():

    print("=== Save LDSR-S2 2.5 m Image ===")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load SR output
    # ---------------------------------------------------------

    sr = np.load(SR_FILE).astype(np.float32)

    print(f"SR shape: {sr.shape}")

    if sr.shape != (512, 512, 4):
        raise RuntimeError(
            f"Expected (512,512,4), got {sr.shape}"
        )

    # ---------------------------------------------------------
    # Load geospatial metadata
    # ---------------------------------------------------------

    with np.load(
        PATCH_FILE,
        allow_pickle=True,
    ) as data:

        transform_data = data["transform"]
        crs_data = data["crs"]

    # Handle transform saved as tuple/list/object.
    transform_values = np.asarray(
        transform_data
    ).flatten()

    if len(transform_values) != 9:
        raise RuntimeError(
            f"Unexpected transform: "
            f"{transform_data}"
        )

    transform = Affine(*transform_values[:6])

    crs = str(crs_data)

    print(f"CRS: {crs}")
    print(f"Input transform: {transform}")

    # ---------------------------------------------------------
    # 10 m -> 2.5 m transform
    # ---------------------------------------------------------

    sr_transform = Affine(
        transform.a / 4,
        transform.b,
        transform.c,
        transform.d,
        transform.e / 4,
        transform.f,
    )

    print(
        f"SR transform: {sr_transform}"
    )

    # ---------------------------------------------------------
    # Save 4-band 2.5 m GeoTIFF
    # ---------------------------------------------------------

    geotiff_path = (
        OUTPUT_DIR /
        "patch_00000_ldsr_s2_2p5m.tif"
    )

    with rasterio.open(
        geotiff_path,
        "w",
        driver="GTiff",
        height=512,
        width=512,
        count=4,
        dtype="float32",
        crs=crs,
        transform=sr_transform,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:

        for band in range(4):
            dst.write(
                sr[:, :, band],
                band + 1,
            )

        dst.set_band_description(1, "B02")
        dst.set_band_description(2, "B03")
        dst.set_band_description(3, "B04")
        dst.set_band_description(4, "B08")

    print(
        f"Saved GeoTIFF:\n{geotiff_path}"
    )

    # ---------------------------------------------------------
    # Create RGB PNG
    # ---------------------------------------------------------

    # B04 = Red
    # B03 = Green
    # B02 = Blue

    rgb = sr[:, :, [2, 1, 0]]

    rgb = normalize_rgb(rgb)

    rgb_uint8 = (
        rgb * 255
    ).round().astype(np.uint8)

    png_path = (
        OUTPUT_DIR /
        "patch_00000_ldsr_s2_rgb.png"
    )

    from PIL import Image

    Image.fromarray(
        rgb_uint8
    ).save(png_path)

    print(
        f"Saved RGB image:\n{png_path}"
    )

    # ---------------------------------------------------------
    # NIR false color PNG
    # ---------------------------------------------------------

    # NIR = B08
    # Red = B04
    # Green = B03

    false_color = sr[:, :, [3, 2, 1]]

    false_color = normalize_rgb(
        false_color
    )

    false_color_uint8 = (
        false_color * 255
    ).round().astype(np.uint8)

    false_color_path = (
        OUTPUT_DIR /
        "patch_00000_ldsr_s2_nir.png"
    )

    Image.fromarray(
        false_color_uint8
    ).save(false_color_path)

    print(
        f"Saved NIR image:\n{false_color_path}"
    )

    print("\n=== COMPLETE ===")


if __name__ == "__main__":
    main()