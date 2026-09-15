from pathlib import Path
import json
import re

import numpy as np
import rasterio
from rasterio.transform import Affine


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

INPUT_TIF = Path(
    "dataset/plan2/geotiff/train_test_512_10m.tif"
)

TEMP_DIR = Path(
    "dataset/plan2/geotiff/temp_20260914_185702_944624_18264"
)

OUTPUT_TIF = Path(
    "dataset/plan2/ldsr_s2_test/train_test_512_ldsr_s2_2p5m.tif"
)

INDEX_JSON = TEMP_DIR / "index.json"

SCALE = 4


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    OUTPUT_TIF.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Read input GeoTIFF metadata
    # -------------------------------------------------------------

    with rasterio.open(INPUT_TIF) as src:

        input_width = src.width
        input_height = src.height
        input_transform = src.transform
        input_crs = src.crs

        band_names = ["B02", "B03", "B04", "B08"]

        print("Input:")
        print(f"  Size: {input_width} x {input_height}")
        print(f"  Resolution: {src.res}")
        print(f"  CRS: {input_crs}")
        print()

    # -------------------------------------------------------------
    # Expected SR dimensions
    # -------------------------------------------------------------

    output_width = input_width * SCALE
    output_height = input_height * SCALE

    print("Expected SR:")
    print(f"  Size: {output_width} x {output_height}")
    print("  Resolution: 2.5 m")
    print()

    # -------------------------------------------------------------
    # Read opensr-utils index.json
    # -------------------------------------------------------------

    if not INDEX_JSON.exists():
        raise FileNotFoundError(
            f"index.json not found:\n{INDEX_JSON}"
        )

    with open(INDEX_JSON, "r", encoding="utf-8") as f:
        index = json.load(f)

    entries = index["entries"]

    print(f"Found {len(entries)} SR patches.")
    print()

    # -------------------------------------------------------------
    # Check metadata
    # -------------------------------------------------------------

    factor = index.get("factor")

    if factor != SCALE:
        raise ValueError(
            f"Unexpected scale factor in index.json: {factor}"
        )

    output_bands = index.get("output_bands")

    if output_bands != 4:
        raise ValueError(
            f"Expected 4 output bands, got {output_bands}"
        )

    # -------------------------------------------------------------
    # Allocate accumulation arrays
    #
    # IMPORTANT:
    # We use row_off_hr / col_off_hr directly from index.json.
    # We DO NOT calculate positions from filenames.
    # -------------------------------------------------------------

    accumulation = np.zeros(
        (4, output_height, output_width),
        dtype=np.float64,
    )

    weights = np.zeros(
        (output_height, output_width),
        dtype=np.float32,
    )

    # -------------------------------------------------------------
    # Process every SR patch
    # -------------------------------------------------------------

    for i, entry in enumerate(entries, start=1):

        patch_path = Path(entry["path"])

        # Handle paths relative to the project root.
        if not patch_path.is_absolute():
            patch_path = Path.cwd() / patch_path

        if not patch_path.exists():
            raise FileNotFoundError(
                f"SR patch does not exist:\n{patch_path}"
            )

        # ---------------------------------------------------------
        # Read actual HR placement from index.json
        # ---------------------------------------------------------

        row_off = int(entry["row_off_hr"])
        col_off = int(entry["col_off_hr"])

        patch_height = int(entry["height_hr"])
        patch_width = int(entry["width_hr"])

        # ---------------------------------------------------------
        # Load SR patch
        # ---------------------------------------------------------

        sr = np.load(patch_path)

        # opensr-utils stores:
        #   (bands, height, width)
        #
        # We also support:
        #   (height, width, bands)
        # ---------------------------------------------------------

        if sr.ndim != 3:
            raise ValueError(
                f"Unexpected SR shape {sr.shape} "
                f"for {patch_path}"
            )

        if sr.shape == (
            4,
            patch_height,
            patch_width,
        ):
            # CHW
            sr_chw = sr.astype(np.float32)

        elif sr.shape == (
            patch_height,
            patch_width,
            4,
        ):
            # HWC -> CHW
            sr_chw = np.transpose(
                sr,
                (2, 0, 1)
            ).astype(np.float32)

        else:
            raise ValueError(
                f"SR shape {sr.shape} does not match "
                f"expected "
                f"(4,{patch_height},{patch_width}) "
                f"or "
                f"({patch_height},{patch_width},4)"
            )

        # ---------------------------------------------------------
        # opensr-utils index says saved dtype is uint16 with
        # saved_scale = 10000.
        #
        # Convert back to reflectance [0,1].
        # ---------------------------------------------------------

        saved_dtype = index.get("saved_dtype")
        saved_scale = index.get("saved_scale")

        if saved_dtype == "uint16" and saved_scale:
            sr_chw = sr_chw / float(saved_scale)

        # Safety conversion if values are still outside [0,1].
        sr_chw = np.clip(sr_chw, 0.0, 1.0)

        # ---------------------------------------------------------
        # Validate patch placement
        # ---------------------------------------------------------

        row_end = row_off + patch_height
        col_end = col_off + patch_width

        if row_off < 0 or col_off < 0:
            raise ValueError(
                f"Invalid negative patch offset: "
                f"{row_off}, {col_off}"
            )

        if row_end > output_height:
            raise ValueError(
                f"Patch exceeds output height: "
                f"{row_end} > {output_height}"
            )

        if col_end > output_width:
            raise ValueError(
                f"Patch exceeds output width: "
                f"{col_end} > {output_width}"
            )

        # ---------------------------------------------------------
        # Accumulate overlapping patches
        # ---------------------------------------------------------

        accumulation[
            :,
            row_off:row_end,
            col_off:col_end
        ] += sr_chw

        weights[
            row_off:row_end,
            col_off:col_end
        ] += 1.0

        # Print only selected patches to keep output readable.
        if (
            i <= 4
            or i % 5 == 0
            or i == len(entries)
        ):
            print(
                f"[{i:02d}/{len(entries)}] "
                f"{patch_path.name} "
                f"-> HR "
                f"row={row_off}, "
                f"col={col_off}, "
                f"size={patch_width}x{patch_height}"
            )

    # -------------------------------------------------------------
    # Coverage check
    # -------------------------------------------------------------

    uncovered = np.count_nonzero(weights == 0)

    total_pixels = output_width * output_height

    coverage = (
        100.0
        * (total_pixels - uncovered)
        / total_pixels
    )

    if uncovered > 0:

        print()
        print(
            f"WARNING: {uncovered} output pixels "
            f"were not covered by an SR patch."
        )

        print(
            f"Coverage: {coverage:.4f}%"
        )

    else:

        print()
        print(
            "Coverage check: 100% of output pixels covered."
        )

    # -------------------------------------------------------------
    # Average overlapping patches
    # -------------------------------------------------------------

    safe_weights = np.maximum(weights, 1.0)

    output = (
        accumulation
        / safe_weights[None, :, :]
    )

    # -------------------------------------------------------------
    # For uncovered pixels, explicitly set to 0.
    # Normally this should never happen.
    # -------------------------------------------------------------

    if uncovered > 0:

        output[:, weights == 0] = 0.0

    output = np.clip(
        output,
        0.0,
        1.0
    ).astype(np.float32)

    # -------------------------------------------------------------
    # Create 2.5 m transform
    # -------------------------------------------------------------

    output_transform = Affine(
        input_transform.a / SCALE,
        input_transform.b,
        input_transform.c,
        input_transform.d,
        input_transform.e / SCALE,
        input_transform.f,
    )

    # -------------------------------------------------------------
    # Write output GeoTIFF
    # -------------------------------------------------------------

    print()
    print("Writing output GeoTIFF...")

    with rasterio.open(
        OUTPUT_TIF,
        "w",
        driver="GTiff",
        width=output_width,
        height=output_height,
        count=4,
        dtype="float32",
        crs=input_crs,
        transform=output_transform,
        compress="deflate",
        predictor=3,
        tiled=True,
        BIGTIFF="IF_SAFER",
    ) as dst:

        for band_idx, band_name in enumerate(
            band_names,
            start=1
        ):

            dst.write(
                output[band_idx - 1],
                band_idx
            )

            dst.set_band_description(
                band_idx,
                band_name
            )

    # -------------------------------------------------------------
    # Verify output
    # -------------------------------------------------------------

    with rasterio.open(OUTPUT_TIF) as src:

        print()
        print("=" * 70)
        print("LDSR-S2 SR GEOTIFF CREATED")
        print("=" * 70)

        print(f"Output     : {OUTPUT_TIF}")
        print(
            f"Size       : "
            f"{src.width} x {src.height}"
        )
        print(f"Bands      : {src.count}")
        print(f"Resolution : {src.res}")
        print(f"CRS        : {src.crs}")
        print(f"Bounds     : {src.bounds}")
        print(f"Dtype      : {src.dtypes}")
        print(
            f"Bands      : "
            f"{tuple(src.descriptions)}"
        )

        # Read a small window for sanity check.
        sample = src.read(
            window=rasterio.windows.Window(
                0,
                0,
                min(512, src.width),
                min(512, src.height),
            )
        )

        print(
            f"Sample value range: "
            f"{sample.min():.6f} "
            f"→ "
            f"{sample.max():.6f}"
        )

    print()
    print("STITCHING COMPLETE.")
    print()
    print(
        "Final product:"
    )
    print(
        "  10 m Sentinel-2"
        " → 4× LDSR-S2"
        " → 2.5 m GeoTIFF"
    )


if __name__ == "__main__":
    main()