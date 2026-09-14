from pathlib import Path
from io import StringIO

import numpy as np
import requests
import torch
import rasterio
from rasterio.transform import Affine
from omegaconf import OmegaConf

import opensr_model


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_TIF = Path(
    "dataset/plan2/geotiff/train_test_512_10m.tif"
)

OUTPUT_TIF = Path(
    "dataset/plan2/ldsr_s2_test/"
    "train_test_512_ldsr_s2_2p5m_10steps.tif"
)

CONFIG_URL = (
    "https://raw.githubusercontent.com/"
    "ESAOpenSR/opensr-model/"
    "refs/heads/main/"
    "opensr_model/configs/config_10m.yaml"
)

# ------------------------------------------------------------
# LDSR-S2 parameters
# ------------------------------------------------------------

SAMPLING_STEPS = 10

PATCH_SIZE = 128
SCALE = 4
OVERLAP = 12

STRIDE = PATCH_SIZE - OVERLAP


# ============================================================
# PATCH GRID
# ============================================================

def create_patch_positions(height, width):
    """
    Create overlapping 128x128 LR patches.

    For a 512x512 input:

        patch size = 128
        overlap    = 12
        stride     = 116

    This produces 25 patches.
    """

    rows = list(
        range(
            0,
            height - PATCH_SIZE + 1,
            STRIDE,
        )
    )

    cols = list(
        range(
            0,
            width - PATCH_SIZE + 1,
            STRIDE,
        )
    )

    # Force final patch to touch image boundaries.

    final_row = height - PATCH_SIZE
    final_col = width - PATCH_SIZE

    if rows[-1] != final_row:
        rows.append(final_row)

    if cols[-1] != final_col:
        cols.append(final_col)

    positions = []

    for row in rows:
        for col in cols:
            positions.append(
                (row, col)
            )

    return positions


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LDSR-S2 — MANUAL 10 STEP TEST")
    print("=" * 70)

    # ========================================================
    # 1. DEVICE
    # ========================================================

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Input :", INPUT_TIF)
    print("Device:", device)

    if device == "cuda":

        print(
            "GPU   :",
            torch.cuda.get_device_name(0)
        )

    print(
        "Sampling steps:",
        SAMPLING_STEPS
    )

    print()

    # ========================================================
    # 2. CHECK INPUT FILE
    # ========================================================

    if not INPUT_TIF.exists():

        raise FileNotFoundError(
            f"Input GeoTIFF not found:\n"
            f"{INPUT_TIF}"
        )

    # ========================================================
    # 3. DOWNLOAD OFFICIAL CONFIG
    # ========================================================

    print(
        "Downloading official "
        "LDSR-S2 configuration..."
    )

    response = requests.get(
        CONFIG_URL,
        timeout=30,
    )

    response.raise_for_status()

    # IMPORTANT:
    #
    # This is exactly the configuration-loading method
    # used by the known-working single-patch script.

    config = OmegaConf.load(
        StringIO(response.text)
    )

    print(
        "Configuration loaded."
    )

    # ========================================================
    # 4. CREATE LDSR-S2 MODEL
    # ========================================================

    print()
    print(
        "Creating LDSR-S2 model..."
    )

    model = opensr_model.SRLatentDiffusion(
        config,
        device=device,
    )

    print(
        "Model created."
    )

    # ========================================================
    # 5. LOAD PRETRAINED CHECKPOINT
    # ========================================================

    print(
        "Loading pretrained checkpoint: "
        f"{config.ckpt_version}"
    )

    model.load_pretrained(
        config.ckpt_version
    )

    print(
        "Pretrained checkpoint loaded."
    )

    # ========================================================
    # 6. VERIFY EVALUATION MODE
    # ========================================================

    if model.training:

        raise RuntimeError(
            "LDSR-S2 model is unexpectedly "
            "in training mode."
        )

    print(
        "Model is in evaluation mode."
    )

    # ========================================================
    # 7. READ INPUT GEOTIFF
    # ========================================================

    print()
    print(
        "Opening input GeoTIFF..."
    )

    with rasterio.open(
        INPUT_TIF
    ) as src:

        data = src.read().astype(
            np.float32
        )

        profile = src.profile.copy()

        transform = src.transform

        crs = src.crs

        height = src.height

        width = src.width

        descriptions = src.descriptions

        bounds = src.bounds

    print(
        f"Input shape: "
        f"{data.shape}"
    )

    print(
        f"Input size: "
        f"{width} x {height}"
    )

    print(
        f"Input CRS: "
        f"{crs}"
    )

    print(
        f"Input resolution: "
        f"{transform.a} x "
        f"{abs(transform.e)} m"
    )

    print(
        f"Input bounds: "
        f"{bounds}"
    )

    print(
        f"Band descriptions: "
        f"{descriptions}"
    )

    print(
        f"Input range: "
        f"{data.min():.6f} -> "
        f"{data.max():.6f}"
    )

    print()

    # ========================================================
    # 8. VALIDATE INPUT
    # ========================================================

    expected_input_shape = (
        4,
        512,
        512,
    )

    if data.shape != expected_input_shape:

        raise RuntimeError(
            f"Expected input shape "
            f"{expected_input_shape}, "
            f"got {data.shape}"
        )

    if not np.isfinite(data).all():

        raise RuntimeError(
            "Input contains NaN or "
            "infinite values."
        )

    # LDSR-S2 uses reflectance-like values.

    data = np.clip(
        data,
        0.0,
        1.0,
    )

    print(
        f"Clipped input range: "
        f"{data.min():.6f} -> "
        f"{data.max():.6f}"
    )

    print()

    # ========================================================
    # 9. CREATE PATCH GRID
    # ========================================================

    positions = create_patch_positions(
        height,
        width,
    )

    print(
        "Patch configuration:"
    )

    print(
        f"  Patch size : "
        f"{PATCH_SIZE} x {PATCH_SIZE}"
    )

    print(
        f"  Overlap    : "
        f"{OVERLAP} LR pixels"
    )

    print(
        f"  Stride     : "
        f"{STRIDE} LR pixels"
    )

    print(
        f"  Total      : "
        f"{len(positions)} patches"
    )

    print()

    if len(positions) != 25:

        raise RuntimeError(
            f"Expected 25 patches, "
            f"got {len(positions)}"
        )

    # ========================================================
    # 10. CREATE HR OUTPUT ARRAYS
    # ========================================================

    hr_height = (
        height * SCALE
    )

    hr_width = (
        width * SCALE
    )

    output_sum = np.zeros(
        (
            4,
            hr_height,
            hr_width,
        ),
        dtype=np.float32,
    )

    output_weight = np.zeros(
        (
            hr_height,
            hr_width,
        ),
        dtype=np.float32,
    )

    # ========================================================
    # 11. RESET CUDA MEMORY STATISTICS
    # ========================================================

    if device == "cuda":

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

    # ========================================================
    # 12. SEQUENTIAL LDSR-S2 INFERENCE
    # ========================================================

    print("=" * 70)
    print(
        "STARTING LDSR-S2 "
        "10-STEP INFERENCE"
    )
    print("=" * 70)

    print()
    print(
        "Processing ONE 128x128 patch "
        "at a time."
    )

    print(
        "This avoids the previous "
        "GPU batching/OOM problem."
    )

    print()

    for index, (row, col) in enumerate(
        positions,
        start=1,
    ):

        print(
            f"[{index:02d}/{len(positions)}] "
            f"LR row={row}, "
            f"col={col}"
        )

        # ----------------------------------------------------
        # Extract LR patch
        # ----------------------------------------------------

        patch = data[
            :,
            row:row + PATCH_SIZE,
            col:col + PATCH_SIZE,
        ]

        if patch.shape != (
            4,
            128,
            128,
        ):

            raise RuntimeError(
                f"Invalid patch shape: "
                f"{patch.shape}"
            )

        # ----------------------------------------------------
        # C,H,W -> B,C,H,W
        # ----------------------------------------------------

        tensor = torch.from_numpy(
            patch
        ).unsqueeze(0)

        tensor = tensor.to(
            device=device,
            dtype=torch.float32,
        )

        # ----------------------------------------------------
        # LDSR-S2 inference
        #
        # EXACTLY 10 sampling steps.
        # ----------------------------------------------------

        if device == "cuda":

            torch.cuda.synchronize()

        try:

            with torch.inference_mode():

                sr = model.forward(
                    tensor,
                    sampling_steps=SAMPLING_STEPS,
                )

        except torch.cuda.OutOfMemoryError:

            print()
            print(
                "CUDA OUT OF MEMORY "
                f"ON PATCH {index}!"
            )

            del tensor

            if device == "cuda":

                torch.cuda.empty_cache()

            raise

        # ----------------------------------------------------
        # Verify SR tensor
        # ----------------------------------------------------

        expected_sr_shape = (
            1,
            4,
            512,
            512,
        )

        if tuple(sr.shape) != expected_sr_shape:

            raise RuntimeError(
                f"Unexpected SR shape: "
                f"{tuple(sr.shape)}; "
                f"expected "
                f"{expected_sr_shape}"
            )

        # ----------------------------------------------------
        # BCHW -> CHW
        # ----------------------------------------------------

        sr_numpy = (
            sr
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # Numerical validation
        # ----------------------------------------------------

        if not np.isfinite(
            sr_numpy
        ).all():

            raise RuntimeError(
                f"Patch {index} "
                "contains NaN or "
                "infinite values."
            )

        # ----------------------------------------------------
        # Reflectance clipping
        # ----------------------------------------------------

        sr_numpy = np.clip(
            sr_numpy,
            0.0,
            1.0,
        )

        print(
            f"    SR range: "
            f"{sr_numpy.min():.6f} -> "
            f"{sr_numpy.max():.6f}"
        )

        # ----------------------------------------------------
        # Convert LR coordinates to HR coordinates
        # ----------------------------------------------------

        hr_row = (
            row * SCALE
        )

        hr_col = (
            col * SCALE
        )

        sr_height = sr_numpy.shape[1]
        sr_width = sr_numpy.shape[2]

        # ----------------------------------------------------
        # Accumulate patch
        # ----------------------------------------------------

        output_sum[
            :,
            hr_row:hr_row + sr_height,
            hr_col:hr_col + sr_width,
        ] += sr_numpy

        output_weight[
            hr_row:hr_row + sr_height,
            hr_col:hr_col + sr_width,
        ] += 1.0

        # ----------------------------------------------------
        # Free GPU memory
        # ----------------------------------------------------

        del tensor
        del sr
        del sr_numpy

        if device == "cuda":

            torch.cuda.empty_cache()

            peak_memory = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 3)
            )

            print(
                f"    Peak GPU memory: "
                f"{peak_memory:.2f} GB"
            )

        print()

    # ========================================================
    # 13. COVERAGE CHECK
    # ========================================================

    print("=" * 70)
    print(
        "CHECKING OUTPUT COVERAGE"
    )
    print("=" * 70)

    uncovered = np.sum(
        output_weight == 0
    )

    total_pixels = (
        hr_height * hr_width
    )

    covered_pixels = (
        total_pixels - uncovered
    )

    coverage = (
        covered_pixels
        / total_pixels
        * 100.0
    )

    print(
        f"Covered pixels : "
        f"{covered_pixels:,}"
    )

    print(
        f"Total pixels   : "
        f"{total_pixels:,}"
    )

    print(
        f"Coverage       : "
        f"{coverage:.4f}%"
    )

    if uncovered > 0:

        raise RuntimeError(
            f"{uncovered:,} HR pixels "
            "were not covered."
        )

    print(
        "Coverage: 100%"
    )

    print()

    # ========================================================
    # 14. BLEND OVERLAPPING PATCHES
    # ========================================================

    print(
        "Blending overlapping patches..."
    )

    output = (
        output_sum
        / output_weight[None, :, :]
    )

    output = np.clip(
        output,
        0.0,
        1.0,
    )

    # Free intermediate arrays.

    del output_sum
    del output_weight

    # ========================================================
    # 15. FINAL NUMERICAL VALIDATION
    # ========================================================

    if not np.isfinite(
        output
    ).all():

        raise RuntimeError(
            "Final output contains "
            "NaN or infinite values."
        )

    print(
        "Final output range: "
        f"{output.min():.6f} -> "
        f"{output.max():.6f}"
    )

    print()

    # ========================================================
    # 16. CREATE 2.5m AFFINE TRANSFORM
    # ========================================================
    #
    # IMPORTANT:
    # rasterio.Affine cannot be multiplied by a scalar
    # in the way the previous script attempted.
    #
    # We explicitly divide the pixel-size components.
    #

    output_transform = Affine(
        transform.a / SCALE,
        transform.b,
        transform.c,
        transform.d,
        transform.e / SCALE,
        transform.f,
    )

    # ========================================================
    # 17. OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_TIF.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 18. OUTPUT GEOTIFF PROFILE
    # ========================================================

    output_profile = profile.copy()

    output_profile.update(
        driver="GTiff",
        height=hr_height,
        width=hr_width,
        count=4,
        dtype="float32",
        transform=output_transform,
        crs=crs,
        compress="deflate",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    # ========================================================
    # 19. WRITE FINAL GEOTIFF
    # ========================================================

    print("=" * 70)
    print(
        "WRITING 2.5m LDSR-S2 OUTPUT"
    )
    print("=" * 70)

    print(
        "Output:",
        OUTPUT_TIF
    )

    with rasterio.open(
        OUTPUT_TIF,
        "w",
        **output_profile,
    ) as dst:

        dst.write(output)

        dst.set_band_description(
            1,
            "B02",
        )

        dst.set_band_description(
            2,
            "B03",
        )

        dst.set_band_description(
            3,
            "B04",
        )

        dst.set_band_description(
            4,
            "B08",
        )

    # ========================================================
    # 20. VERIFY FINAL GEOTIFF
    # ========================================================

    print()
    print("=" * 70)
    print(
        "VERIFYING FINAL OUTPUT"
    )
    print("=" * 70)

    with rasterio.open(
        OUTPUT_TIF
    ) as src:

        print(
            "Output:",
            OUTPUT_TIF
        )

        print(
            "Size:",
            src.width,
            "x",
            src.height
        )

        print(
            "Bands:",
            src.count
        )

        print(
            "Resolution:",
            src.res
        )

        print(
            "CRS:",
            src.crs
        )

        print(
            "Bounds:",
            src.bounds
        )

        print(
            "Band descriptions:",
            src.descriptions
        )

        # Small verification window.

        sample = src.read(
            window=rasterio.windows.Window(
                0,
                0,
                min(256, src.width),
                min(256, src.height),
            )
        )

    print(
        "Sample range:",
        f"{sample.min():.6f}",
        "->",
        f"{sample.max():.6f}"
    )

    # ========================================================
    # 21. FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "LDSR-S2 10-STEP TEST COMPLETE"
    )
    print("=" * 70)

    print()

    print("INPUT")
    print(
        "  Sentinel-2"
    )
    print(
        "  10 m"
    )
    print(
        "  512 x 512"
    )
    print(
        "  B02 / B03 / B04 / B08"
    )

    print()

    print("MODEL")
    print(
        "  LDSR-S2"
    )
    print(
        f"  Sampling steps: "
        f"{SAMPLING_STEPS}"
    )
    print(
        "  Scale: 4x"
    )

    print()

    print("OUTPUT")
    print(
        "  2.5 m"
    )
    print(
        "  2048 x 2048"
    )
    print(
        "  B02 / B03 / B04 / B08"
    )

    print()

    print("PIPELINE")
    print(
        "  10m Sentinel-2"
    )
    print(
        "       ↓"
    )
    print(
        "  25 overlapping 128x128 patches"
    )
    print(
        "       ↓"
    )
    print(
        "  LDSR-S2 — 10 sampling steps"
    )
    print(
        "       ↓"
    )
    print(
        "  25 × 512x512 SR patches"
    )
    print(
        "       ↓"
    )
    print(
        "  overlap blending"
    )
    print(
        "       ↓"
    )
    print(
        "  2048x2048 @ 2.5m"
    )

    print()

    print(
        "Final output:"
    )

    print(
        f"  {OUTPUT_TIF}"
    )

    print()

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()