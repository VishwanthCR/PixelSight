from io import StringIO
from pathlib import Path

import requests
import rasterio
import torch
from omegaconf import OmegaConf

import opensr_model
import opensr_utils


# =========================================================
# Configuration
# =========================================================

INPUT = Path(
    "dataset/plan2/geotiff/train_test_512_10m.tif"
)

OUTPUT_DIR = Path(
    "dataset/plan2/ldsr_s2_test"
)

CONFIG_URL = (
    "https://raw.githubusercontent.com/ESAOpenSR/"
    "opensr-model/refs/heads/main/"
    "opensr_model/configs/config_10m.yaml"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BAND_NAMES = [
    "B02",
    "B03",
    "B04",
    "B08",
]


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("PixelSight — LDSR-S2 REAL SENTINEL-2 TEST")
    print("=" * 70)

    print(f"Input : {INPUT}")
    print(f"Device: {DEVICE}")

    if DEVICE == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    print()


    # -----------------------------------------------------
    # Verify input
    # -----------------------------------------------------

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input GeoTIFF not found: {INPUT}"
        )

    with rasterio.open(INPUT) as src:

        print("Input GeoTIFF:")
        print(f"  Size       : {src.width} x {src.height}")
        print(f"  Bands      : {src.count}")
        print(f"  Resolution : {src.res}")
        print(f"  CRS        : {src.crs}")
        print(f"  Bounds     : {src.bounds}")
        print(f"  Dtype      : {src.dtypes}")
        print(f"  Band names : {src.descriptions}")
        print()


        # Basic validation
        if src.count != 4:
            raise ValueError(
                f"Expected 4 bands, found {src.count}"
            )

        if src.width != 512 or src.height != 512:
            raise ValueError(
                "Expected a 512 x 512 test tile."
            )

        if src.res != (10.0, 10.0):
            raise ValueError(
                f"Expected 10 m resolution, found {src.res}"
            )


    # -----------------------------------------------------
    # Load official LDSR-S2 configuration
    # -----------------------------------------------------

    print("Loading official LDSR-S2 configuration...")

    response = requests.get(
        CONFIG_URL,
        timeout=30
    )

    response.raise_for_status()

    config = OmegaConf.load(
        StringIO(response.text)
    )

    print("Configuration loaded.")
    print()


    # -----------------------------------------------------
    # Create LDSR-S2 model
    # -----------------------------------------------------

    print("Creating LDSR-S2 model...")

    model = opensr_model.SRLatentDiffusion(
        config,
        device=DEVICE,
    )

    print("Model created.")
    print()


    # -----------------------------------------------------
    # Load official pretrained checkpoint
    # -----------------------------------------------------

    print("Loading pretrained checkpoint...")

    model.load_pretrained(
        config.ckpt_version
    )

    model.eval()

    print("Pretrained LDSR-S2 loaded.")
    print()


    # -----------------------------------------------------
    # Prepare output directory
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # -----------------------------------------------------
    # LDSR-S2 / OpenSR parameters
    # -----------------------------------------------------

    print("LDSR-S2 inference configuration:")
    print("  Input window      : 128 x 128")
    print("  Super-resolution  : 4x")
    print("  Overlap           : 12 pixels")
    print("  Border elimination: 2 pixels")
    print("  Batch size        : 1")
    print("  DataLoader workers: 0")
    print()


    # -----------------------------------------------------
    # Create OpenSR processing pipeline
    # -----------------------------------------------------

    print("Creating OpenSR large-file processor...")

    processor = opensr_utils.large_file_processing(
        root=str(INPUT),
        model=model,

        window_size=(128, 128),

        factor=4,

        overlap=12,

        eliminate_border_px=2,

        device=DEVICE,

        gpus=0,

        save_preview=True,

        debug=True,

        auto_run=False,

        cleanup=True,

        overwrite=True,

        batch_size=1,

        # IMPORTANT FOR WINDOWS
        num_workers=0,

        compressed_patches=False,
    )

    print()
    print("Processor created.")
    print()


    # -----------------------------------------------------
    # Start super-resolution
    # -----------------------------------------------------

    print("=" * 70)
    print("STARTING LDSR-S2 SUPER-RESOLUTION")
    print("=" * 70)

    print()
    print("Real Sentinel-2 input:")
    print("  512 x 512 pixels")
    print("  10 m resolution")
    print("  4 bands: B02/B03/B04/B08")

    print()
    print("Expected SR:")
    print("  2048 x 2048 pixels")
    print("  2.5 m resolution")
    print("  4 bands")

    print()

    result = processor.start_super_resolution()


    # -----------------------------------------------------
    # Finished
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("LDSR-S2 TEST COMPLETE")
    print("=" * 70)

    print()
    print("OpenSR result:")
    print(result)

    print()
    print("Check the dataset/plan2/geotiff directory")
    print("for the generated SR product and processing logs.")


# =========================================================
# Windows multiprocessing protection
# =========================================================

if __name__ == "__main__":
    main()