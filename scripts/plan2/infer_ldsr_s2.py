from io import StringIO
from pathlib import Path
import argparse

import numpy as np
import requests
import rasterio
import torch
from omegaconf import OmegaConf
from rasterio.transform import Affine

import opensr_model


CONFIG_URL = (
    "https://raw.githubusercontent.com/"
    "ESAOpenSR/opensr-model/"
    "refs/heads/main/"
    "opensr_model/configs/config_10m.yaml"
)

MODEL_PATCH_SIZE = 128
SCALE = 4
SAMPLING_STEPS = 100
OVERLAP = 12


def load_model(device):
    print("Loading LDSR-S2 configuration...")

    response = requests.get(CONFIG_URL, timeout=30)
    response.raise_for_status()

    config = OmegaConf.load(StringIO(response.text))

    print("Creating LDSR-S2 model...")
    model = opensr_model.SRLatentDiffusion(config, device=device)

    print("Loading pretrained checkpoint...")
    model.load_pretrained(config.ckpt_version)

    model.eval()

    print("LDSR-S2 model loaded successfully.")

    return model


def validate_input(src):
    if src.count != 4:
        raise ValueError(
            f"Expected 4 bands (B02/B03/B04/B08), found {src.count}."
        )

    if src.width < MODEL_PATCH_SIZE or src.height < MODEL_PATCH_SIZE:
        raise ValueError(
            f"Input must be at least "
            f"{MODEL_PATCH_SIZE}×{MODEL_PATCH_SIZE} pixels."
        )

    if src.crs is None:
        raise ValueError("Input GeoTIFF has no CRS.")

    if src.transform.a == 0 or src.transform.e == 0:
        raise ValueError("Invalid GeoTIFF transform.")

    print(f"Input size       : {src.width} × {src.height}")
    print(f"Input resolution : {src.res}")
    print(f"Input CRS        : {src.crs}")
    print(f"Input bands      : {src.count}")
    print(f"Input bounds     : {src.bounds}")


def normalize_image(image):
    """
    Convert Sentinel-2 data to float32 reflectance-like values in [0, 1].

    Handles:
      - already normalized floating-point data
      - Sentinel-2-style integer/DN data around 0–10000
    """

    image = image.astype(np.float32)

    finite = np.isfinite(image)

    if not finite.any():
        raise ValueError("Input contains no finite pixel values.")

    max_value = float(np.nanmax(image))

    if max_value > 1.5:
        print(
            f"Detected DN-style input (max={max_value:.2f}). "
            "Scaling by 10000."
        )
        image /= 10000.0
    else:
        print(
            f"Detected normalized input (max={max_value:.4f})."
        )

    image = np.clip(image, 0.0, 1.0)

    return image


def get_positions(length, patch_size, overlap):
    """
    Generate tile positions while guaranteeing complete coverage.
    """

    stride = patch_size - overlap

    if length <= patch_size:
        return [0]

    positions = list(range(0, length - patch_size + 1, stride))

    last = length - patch_size

    if positions[-1] != last:
        positions.append(last)

    return positions


def infer_tile(model, tile, device):
    """
    Run one 128×128×4 tile through LDSR-S2.
    """

    tensor = (
        torch.from_numpy(tile)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )

    with torch.inference_mode():
        sr = model.forward(
            tensor,
            sampling_steps=SAMPLING_STEPS,
        )

    sr = sr.squeeze(0).permute(1, 2, 0).cpu().numpy()

    del tensor

    if not np.isfinite(sr).all():
        raise ValueError("LDSR output contains NaN or Inf values.")

    return sr.astype(np.float32)


def run_inference(model, image, device):
    """
    Tile the input image, run LDSR-S2 sequentially,
    and blend overlapping SR tiles.
    """

    height, width, channels = image.shape

    y_positions = get_positions(
        height,
        MODEL_PATCH_SIZE,
        OVERLAP,
    )

    x_positions = get_positions(
        width,
        MODEL_PATCH_SIZE,
        OVERLAP,
    )

    total_tiles = len(y_positions) * len(x_positions)

    print(
        f"Input dimensions : {width} × {height}"
    )
    print(
        f"Tile size        : {MODEL_PATCH_SIZE} × {MODEL_PATCH_SIZE}"
    )
    print(
        f"Overlap          : {OVERLAP} pixels"
    )
    print(
        f"Total tiles      : {total_tiles}"
    )

    output_height = height * SCALE
    output_width = width * SCALE

    accumulation = np.zeros(
        (output_height, output_width, channels),
        dtype=np.float32,
    )

    weights = np.zeros(
        (output_height, output_width),
        dtype=np.float32,
    )

    tile_number = 0

    for y in y_positions:
        for x in x_positions:

            tile_number += 1

            print(
                f"[{tile_number}/{total_tiles}] "
                f"Input tile: row={y}, col={x}"
            )

            tile = image[
                y:y + MODEL_PATCH_SIZE,
                x:x + MODEL_PATCH_SIZE,
                :
            ]

            sr = infer_tile(
                model,
                tile,
                device,
            )

            sr_y = y * SCALE
            sr_x = x * SCALE

            sr_h, sr_w = sr.shape[:2]

            accumulation[
                sr_y:sr_y + sr_h,
                sr_x:sr_x + sr_w,
                :
            ] += sr

            weights[
                sr_y:sr_y + sr_h,
                sr_x:sr_x + sr_w
            ] += 1.0

            del tile
            del sr

            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    print("Blending overlapping tiles...")

    valid = weights > 0

    output = np.zeros_like(accumulation)

    output[valid] = (
        accumulation[valid] /
        weights[valid, None]
    )

    if not np.isfinite(output).all():
        raise ValueError(
            "Final output contains NaN or Inf values."
        )

    print(
        f"Final output range: "
        f"{output.min():.6f} -> {output.max():.6f}"
    )

    return output


def create_output_transform(transform):
    """
    Convert a 10m transform to a 2.5m transform.
    """

    return Affine(
        transform.a / SCALE,
        transform.b,
        transform.c,
        transform.d,
        transform.e / SCALE,
        transform.f,
    )


def save_output(output, output_path, source_profile):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile = source_profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)

    profile.update(
        driver="GTiff",
        height=output.shape[0],
        width=output.shape[1],
        count=4,
        dtype="float32",
        transform=create_output_transform(
            source_profile["transform"]
        ),
        compress="deflate",
        predictor=3,
        tiled=False,
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:

        for band in range(4):
            dst.write(
                output[:, :, band],
                band + 1,
            )

        dst.set_band_description(1, "B02")
        dst.set_band_description(2, "B03")
        dst.set_band_description(3, "B04")
        dst.set_band_description(4, "B08")

    print(f"Saved output: {output_path}")


def verify_output(output_path, expected_bounds):
    with rasterio.open(output_path) as src:

        print("\nOutput verification")
        print("-------------------")
        print(f"Size        : {src.width} x {src.height}")
        print(f"Resolution  : {src.res}")
        print(f"CRS         : {src.crs}")
        print(f"Bands       : {src.count}")
        print(f"Bounds      : {src.bounds}")

        expected_width = None
        expected_height = None

        if not np.allclose(
            [
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
            ],
            [
                expected_bounds.left,
                expected_bounds.bottom,
                expected_bounds.right,
                expected_bounds.top,
            ],
            atol=1e-4,
        ):
            raise ValueError(
                "Output geographic bounds do not match input."
            )

        data = src.read()

        if not np.isfinite(data).all():
            raise ValueError(
                "Output GeoTIFF contains NaN or Inf."
            )

        print("Bounds       : PASS")
        print("Finite data  : PASS")
        print("Verification : PASS")


def main():

    parser = argparse.ArgumentParser(
        description=(
            "PixelSight Plan 2 production "
            "LDSR-S2 4× inference pipeline"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input 4-band Sentinel-2 GeoTIFF.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output 2.5m 4-band GeoTIFF.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}"
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("PixelSight Plan 2")
    print("LDSR-S2 Production Inference")
    print("=" * 60)

    print(f"Device: {device}")

    if device == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    with rasterio.open(args.input) as src:

        validate_input(src)

        profile = src.profile.copy()
        transform = src.transform
        bounds = src.bounds

        raw = src.read()

    # Convert:
    # (4, H, W) → (H, W, 4)
    image = np.moveaxis(raw, 0, -1)

    image = normalize_image(image)

    print(
        f"Normalized range: "
        f"{image.min():.6f} → {image.max():.6f}"
    )

    model = load_model(device)

    output = run_inference(
        model,
        image,
        device,
    )

    profile["transform"] = transform

    save_output(
        output,
        args.output,
        profile,
    )

    verify_output(
        args.output,
        bounds,
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()