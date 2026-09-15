from pathlib import Path
import json

import numpy as np
import rasterio
import torch
from omegaconf import OmegaConf

import opensr_model
import opensr_utils


INPUT_TIF = Path(
    "dataset/plan2/geotiff/train_test_512_10m.tif"
)

OUTPUT_TIF = Path(
    "dataset/plan2/ldsr_s2_test/"
    "train_test_512_ldsr_s2_2p5m_10steps.tif"
)

CONFIG_URL = (
    "https://raw.githubusercontent.com/"
    "ESAOpenSR/opensr-model/refs/heads/main/"
    "opensr_model/configs/config_10m.yaml"
)

SAMPLING_STEPS = 10
SCALE = 4
PROCESSING_INPUT_TIF = Path(
    "dataset/plan2/geotiff/train_test_512_10m_opensr_10000.tif"
)


class LDSR10StepWrapper(torch.nn.Module):
    """
    Forces LDSR-S2 to use a fixed number of
    diffusion sampling steps.
    """

    def __init__(self, model, sampling_steps):
        super().__init__()

        self.model = model
        self.sampling_steps = sampling_steps

    def forward(self, x):
        return self.model.forward(
            x,
            sampling_steps=self.sampling_steps
        )


def load_config():

    import requests

    response = requests.get(
        CONFIG_URL,
        timeout=30
    )

    response.raise_for_status()

    config_file = Path(
        "dataset/plan2/config_10m.yaml"
    )

    config_file.write_text(
        response.text,
        encoding="utf-8"
    )

    return OmegaConf.load(config_file)


def stitch_from_index(
    temp_dir,
    input_tif,
    output_tif
):

    index_file = temp_dir / "index.json"

    with open(
        index_file,
        "r",
        encoding="utf-8"
    ) as f:
        index = json.load(f)

    entries = index["entries"]

    with rasterio.open(input_tif) as src:

        input_height = src.height
        input_width = src.width
        transform = src.transform
        crs = src.crs

    output_height = input_height * SCALE
    output_width = input_width * SCALE

    accumulation = np.zeros(
        (4, output_height, output_width),
        dtype=np.float64
    )

    weights = np.zeros(
        (output_height, output_width),
        dtype=np.float32
    )

    print()
    print(f"Found {len(entries)} SR patches.")
    print()

    for i, entry in enumerate(
        entries,
        start=1
    ):

        patch_path = Path(entry["path"])

        if not patch_path.is_absolute():
            patch_path = Path.cwd() / patch_path

        sr = np.load(patch_path)

        row = int(entry["row_off_hr"])
        col = int(entry["col_off_hr"])

        height = int(entry["height_hr"])
        width = int(entry["width_hr"])

        if sr.shape == (
            4,
            height,
            width
        ):
            sr = sr.astype(np.float32)

        elif sr.shape == (
            height,
            width,
            4
        ):
            sr = np.transpose(
                sr,
                (2, 0, 1)
            ).astype(np.float32)

        else:
            raise ValueError(
                f"Unexpected SR shape: "
                f"{sr.shape}"
            )

        # opensr-utils stores uint16 / 10000.
        if index.get("saved_dtype") == "uint16":

            scale = float(
                index.get("saved_scale", 10000)
            )

            sr /= scale

        sr = np.clip(
            sr,
            0.0,
            1.0
        )

        row_end = row + height
        col_end = col + width

        accumulation[
            :,
            row:row_end,
            col:col_end
        ] += sr

        weights[
            row:row_end,
            col:col_end
        ] += 1

        if (
            i <= 5
            or i % 5 == 0
            or i == len(entries)
        ):

            print(
                f"[{i:02d}/{len(entries)}] "
                f"{patch_path.name} "
                f"-> HR row={row}, "
                f"col={col}"
            )

    uncovered = np.count_nonzero(
        weights == 0
    )

    total = (
        output_height *
        output_width
    )

    coverage = (
        100 *
        (total - uncovered) /
        total
    )

    print()
    print(
        f"Coverage: {coverage:.4f}%"
    )

    if uncovered:
        raise RuntimeError(
            f"{uncovered} output pixels "
            f"were not covered."
        )

    output = (
        accumulation /
        weights[None, :, :]
    )

    output = np.clip(
        output,
        0,
        1
    ).astype(np.float32)

    output_transform = rasterio.Affine(
        transform.a / SCALE,
        transform.b,
        transform.c,
        transform.d,
        transform.e / SCALE,
        transform.f
    )

    output_tif.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with rasterio.open(
        output_tif,
        "w",
        driver="GTiff",
        width=output_width,
        height=output_height,
        count=4,
        dtype="float32",
        crs=crs,
        transform=output_transform,
        compress="deflate",
        predictor=3,
        tiled=True,
        BIGTIFF="IF_SAFER"
    ) as dst:

        for i, name in enumerate(
            ["B02", "B03", "B04", "B08"],
            start=1
        ):

            dst.write(
                output[i - 1],
                i
            )

            dst.set_band_description(
                i,
                name
            )

    print()
    print("=" * 70)
    print("10-STEP LDSR-S2 OUTPUT CREATED")
    print("=" * 70)

    print(
        f"Output     : {output_tif}"
    )

    print(
        f"Size       : "
        f"{output_width} x {output_height}"
    )

    print(
        "Resolution : 2.5 m"
    )

    print(
        f"Sampling   : "
        f"{SAMPLING_STEPS} steps"
    )

    with rasterio.open(output_tif) as result:
        data = result.read()

        if (
            result.width != output_width
            or result.height != output_height
            or result.count != 4
            or result.res != (2.5, 2.5)
        ):
            raise RuntimeError(
                "SR GeoTIFF metadata does not match the expected 4x output."
            )

        if not np.isfinite(data).all() or not np.any(data > 0):
            raise RuntimeError(
                "SR GeoTIFF contains no finite, nonzero signal."
            )

        report = {
            "input": str(input_tif),
            "output": str(output_tif),
            "sampling_steps": SAMPLING_STEPS,
            "scale": SCALE,
            "input_size": [input_width, input_height],
            "output_size": [result.width, result.height],
            "resolution_m": list(result.res),
            "crs": str(result.crs),
            "bands": list(result.descriptions),
            "finite": bool(np.isfinite(data).all()),
            "min": float(data.min()),
            "max": float(data.max()),
            "mean": float(data.mean()),
            "std": float(data.std()),
        }

    report_path = output_tif.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(f"Validation report: {report_path}")


def main():

    print("=" * 70)
    print("LDSR-S2 — 10 STEP TEST")
    print("=" * 70)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(f"Input : {INPUT_TIF}")
    print(f"Device: {device}")

    if device == "cuda":

        print(
            f"GPU   : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print()
    print(
        f"Sampling steps: "
        f"{SAMPLING_STEPS}"
    )

    config = load_config()

    with rasterio.open(INPUT_TIF) as source:
        source_data = source.read()
        source_profile = source.profile.copy()

    processing_input = INPUT_TIF

    if float(source_data.max()) <= 1.0:
        scaled_data = (source_data * 10000.0).round().astype(np.uint16)
        source_profile.update(dtype="uint16")

        with rasterio.open(
            PROCESSING_INPUT_TIF,
            "w",
            **source_profile
        ) as destination:
            destination.write(scaled_data)
            for band, name in enumerate(
                ["B02", "B03", "B04", "B08"],
                start=1
            ):
                destination.set_band_description(band, name)

        processing_input = PROCESSING_INPUT_TIF
        print(
            f"Scaled normalized input for OpenSR: {processing_input}"
        )

    print()
    print("Creating LDSR-S2 model...")

    base_model = (
        opensr_model.SRLatentDiffusion(
            config,
            device=device
        )
    )

    print(
        "Loading pretrained checkpoint..."
    )

    base_model.load_pretrained(
        config.ckpt_version
    )

    base_model.eval()

    model = LDSR10StepWrapper(
        base_model,
        SAMPLING_STEPS
    )

    model.eval()

    print(
        "10-step model wrapper ready."
    )

    print()
    print(
        "Starting opensr-utils inference..."
    )

    runner = (
        opensr_utils.large_file_processing(
            root=str(processing_input.resolve()),
            model=model,
            window_size=(128, 128),
            factor=4,
            overlap=12,
            eliminate_border_px=2,
            device=device,
            gpus=0,
            save_preview=True,
            debug=True,
            overwrite=True,
            batch_size=1,
            num_workers=0,
            compressed_patches=False,
            auto_run=False,
            cleanup=False
        )
    )

    runner.run()

    # Find newest temporary directory.
    temp_dirs = sorted(
        (
            path for path in INPUT_TIF.parent.glob("temp_*")
            if (path / "index.json").exists()
        ),
        key=lambda p: p.stat().st_mtime
    )

    if not temp_dirs:

        raise RuntimeError(
            "Could not find opensr-utils "
            "temporary output directory."
        )

    temp_dir = temp_dirs[-1]

    print()
    print(
        f"Using SR patches from:\n"
        f"{temp_dir}"
    )

    stitch_from_index(
        temp_dir,
        INPUT_TIF,
        OUTPUT_TIF
    )


if __name__ == "__main__":
    main()