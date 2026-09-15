from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import zoom


ROOT = Path(__file__).resolve().parents[2]

NATIVE_FILE = ROOT / "dataset/plan2/geotiff/train_test_512_10m.tif"
SR_FILE = ROOT / "dataset/plan2/geotiff/sr.tif"
WORLDCOVER_FILE = (
    ROOT
    / "dataset/segmentation/aligned_labels/"
    / "train_B02_B03_B04_B08_10m_worldcover_aligned.tif"
)

OUTPUT_DIR = ROOT / "results/plan4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCALE = 4

# Sentinel-2 band indices in the 4-band stack
RED = 2       # B04
NIR = 3       # B08

BUILT_UP = 50
CROPLAND = 40
IGNORE_CLASSES = {70, 90, 95, 100}


def read_raster(path, expected_shape):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        bounds = src.bounds
        transform = src.transform
        crs = src.crs

    if data.shape != expected_shape:
        raise ValueError(
            f"{path.name}: expected {expected_shape}, got {data.shape}"
        )

    return data, bounds, transform, crs


def read_native():
    image, bounds, transform, crs = read_raster(
        NATIVE_FILE,
        (4, 512, 512),
    )
    return np.clip(image, 0.0, 1.0), bounds, transform, crs


def read_sr():
    image, bounds, transform, crs = read_raster(
        SR_FILE,
        (4, 2048, 2048),
    )

    # sr.tif stores reflectance scaled by 10000.
    image /= 10000.0

    return np.clip(image, 0.0, 1.0), bounds, transform, crs


def create_bicubic(native):
    result = np.empty(
        (4, 2048, 2048),
        dtype=np.float32,
    )

    for b in range(4):
        result[b] = zoom(
            native[b],
            4,
            order=3,
        )

    return np.clip(result, 0.0, 1.0)


def read_worldcover(bounds):
    with rasterio.open(WORLDCOVER_FILE) as src:

        window = from_bounds(
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
            transform=src.transform,
        )

        window = window.round_offsets().round_lengths()

        label = src.read(1, window=window)

    if label.shape != (512, 512):
        raise ValueError(
            f"Unexpected WorldCover shape: {label.shape}"
        )

    return label


def upsample_mask(mask):
    return np.repeat(
        np.repeat(mask, SCALE, axis=0),
        SCALE,
        axis=1,
    )


def ndvi(image):
    red = image[RED]
    nir = image[NIR]

    denominator = nir + red

    output = np.zeros_like(red, dtype=np.float32)

    valid = denominator > 1e-8

    output[valid] = (
        (nir[valid] - red[valid])
        / denominator[valid]
    )

    return output


def ndvi_stats(values, mask=None):
    valid = np.isfinite(values)

    if mask is not None:
        valid &= mask

    values = values[valid]

    if values.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "pixels": 0,
        }

    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "pixels": int(values.size),
    }


def downsample_4x(arr):
    h, w = arr.shape

    return arr.reshape(
        h // 4,
        4,
        w // 4,
        4,
    ).mean(axis=(1, 3))


def spatial_gradient_energy(image):
    """
    Simple high-frequency spatial-detail measure.

    Mean gradient magnitude across the four bands.
    This is NOT a perceptual quality score.
    It is only a relative spatial-detail indicator.
    """

    energies = []

    for band in image:

        gx = np.diff(band, axis=1)
        gy = np.diff(band, axis=0)

        energy = (
            np.mean(np.abs(gx))
            + np.mean(np.abs(gy))
        ) / 2.0

        energies.append(float(energy))

    return float(np.mean(energies))


def main():

    print("=" * 70)
    print("PixelSight P4")
    print("Urban + Crop Application Analysis")
    print("=" * 70)

    for path in [
        NATIVE_FILE,
        SR_FILE,
        WORLDCOVER_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

        print("Found:", path)

    # ---------------------------------------------------------
    # Load imagery
    # ---------------------------------------------------------

    print("\nLoading native 10m...")
    native, native_bounds, _, _ = read_native()

    print("Loading LDSR 2.5m...")
    ldsr, ldsr_bounds, _, _ = read_sr()

    if not np.allclose(
        [
            native_bounds.left,
            native_bounds.bottom,
            native_bounds.right,
            native_bounds.top,
        ],
        [
            ldsr_bounds.left,
            ldsr_bounds.bottom,
            ldsr_bounds.right,
            ldsr_bounds.top,
        ],
    ):
        raise ValueError(
            "Native and LDSR geographic bounds do not match."
        )

    print("Geographic alignment: VERIFIED")

    print("\nCreating bicubic 2.5m baseline...")
    bicubic = create_bicubic(native)

    # ---------------------------------------------------------
    # WorldCover
    # ---------------------------------------------------------

    print("\nLoading matching WorldCover...")
    worldcover = read_worldcover(native_bounds)

    built_mask_10m = worldcover == BUILT_UP
    crop_mask_10m = worldcover == CROPLAND

    built_mask_2p5m = upsample_mask(built_mask_10m)
    crop_mask_2p5m = upsample_mask(crop_mask_10m)

    print(
        "Built-up pixels:",
        int(built_mask_10m.sum())
    )

    print(
        "Cropland pixels:",
        int(crop_mask_10m.sum())
    )

    # ---------------------------------------------------------
    # NDVI
    # ---------------------------------------------------------

    print("\nCalculating NDVI...")

    native_ndvi = ndvi(native)
    bicubic_ndvi = ndvi(bicubic)
    ldsr_ndvi = ndvi(ldsr)

    native_crop_stats = ndvi_stats(
        native_ndvi,
        crop_mask_10m,
    )

    bicubic_crop_stats = ndvi_stats(
        bicubic_ndvi,
        crop_mask_2p5m,
    )

    ldsr_crop_stats = ndvi_stats(
        ldsr_ndvi,
        crop_mask_2p5m,
    )

    ndvi_table = pd.DataFrame([
        {
            "representation": "Native 10m",
            "ndvi_mean": native_crop_stats["mean"],
            "ndvi_std": native_crop_stats["std"],
            "ndvi_min": native_crop_stats["min"],
            "ndvi_max": native_crop_stats["max"],
            "pixels": native_crop_stats["pixels"],
        },
        {
            "representation": "Bicubic 2.5m",
            "ndvi_mean": bicubic_crop_stats["mean"],
            "ndvi_std": bicubic_crop_stats["std"],
            "ndvi_min": bicubic_crop_stats["min"],
            "ndvi_max": bicubic_crop_stats["max"],
            "pixels": bicubic_crop_stats["pixels"],
        },
        {
            "representation": "LDSR 2.5m",
            "ndvi_mean": ldsr_crop_stats["mean"],
            "ndvi_std": ldsr_crop_stats["std"],
            "ndvi_min": ldsr_crop_stats["min"],
            "ndvi_max": ldsr_crop_stats["max"],
            "pixels": ldsr_crop_stats["pixels"],
        },
    ])

    print("\nCrop NDVI statistics:")
    print(ndvi_table.to_string(index=False))

    ndvi_table.to_csv(
        OUTPUT_DIR / "crop_ndvi_comparison.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # NDVI consistency at native resolution
    # ---------------------------------------------------------

    ldsr_ndvi_10m = downsample_4x(ldsr_ndvi)
    bicubic_ndvi_10m = downsample_4x(bicubic_ndvi)

    ldsr_mae = np.mean(
        np.abs(ldsr_ndvi_10m - native_ndvi)
    )

    bicubic_mae = np.mean(
        np.abs(bicubic_ndvi_10m - native_ndvi)
    )

    ldsr_rmse = np.sqrt(
        np.mean(
            (ldsr_ndvi_10m - native_ndvi) ** 2
        )
    )

    bicubic_rmse = np.sqrt(
        np.mean(
            (bicubic_ndvi_10m - native_ndvi) ** 2
        )
    )

    ndvi_consistency = pd.DataFrame([
        {
            "method": "LDSR 2.5m",
            "NDVI_MAE_vs_native": ldsr_mae,
            "NDVI_RMSE_vs_native": ldsr_rmse,
        },
        {
            "method": "Bicubic 2.5m",
            "NDVI_MAE_vs_native": bicubic_mae,
            "NDVI_RMSE_vs_native": bicubic_rmse,
        },
    ])

    print("\nNDVI consistency:")
    print(ndvi_consistency.to_string(index=False))

    ndvi_consistency.to_csv(
        OUTPUT_DIR / "crop_ndvi_consistency.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Urban built-up spectral statistics
    # ---------------------------------------------------------

    print("\nUrban built-up analysis...")

    native_built = native[:, built_mask_10m]

    bicubic_built = bicubic[:, built_mask_2p5m]
    ldsr_built = ldsr[:, built_mask_2p5m]

    urban_rows = []

    for name, values in [
        ("Native 10m", native_built),
        ("Bicubic 2.5m", bicubic_built),
        ("LDSR 2.5m", ldsr_built),
    ]:

        for band_idx, band_name in enumerate(
            ["B02", "B03", "B04", "B08"]
        ):
            urban_rows.append(
                {
                    "representation": name,
                    "band": band_name,
                    "mean_reflectance": float(
                        values[band_idx].mean()
                    ),
                    "std_reflectance": float(
                        values[band_idx].std()
                    ),
                }
            )

    urban_table = pd.DataFrame(urban_rows)

    print(
        "\nBuilt-up spectral statistics:"
    )
    print(
        urban_table.to_string(index=False)
    )

    urban_table.to_csv(
        OUTPUT_DIR / "urban_builtup_spectral_comparison.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Spatial detail
    # ---------------------------------------------------------

    print("\nSpatial-detail analysis...")

    native_detail = spatial_gradient_energy(native)
    bicubic_detail = spatial_gradient_energy(bicubic)
    ldsr_detail = spatial_gradient_energy(ldsr)

    detail_table = pd.DataFrame([
        {
            "representation": "Native 10m",
            "gradient_energy": native_detail,
        },
        {
            "representation": "Bicubic 2.5m",
            "gradient_energy": bicubic_detail,
        },
        {
            "representation": "LDSR 2.5m",
            "gradient_energy": ldsr_detail,
        },
    ])

    print(
        detail_table.to_string(index=False)
    )

    detail_table.to_csv(
        OUTPUT_DIR / "spatial_detail_comparison.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Save arrays
    # ---------------------------------------------------------

    np.save(
        OUTPUT_DIR / "native_ndvi.npy",
        native_ndvi,
    )

    np.save(
        OUTPUT_DIR / "bicubic_2p5m_ndvi.npy",
        bicubic_ndvi,
    )

    np.save(
        OUTPUT_DIR / "ldsr_2p5m_ndvi.npy",
        ldsr_ndvi,
    )

    np.save(
        OUTPUT_DIR / "worldcover_10m.npy",
        worldcover,
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    summary = pd.DataFrame([
        {
            "application": "Crop monitoring",
            "comparison": "LDSR vs Bicubic",
            "metric": "NDVI MAE vs native",
            "LDSR": float(ldsr_mae),
            "Bicubic": float(bicubic_mae),
            "lower_is_better": True,
        },
        {
            "application": "Crop monitoring",
            "comparison": "LDSR vs Bicubic",
            "metric": "NDVI RMSE vs native",
            "LDSR": float(ldsr_rmse),
            "Bicubic": float(bicubic_rmse),
            "lower_is_better": True,
        },
        {
            "application": "Urban analysis",
            "comparison": "Spatial detail",
            "metric": "Mean gradient energy",
            "LDSR": float(ldsr_detail),
            "Bicubic": float(bicubic_detail),
            "lower_is_better": False,
        },
    ])

    summary.to_csv(
        OUTPUT_DIR / "p4_application_summary.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("P4 APPLICATION ANALYSIS COMPLETE")
    print("=" * 70)
    print("Results:", OUTPUT_DIR)


if __name__ == "__main__":
    main()