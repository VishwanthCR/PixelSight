from pathlib import Path
import rasterio


SPLITS = ["train", "validation", "test"]

ROOT = Path("dataset/plan2/sentinel2")
OUT_ROOT = Path("dataset/plan2/geotiff")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

BANDS = ["B02", "B03", "B04", "B08"]


def find_band(folder, band):
    matches = list(folder.glob(f"*_{band}_10m.jp2"))

    if not matches:
        # Fallback for different naming conventions
        matches = list(folder.glob(f"*{band}*.jp2"))

    if not matches:
        raise FileNotFoundError(f"Could not find {band} in {folder}")

    return matches[0]


for split in SPLITS:
    folder = ROOT / split

    print(f"\nProcessing {split}...")

    band_files = {
        band: find_band(folder, band)
        for band in BANDS
    }

    for band, path in band_files.items():
        print(f"{band}: {path.name}")

    with rasterio.open(band_files["B02"]) as src:
        profile = src.profile.copy()

        profile.update(
            driver="GTiff",
            count=4,
            dtype="float32",
            compress="deflate",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        output = OUT_ROOT / f"{split}_B02_B03_B04_B08_10m.tif"

        with rasterio.open(output, "w", **profile) as dst:
            for idx, band in enumerate(BANDS, start=1):
                with rasterio.open(band_files[band]) as src_band:
                    data = src_band.read(1).astype("float32")

                    # Sentinel-2 L2A reflectance scaling
                    data /= 10000.0

                    data.clip(0.0, 1.0, out=data)

                    dst.write(data, idx)
                    dst.set_band_description(idx, band)

        print(f"Created: {output}")


print("\nAll GeoTIFFs created successfully.")