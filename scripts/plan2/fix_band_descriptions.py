from pathlib import Path
import rasterio

TILE_ROOT = Path("dataset/plan2/geotiff/tiles")

BAND_DESCRIPTIONS = (
    "B02",
    "B03",
    "B04",
    "B08",
)


def fix_file(path: Path):
    with rasterio.open(path, "r+") as src:
        if src.count != 4:
            print(f"SKIP {path} - expected 4 bands, found {src.count}")
            return

        src.descriptions = BAND_DESCRIPTIONS

    print(f"FIXED: {path}")


def main():
    files = sorted(TILE_ROOT.rglob("*.tif"))

    print(f"Found {len(files)} TIFF files")

    for path in files:
        fix_file(path)

    print("\nDone.")


if __name__ == "__main__":
    main()