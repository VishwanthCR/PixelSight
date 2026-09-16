from pathlib import Path
import math
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine


# ============================================================
# PixelSight - GeoTIFF 512x512 Tiler
# ============================================================

TILE_SIZE = 512

INPUTS = {
    "train": Path("dataset/plan2/geotiff/train_B02_B03_B04_B08_10m.tif"),
    "validation": Path("dataset/plan2/geotiff/validation_B02_B03_B04_B08_10m.tif"),
}

OUTPUT_ROOT = Path("dataset/plan2/geotiff/tiles")


def tile_geotiff(input_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Input : {input_path}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    with rasterio.open(input_path) as src:

        width = src.width
        height = src.height

        print(f"Size       : {width} x {height}")
        print(f"Bands      : {src.count}")
        print(f"Resolution : {src.res}")
        print(f"CRS        : {src.crs}")

        # Number of tiles needed
        cols = math.ceil(width / TILE_SIZE)
        rows = math.ceil(height / TILE_SIZE)

        print(f"Tile size  : {TILE_SIZE} x {TILE_SIZE}")
        print(f"Grid       : {rows} rows x {cols} columns")
        print(f"Max tiles  : {rows * cols}")
        print()

        tile_number = 0

        for row in range(rows):
            for col in range(cols):

                x = col * TILE_SIZE
                y = row * TILE_SIZE

                # Actual window dimensions at image edges
                window_width = min(TILE_SIZE, width - x)
                window_height = min(TILE_SIZE, height - y)

                window = Window(
                    col_off=x,
                    row_off=y,
                    width=window_width,
                    height=window_height,
                )

                # Read only this window
                data = src.read(window=window)

                # ------------------------------------------------
                # Pad edge tiles to exactly 512 x 512
                # ------------------------------------------------
                padded = src.read(
                    window=window,
                    boundless=True,
                    fill_value=0,
                    out_shape=(src.count, TILE_SIZE, TILE_SIZE),
                )

                # Correct geotransform for this tile
                transform = src.window_transform(window)

                output_path = output_dir / (
                    f"tile_r{row:02d}_c{col:02d}.tif"
                )

                profile = src.profile.copy()

                profile.update(
                    {
                        "driver": "GTiff",
                        "height": TILE_SIZE,
                        "width": TILE_SIZE,
                        "transform": transform,
                        "compress": "deflate",
                        "predictor": 2,
                        "tiled": True,
                    }
                )

                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(padded)

                tile_number += 1

                print(
                    f"[{tile_number:03d}/{rows * cols}] "
                    f"row={row:02d} col={col:02d} "
                    f"-> {output_path.name}"
                )

    print()
    print(f"Finished: {tile_number} tiles")
    print()


def main():
    for name, input_path in INPUTS.items():

        if not input_path.exists():
            print(f"ERROR: Input file not found:")
            print(f"       {input_path}")
            continue

        output_dir = OUTPUT_ROOT / name

        tile_geotiff(input_path, output_dir)

    print("=" * 70)
    print("ALL TILING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()