from pathlib import Path

import rasterio
from rasterio.windows import Window


INPUT = Path(
    "dataset/plan2/geotiff/train_B02_B03_B04_B08_10m.tif"
)

OUTPUT = Path(
    "dataset/plan2/geotiff/train_test_512_10m.tif"
)

SIZE = 512

BAND_NAMES = [
    "B02",
    "B03",
    "B04",
    "B08",
]


with rasterio.open(INPUT) as src:

    # Central 512 x 512 region
    x = (src.width - SIZE) // 2
    y = (src.height - SIZE) // 2

    window = Window(x, y, SIZE, SIZE)

    profile = src.profile.copy()

    profile.update(
        width=SIZE,
        height=SIZE,
        transform=src.window_transform(window),
        count=4,
        dtype="float32",
        compress="deflate",
        tiled=True,
    )

    with rasterio.open(OUTPUT, "w", **profile) as dst:

        for band in range(1, 5):

            data = src.read(
                band,
                window=window
            ).astype("float32")

            dst.write(data, band)

            dst.set_band_description(
                band,
                BAND_NAMES[band - 1]
            )


print("Created test tile:")
print(OUTPUT)
print(f"Size: {SIZE} x {SIZE}")
print("Input resolution: 10 m")
print("Bands: B02, B03, B04, B08")