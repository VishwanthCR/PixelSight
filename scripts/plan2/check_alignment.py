from pathlib import Path
import rasterio

ROOT = Path("dataset/plan2/sentinel2")
BANDS = ("B02", "B03", "B04", "B08")

for split in ("train", "validation", "test"):
    print(f"\n=== {split.upper()} ===")

    files = sorted(
        ROOT.joinpath(split).glob("*_B*_10m.jp2")
    )

    reference = None

    for path in files:
        band = path.stem.split("_")[-2]

        with rasterio.open(path) as src:
            info = {
                "shape": (src.height, src.width),
                "transform": src.transform,
                "crs": src.crs,
                "bounds": src.bounds,
                "res": src.res,
            }

        print(f"{band}:")
        print(f"  shape  = {info['shape']}")
        print(f"  res    = {info['res']}")
        print(f"  CRS    = {info['crs']}")

        if reference is None:
            reference = info
        else:
            assert info["shape"] == reference["shape"], \
                f"{band}: shape mismatch"
            assert info["transform"] == reference["transform"], \
                f"{band}: transform mismatch"
            assert info["crs"] == reference["crs"], \
                f"{band}: CRS mismatch"
            assert info["bounds"] == reference["bounds"], \
                f"{band}: bounds mismatch"

    print("✓ All 10 m bands are perfectly aligned.")