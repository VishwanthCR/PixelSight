"""Read-only readiness check for the supported PixelSight Plan 2 workflow."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pixelsight.data.plan2.sentinel2 import REQUIRED_BANDS  # noqa: E402


PATCH_DIR = ROOT / "dataset" / "plan2" / "patches" / "train"
SR_DIR = ROOT / "dataset" / "plan2" / "ldsr_s2" / "train"
FULL_SCENE_DIR = ROOT / "dataset" / "plan2" / "ldsr_s2_test"


def check_completed_sample(sample_id: int) -> None:
    name = f"patch_{sample_id:05d}"
    patch_path = PATCH_DIR / f"{name}.npz"
    sr_path = SR_DIR / f"{name}_sr.npz"

    if not patch_path.is_file() or not sr_path.is_file():
        raise FileNotFoundError(f"Missing completed pair for {name}.")

    with np.load(patch_path) as source:
        image = source["image"]
    with np.load(sr_path) as source:
        sr = source["sr"]

    if image.shape != (128, 128, 4) or sr.shape != (512, 512, 4):
        raise ValueError(f"Invalid shapes for {name}: {image.shape}, {sr.shape}")
    if not np.isfinite(image).all() or not np.isfinite(sr).all():
        raise ValueError(f"Non-finite values in {name}.")


def main() -> None:
    print("PixelSight Plan 2 readiness check")
    print(f"Required bands: {', '.join(REQUIRED_BANDS)}")

    for sample_id in (0, 2, 4, 6, 8):
        check_completed_sample(sample_id)
    print("Completed 100-step patch pairs: OK (5 gallery samples)")

    gallery = ROOT / "results" / "plan2" / "sample_comparisons"
    figures = list(gallery.glob("sample_*_10m_vs_ldsr_s2_100steps.png"))
    if len(figures) != 5:
        raise FileNotFoundError("Expected five gallery figures under results/plan2.")
    print("Results gallery: OK (5 PNG files)")

    full_scene_files = (
        "train_test_512_ldsr_s2_2p5m.tif",
        "train_test_512_ldsr_s2_2p5m_10steps.tif",
    )
    missing = [name for name in full_scene_files if not (FULL_SCENE_DIR / name).is_file()]
    if missing:
        print("Full-scene 10-vs-100 comparison: unavailable (missing GeoTIFF outputs)")
    else:
        print("Full-scene 10-vs-100 comparison: inputs available")

    print("Plan 2 supported patch workflow: READY")


if __name__ == "__main__":
    main()
