"""Create RGB comparison figures from completed LDSR-S2 patch inference.

This script performs no model inference.  It reads existing 128 x 128
Sentinel-2 patches (10 m) and their completed 512 x 512 LDSR-S2 outputs
(nominal 2.5 m), then writes presentation-ready comparisons to results/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PATCH_DIR = ROOT / "dataset" / "plan2" / "patches" / "train"
SR_DIR = ROOT / "dataset" / "plan2" / "ldsr_s2" / "train"
OUTPUT_DIR = ROOT / "results" / "plan2" / "sample_comparisons"

# Five spatially distributed patches already completed with 100 LDSR steps.
SAMPLE_IDS = (0, 2, 4, 6, 8)
RGB_BANDS = (2, 1, 0)  # B04, B03, B02 in the project's B02/B03/B04/B08 order


def rgb_display(image: np.ndarray) -> np.ndarray:
    """Return a robustly stretched Sentinel-2 RGB image for display."""
    rgb = image[..., RGB_BANDS].astype(np.float32)
    low, high = np.percentile(rgb, (2, 98))
    if high <= low:
        return np.clip(rgb, 0.0, 1.0)
    return np.clip((rgb - low) / (high - low), 0.0, 1.0)


def load_sample(sample_id: int) -> tuple[np.ndarray, np.ndarray]:
    patch_name = f"patch_{sample_id:05d}"
    input_path = PATCH_DIR / f"{patch_name}.npz"
    sr_path = SR_DIR / f"{patch_name}_sr.npz"

    if not input_path.exists() or not sr_path.exists():
        raise FileNotFoundError(
            f"Completed input/output pair not found for {patch_name}."
        )

    with np.load(input_path) as source:
        input_image = source["image"]
    with np.load(sr_path) as source:
        sr_image = source["sr"]

    if input_image.shape != (128, 128, 4):
        raise ValueError(f"Unexpected input shape: {input_image.shape}")
    if sr_image.shape != (512, 512, 4):
        raise ValueError(f"Unexpected SR shape: {sr_image.shape}")

    return input_image, sr_image


def save_comparison(sample_id: int) -> Path:
    input_image, sr_image = load_sample(sample_id)
    input_rgb = rgb_display(input_image)
    sr_rgb = rgb_display(sr_image)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axes[0].imshow(input_rgb)
    axes[0].set_title("Sentinel-2 input\n10 m | 128 × 128")
    axes[1].imshow(sr_rgb)
    axes[1].set_title("LDSR-S2 output (100 steps)\nnominal 2.5 m | 512 × 512")

    for axis in axes:
        axis.set_axis_off()

    fig.suptitle(f"PixelSight Plan 2 — Sample {sample_id:05d}", fontsize=15)
    output_path = OUTPUT_DIR / f"sample_{sample_id:05d}_10m_vs_ldsr_s2_100steps.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for sample_id in SAMPLE_IDS:
        output_path = save_comparison(sample_id)
        print(f"Saved {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
