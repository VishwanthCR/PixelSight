from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PATCH_DIR = ROOT / "dataset" / "plan2" / "patches" / "train"
SR_DIR = ROOT / "dataset" / "plan2" / "ldsr_s2" / "train"
OUTPUT_DIR = ROOT / "results" / "plan2" / "sample_comparisons"

# Generate 15 samples:
# 0, 2, 4, ..., 28
SAMPLE_IDS = tuple(range(10))

# RGB = B04, B03, B02
RGB_BANDS = (2, 1, 0)


def rgb_display(image):
    """
    Convert a 4-band Sentinel-2 image into an RGB display
    using percentile stretching.
    """
    rgb = image[:, :, RGB_BANDS].astype(np.float32)

    output = np.zeros_like(rgb)

    for channel in range(3):
        band = rgb[:, :, channel]

        low = np.percentile(band, 2)
        high = np.percentile(band, 98)

        if high > low:
            output[:, :, channel] = (band - low) / (high - low)
        else:
            output[:, :, channel] = 0

    return np.clip(output, 0, 1)


def load_sample(sample_id):
    """
    Load one LR Sentinel-2 patch and its corresponding
    LDSR-S2 super-resolved patch.
    """

    patch_path = PATCH_DIR / f"patch_{sample_id:05d}.npz"
    sr_path = SR_DIR / f"patch_{sample_id:05d}_sr.npz"

    if not patch_path.exists():
        print(f"Skipping {sample_id}: LR patch not found")
        return None, None

    if not sr_path.exists():
        print(f"Skipping {sample_id}: SR output not found")
        return None, None

    patch_data = np.load(patch_path)
    sr_data = np.load(sr_path)

    # Load image arrays
    lr = patch_data["image"]
    sr = sr_data["sr"]

    # Make sure dimensions are H x W x C
    if lr.ndim != 3:
        raise ValueError(
            f"Unexpected LR shape for sample {sample_id}: {lr.shape}"
        )

    if sr.ndim != 3:
        raise ValueError(
            f"Unexpected SR shape for sample {sample_id}: {sr.shape}"
        )

    print(
        f"Sample {sample_id:05d}: "
        f"LR={lr.shape}, SR={sr.shape}"
    )

    return lr, sr


def save_comparison(sample_id, lr, sr):
    """
    Create and save a side-by-side comparison:
    Sentinel-2 10m vs LDSR-S2 ~2.5m.
    """

    lr_rgb = rgb_display(lr)
    sr_rgb = rgb_display(sr)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7)
    )

    axes[0].imshow(lr_rgb)
    axes[0].set_title(
        "Sentinel-2 Input\n10 m"
    )
    axes[0].axis("off")

    axes[1].imshow(sr_rgb)
    axes[1].set_title(
        "LDSR-S2 Super-Resolution\n~2.5 m"
    )
    axes[1].axis("off")

    fig.suptitle(
        f"PixelSight Plan 2 — Sample {sample_id:05d}",
        fontsize=16
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"sample_{sample_id:05d}_10m_vs_ldsr_s2_100steps.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved {output_path}")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("PixelSight Plan 2")
    print("LDSR-S2 Sample Comparisons")
    print("=" * 60)

    print(f"LR patches : {PATCH_DIR}")
    print(f"SR patches : {SR_DIR}")
    print(f"Output     : {OUTPUT_DIR}")

    generated = 0

    for sample_id in SAMPLE_IDS:

        lr, sr = load_sample(sample_id)

        if lr is None or sr is None:
            continue

        save_comparison(
            sample_id,
            lr,
            sr
        )

        generated += 1

    print()
    print("=" * 60)
    print(f"Generated {generated} comparison images")
    print(f"Saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()