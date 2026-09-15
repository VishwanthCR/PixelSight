from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


PATCH_ROOT = Path("dataset/plan2/patches")
OUTPUT_ROOT = Path("dataset/plan2/pairs")

SCALE = 4
GAUSSIAN_SIGMA = 1.0


def create_lr(hr):
    """
    Create a synthetic 4x LR observation from a native-resolution HR patch.

    HR:
        (128, 128, 4)

    LR:
        (32, 32, 4)

    Gaussian blur is applied before 4x downsampling.
    """

    # Blur only spatial dimensions.
    blurred = gaussian_filter(
        hr,
        sigma=(GAUSSIAN_SIGMA, GAUSSIAN_SIGMA, 0),
        mode="reflect",
    )

    # Downsample spatial dimensions by 4.
    lr = zoom(
        blurred,
        zoom=(1 / SCALE, 1 / SCALE, 1),
        order=3,
        mode="reflect",
        prefilter=True,
    )

    return np.clip(
        lr.astype(np.float32),
        0.0,
        1.0,
    )


def process_split(split):
    input_dir = PATCH_ROOT / split
    output_dir = OUTPUT_ROOT / split

    lr_dir = output_dir / "lr"
    hr_dir = output_dir / "hr"
    mask_dir = output_dir / "mask"

    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.npz"))

    if not files:
        raise RuntimeError(
            f"No patches found in {input_dir}"
        )

    print(f"\n=== {split.upper()} ===")
    print(f"Input patches: {len(files)}")

    for index, path in enumerate(files):

        with np.load(path) as data:
            hr = data["image"].astype(np.float32)
            mask = data["mask"].astype(bool)

        if hr.shape != (128, 128, 4):
            raise RuntimeError(
                f"Unexpected HR shape in {path}: {hr.shape}"
            )

        if mask.shape != (128, 128):
            raise RuntimeError(
                f"Unexpected mask shape in {path}: {mask.shape}"
            )

        lr = create_lr(hr)

        if lr.shape != (32, 32, 4):
            raise RuntimeError(
                f"Unexpected LR shape in {path}: {lr.shape}"
            )

        name = f"patch_{index:05d}.npy"

        np.save(
            lr_dir / name,
            lr,
        )

        np.save(
            hr_dir / name,
            hr,
        )

        np.save(
            mask_dir / name,
            mask,
        )

        if (index + 1) % 500 == 0:
            print(
                f"Processed {index + 1}/{len(files)}"
            )

    print(f"LR shape: {lr.shape}")
    print(f"HR shape: {hr.shape}")
    print(f"Saved: {len(files)} pairs")


def main():
    print("PixelSight Plan 2 LR/HR Pair Generator")
    print(f"Scale factor: {SCALE}x")
    print(f"Gaussian sigma: {GAUSSIAN_SIGMA}")

    for split in (
        "train",
        "validation",
        "test",
    ):
        process_split(split)

    print("\nLR/HR pair generation complete.")


if __name__ == "__main__":
    main()