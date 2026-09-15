from pathlib import Path
import numpy as np
from scipy.ndimage import zoom


SR_FILE = Path("dataset/plan2/ldsr_test/patch_00000_sr.npy")
LR_FILE = Path("dataset/plan2/patches/train/patch_00000.npz")


def main():
    print("=== LDSR-S2 Result Inspection ===")

    # Load SR
    sr = np.load(SR_FILE)

    # Load original 128x128 input
    with np.load(LR_FILE) as data:
        image = data["image"].astype(np.float32)
        mask = data["mask"].astype(bool)

    print(f"\nInput shape : {image.shape}")
    print(f"SR shape    : {sr.shape}")

    print("\n--- Validity ---")
    print(f"SR finite   : {np.isfinite(sr).all()}")
    print(f"Input finite: {np.isfinite(image).all()}")

    # Expected shape
    if sr.shape != (512, 512, 4):
        raise RuntimeError(
            f"Unexpected SR shape: {sr.shape}"
        )

    print("\n--- Overall statistics ---")
    print(
        f"Input : min={image.min():.6f}, "
        f"max={image.max():.6f}, "
        f"mean={image.mean():.6f}, "
        f"std={image.std():.6f}"
    )

    print(
        f"SR    : min={sr.min():.6f}, "
        f"max={sr.max():.6f}, "
        f"mean={sr.mean():.6f}, "
        f"std={sr.std():.6f}"
    )

    # Per-band statistics
    bands = ["B02", "B03", "B04", "B08"]

    print("\n--- Per-band statistics ---")

    for i, band in enumerate(bands):

        inp = image[:, :, i]
        out = sr[:, :, i]

        print(
            f"{band}: "
            f"input mean={inp.mean():.6f}, "
            f"input std={inp.std():.6f} | "
            f"SR mean={out.mean():.6f}, "
            f"SR std={out.std():.6f}"
        )

    # Downsample SR back to 128x128
    # This gives us a rough consistency check.
    sr_down = zoom(
        sr,
        zoom=(0.25, 0.25, 1),
        order=1
    ).astype(np.float32)

    difference = sr_down - image

    print("\n--- Downsample consistency ---")

    print(
        f"Mean absolute difference: "
        f"{np.mean(np.abs(difference)):.8f}"
    )

    print(
        f"Maximum absolute difference: "
        f"{np.max(np.abs(difference)):.8f}"
    )

    print(
        f"RMSE: "
        f"{np.sqrt(np.mean(difference ** 2)):.8f}"
    )

    # Spatial variation at SR resolution
    print("\n--- SR spatial variation ---")

    for i, band in enumerate(bands):

        out = sr[:, :, i]

        horizontal = np.mean(
            np.abs(
                out[:, 1:] -
                out[:, :-1]
            )
        )

        vertical = np.mean(
            np.abs(
                out[1:, :] -
                out[:-1, :]
            )
        )

        print(
            f"{band}: "
            f"horizontal={horizontal:.8f}, "
            f"vertical={vertical:.8f}"
        )

    # Mask information
    print("\n--- Mask ---")
    print(
        f"Valid pixels: "
        f"{mask.mean() * 100:.2f}%"
    )

    print("\nInspection complete.")


if __name__ == "__main__":
    main()