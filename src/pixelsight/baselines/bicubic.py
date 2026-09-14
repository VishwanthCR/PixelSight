import numpy as np
from pathlib import Path
from scipy.ndimage import zoom
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ============================================================
# CONFIGURATION
# ============================================================

REGION = "region3"

dataset_dir = Path("dataset") / REGION

lr = np.load(dataset_dir / "lr_patches.npy")
hr = np.load(dataset_dir / "hr_patches.npy")

print("Loading dataset...")
print("LR shape:", lr.shape)
print("HR shape:", hr.shape)


# ============================================================
# SAM
# ============================================================

def calculate_sam(pred, target):

    pred_flat = pred.reshape(-1, 4)
    target_flat = target.reshape(-1, 4)

    pred_norm = np.linalg.norm(pred_flat, axis=1)
    target_norm = np.linalg.norm(target_flat, axis=1)

    denominator = pred_norm * target_norm

    # Only calculate SAM where both spectra are non-zero
    valid = denominator > 1e-8

    if not np.any(valid):
        return np.nan

    numerator = np.sum(
        pred_flat[valid] * target_flat[valid],
        axis=1
    )

    cosine = numerator / denominator[valid]

    # Numerical safety
    cosine = np.clip(cosine, -1.0, 1.0)

    angles = np.arccos(cosine)

    return np.mean(angles)


# ============================================================
# STORAGE
# ============================================================

psnr_values = []
ssim_values = []
sam_values = []

zero_mse_count = 0


# ============================================================
# BICUBIC BASELINE
# ============================================================

print("\nRunning corrected bicubic baseline...\n")

for i in range(len(lr)):

    lr_patch = lr[i]
    hr_patch = hr[i]

    # --------------------------------------------------------
    # Bicubic upscale ALL 4 bands
    #
    # zoom factor:
    # 32 -> 64
    # --------------------------------------------------------

    prediction = zoom(
        lr_patch,
        zoom=(2, 2, 1),
        order=3
    ).astype(np.float32)

    # Keep values inside expected range
    prediction = np.clip(prediction, 0.0, 1.0)

    # --------------------------------------------------------
    # MSE
    # --------------------------------------------------------

    mse = np.mean((hr_patch - prediction) ** 2)

    if mse == 0:
        zero_mse_count += 1
        psnr = np.inf
    else:
        psnr = peak_signal_noise_ratio(
            hr_patch,
            prediction,
            data_range=1.0
        )

    # --------------------------------------------------------
    # SSIM
    # --------------------------------------------------------

    ssim = structural_similarity(
        hr_patch,
        prediction,
        data_range=1.0,
        channel_axis=2
    )

    # --------------------------------------------------------
    # SAM
    # --------------------------------------------------------

    sam = calculate_sam(
        prediction,
        hr_patch
    )

    psnr_values.append(psnr)
    ssim_values.append(ssim)
    sam_values.append(sam)

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(lr)}")


# ============================================================
# CONVERT TO ARRAYS
# ============================================================

psnr_values = np.array(psnr_values)
ssim_values = np.array(ssim_values)
sam_values = np.array(sam_values)


# ============================================================
# ROBUST STATISTICS
# ============================================================

finite_psnr = psnr_values[np.isfinite(psnr_values)]
valid_sam = sam_values[np.isfinite(sam_values)]

mean_psnr = np.mean(finite_psnr)
mean_ssim = np.mean(ssim_values)
mean_sam_rad = np.mean(valid_sam)
mean_sam_deg = np.degrees(mean_sam_rad)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("PIXELSIGHT PLAN 1 — CORRECTED BICUBIC BASELINE")
print("=" * 60)

print(f"Test region        : {REGION}")
print(f"Samples             : {len(lr)}")

print()
print(f"PSNR                : {mean_psnr:.4f} dB")
print(f"SSIM                : {mean_ssim:.6f}")
print(f"SAM                 : {mean_sam_rad:.6f} radians")
print(f"SAM                 : {mean_sam_deg:.4f} degrees")

print()
print(f"Zero-MSE patches    : {zero_mse_count}")
print(f"Valid SAM patches   : {len(valid_sam)}")

print("=" * 60)