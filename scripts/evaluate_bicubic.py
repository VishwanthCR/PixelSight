import numpy as np
from pathlib import Path
from scipy.ndimage import zoom
from skimage.metrics import structural_similarity


# ============================================================
# CONFIG
# ============================================================

REGION = "region3"

dataset_dir = Path("dataset") / REGION

lr = np.load(dataset_dir / "lr_patches.npy")
hr = np.load(dataset_dir / "hr_patches.npy")

print("LR shape:", lr.shape)
print("HR shape:", hr.shape)


# ============================================================
# STORAGE
# ============================================================

all_squared_errors = []
content_squared_errors = []

ssim_all = []
ssim_content = []

sam_values = []

empty_patches = 0
content_patches = 0


# ============================================================
# SAM
# ============================================================

def calculate_sam(pred, target):

    pred_flat = pred.reshape(-1, 4)
    target_flat = target.reshape(-1, 4)

    pred_norm = np.linalg.norm(pred_flat, axis=1)
    target_norm = np.linalg.norm(target_flat, axis=1)

    denominator = pred_norm * target_norm

    valid = denominator > 1e-8

    if not np.any(valid):
        return np.nan

    numerator = np.sum(
        pred_flat[valid] * target_flat[valid],
        axis=1
    )

    cosine = numerator / denominator[valid]

    cosine = np.clip(cosine, -1.0, 1.0)

    angles = np.arccos(cosine)

    return np.mean(angles)


# ============================================================
# PROCESS DATA
# ============================================================

print("\nRunning final bicubic evaluation...\n")

for i in range(len(lr)):

    lr_patch = lr[i]
    hr_patch = hr[i]

    # --------------------------------------------------------
    # Bicubic 2×
    # --------------------------------------------------------

    prediction = zoom(
        lr_patch,
        zoom=(2, 2, 1),
        order=3
    ).astype(np.float32)

    prediction = np.clip(prediction, 0.0, 1.0)

    # --------------------------------------------------------
    # Pixel errors
    # --------------------------------------------------------

    error = hr_patch - prediction
    squared_error = error ** 2

    all_squared_errors.append(
        squared_error.ravel()
    )

    # --------------------------------------------------------
    # Content mask
    #
    # A pixel is considered meaningful if the HR spectral
    # vector contains non-zero information.
    # --------------------------------------------------------

    content_mask = np.any(
        hr_patch > 1e-8,
        axis=2
    )

    if np.any(content_mask):

        content_patches += 1

        content_error = squared_error[content_mask]

        content_squared_errors.append(
            content_error.ravel()
        )

        # ----------------------------------------------------
        # SSIM for content-containing patches
        # ----------------------------------------------------

        ssim = structural_similarity(
            hr_patch,
            prediction,
            data_range=1.0,
            channel_axis=2
        )

        ssim_content.append(ssim)

        # ----------------------------------------------------
        # SAM
        # ----------------------------------------------------

        sam = calculate_sam(
            prediction,
            hr_patch
        )

        if np.isfinite(sam):
            sam_values.append(sam)

    else:
        empty_patches += 1

        # SSIM of completely empty patch
        ssim = structural_similarity(
            hr_patch,
            prediction,
            data_range=1.0,
            channel_axis=2
        )

        ssim_all.append(ssim)


    # SSIM for every patch
    if np.any(content_mask):

        # Already calculated above
        ssim_all.append(ssim_content[-1])

    else:
        ssim_all.append(ssim)


    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(lr)}")


# ============================================================
# GLOBAL PSNR
# ============================================================

all_squared_errors = np.concatenate(
    all_squared_errors
)

content_squared_errors = np.concatenate(
    content_squared_errors
)

global_mse_all = np.mean(all_squared_errors)

global_mse_content = np.mean(
    content_squared_errors
)

psnr_all = 10 * np.log10(
    1.0 / global_mse_all
)

psnr_content = 10 * np.log10(
    1.0 / global_mse_content
)


# ============================================================
# SSIM
# ============================================================

mean_ssim_all = np.mean(ssim_all)

mean_ssim_content = np.mean(
    ssim_content
)


# ============================================================
# SAM
# ============================================================

sam_values = np.array(sam_values)

mean_sam_rad = np.mean(sam_values)

mean_sam_deg = np.degrees(
    mean_sam_rad
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 65)
print("PIXELSIGHT PLAN 1 — FINAL BICUBIC BASELINE")
print("=" * 65)

print(f"Test region              : {REGION}")
print(f"Total patches            : {len(lr)}")
print(f"Empty patches            : {empty_patches}")
print(f"Content patches          : {content_patches}")

print("\n--- ALL PIXELS ---")
print(f"Global MSE               : {global_mse_all:.10f}")
print(f"Global PSNR              : {psnr_all:.4f} dB")
print(f"Mean SSIM                : {mean_ssim_all:.6f}")

print("\n--- CONTENT ONLY ---")
print(f"Content MSE              : {global_mse_content:.10f}")
print(f"Content PSNR             : {psnr_content:.4f} dB")
print(f"Content SSIM             : {mean_ssim_content:.6f}")
print(f"Content SAM              : {mean_sam_rad:.6f} radians")
print(f"Content SAM              : {mean_sam_deg:.4f} degrees")

print("\n--- CHECK ---")
print(f"SAM-valid patches        : {len(sam_values)}")

print("=" * 65)