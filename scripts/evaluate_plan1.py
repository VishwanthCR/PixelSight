import numpy as np
import torch
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pixelsight.models.swinir import SwinIR


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR_PATH = Path("dataset/region3/lr_patches.npy")
HR_PATH = Path("dataset/region3/hr_patches.npy")
MODEL_PATH = Path("checkpoints/swinir_best.pth")

THRESHOLD = 1e-8


# ============================================================
# Metrics
# ============================================================

def psnr_from_mse(mse):
    if mse == 0:
        return float("inf")

    return 10 * np.log10(1.0 / mse)


def sam_for_patch(pred, target):
    """
    Calculate mean Spectral Angle Mapper for one patch.
    """

    pred = pred.reshape(-1, 4)
    target = target.reshape(-1, 4)

    pred_norm = np.linalg.norm(pred, axis=1)
    target_norm = np.linalg.norm(target, axis=1)

    valid = (
        (pred_norm > THRESHOLD)
        & (target_norm > THRESHOLD)
    )

    if not np.any(valid):
        return None

    pred = pred[valid]
    target = target[valid]

    pred_norm = np.linalg.norm(pred, axis=1)
    target_norm = np.linalg.norm(target, axis=1)

    cosine = np.sum(pred * target, axis=1) / (
        pred_norm * target_norm
    )

    cosine = np.clip(cosine, -1.0, 1.0)

    angles = np.arccos(cosine)

    return np.mean(angles)


# ============================================================
# Load data
# ============================================================

print("=" * 60)
print("PixelSight SwinIR Evaluation")
print("=" * 60)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

lr_data = np.load(LR_PATH)
hr_data = np.load(HR_PATH)

print("LR shape:", lr_data.shape)
print("HR shape:", hr_data.shape)


# ============================================================
# Load model
# ============================================================

model = SwinIR(
    in_channels=4,
    out_channels=4,
    scale=2,
    embed_dim=60,
    depths=6,
    num_heads=6,
    window_size=8
).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Loaded checkpoint epoch:", checkpoint["epoch"])
print("Validation L1:", checkpoint["val_loss"])


# ============================================================
# Prediction storage
# ============================================================

predictions = []

sam_values = []

empty_patches = 0


# ============================================================
# Run inference
# ============================================================

print("\nRunning inference...")

with torch.no_grad():

    for i in range(len(lr_data)):

        lr = torch.from_numpy(
            lr_data[i]
        ).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):
            pred = model(lr)

        pred = (
            pred
            .squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        pred = np.clip(pred, 0, 1)

        predictions.append(pred)

        hr = hr_data[i]

        # Check whether patch is empty
        if np.all(hr <= THRESHOLD):
            empty_patches += 1

        # SAM
        angle = sam_for_patch(pred, hr)

        if angle is not None:
            sam_values.append(angle)


print("Inference complete.")


# ============================================================
# Convert predictions
# ============================================================

predictions = np.asarray(predictions)


# ============================================================
# ALL-PIXEL METRICS
# ============================================================

all_mse = np.mean(
    (predictions - hr_data) ** 2
)

all_psnr = psnr_from_mse(all_mse)


# ============================================================
# CONTENT-ONLY METRICS
# ============================================================

content_mask = np.any(
    hr_data > THRESHOLD,
    axis=3
)

content_pred = predictions[content_mask]
content_hr = hr_data[content_mask]

content_mse = np.mean(
    (content_pred - content_hr) ** 2
)

content_psnr = psnr_from_mse(content_mse)


# ============================================================
# SSIM
# ============================================================

ssim_values = []

for i in range(len(hr_data)):

    # Average SSIM across the 4 spectral bands
    band_ssim = []

    for band in range(4):

        value = ssim(
            hr_data[i, :, :, band],
            predictions[i, :, :, band],
            data_range=1.0
        )

        band_ssim.append(value)

    ssim_values.append(
        np.mean(band_ssim)
    )


mean_ssim = np.mean(ssim_values)


# ============================================================
# Content-only SSIM
# ============================================================

content_ssim_values = []

for i in range(len(hr_data)):

    mask = content_mask[i]

    if not np.any(mask):
        continue

    band_values = []

    for band in range(4):

        # SSIM is calculated on the full image because
        # it is a structural metric.
        value = ssim(
            hr_data[i, :, :, band],
            predictions[i, :, :, band],
            data_range=1.0
        )

        band_values.append(value)

    content_ssim_values.append(
        np.mean(band_values)
    )


content_ssim = np.mean(
    content_ssim_values
)


# ============================================================
# SAM
# ============================================================

mean_sam_rad = np.mean(sam_values)
mean_sam_deg = np.degrees(mean_sam_rad)


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 60)
print("SWINIR RESULTS — REGION 3")
print("=" * 60)

print("\nALL PIXELS")
print("-" * 40)
print(f"MSE  : {all_mse:.10f}")
print(f"PSNR : {all_psnr:.4f} dB")
print(f"SSIM : {mean_ssim:.6f}")

print("\nCONTENT ONLY")
print("-" * 40)
print(f"MSE  : {content_mse:.10f}")
print(f"PSNR : {content_psnr:.4f} dB")
print(f"SSIM : {content_ssim:.6f}")
print(f"SAM  : {mean_sam_rad:.6f} radians")
print(f"SAM  : {mean_sam_deg:.4f} degrees")

print("\nDATASET")
print("-" * 40)
print(f"Total patches : {len(hr_data)}")
print(f"Empty patches : {empty_patches}")
print(
    f"Content patches : "
    f"{len(hr_data) - empty_patches}"
)

print("=" * 60)