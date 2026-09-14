import sys
from pathlib import Path

# ============================================================
# Project path
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from pixelsight.models.swinir_v2 import SwinIRResidual


# ============================================================
# Configuration
# ============================================================

LR_PATH = (
    ROOT
    / "dataset"
    / "region3"
    / "lr_patches.npy"
)

HR_PATH = (
    ROOT
    / "dataset"
    / "region3"
    / "hr_patches.npy"
)

MODEL_PATH = (
    ROOT
    / "checkpoints"
    / "swinir_v2_best.pth"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Spectral Angle Mapper
# ============================================================

def calculate_sam(
    prediction,
    target
):
    """
    Calculate mean Spectral Angle Mapper.

    prediction: H x W x C
    target:     H x W x C

    Returns SAM in radians.
    """

    pred = prediction.reshape(
        -1,
        prediction.shape[-1]
    )

    true = target.reshape(
        -1,
        target.shape[-1]
    )

    pred_norm = np.linalg.norm(
        pred,
        axis=1
    )

    true_norm = np.linalg.norm(
        true,
        axis=1
    )

    valid = (
        (pred_norm > 1e-8)
        &
        (true_norm > 1e-8)
    )

    if not np.any(valid):
        return np.nan

    pred = pred[valid]
    true = true[valid]

    numerator = np.sum(
        pred * true,
        axis=1
    )

    denominator = (
        np.linalg.norm(
            pred,
            axis=1
        )
        *
        np.linalg.norm(
            true,
            axis=1
        )
    )

    cosine = (
        numerator
        /
        np.maximum(
            denominator,
            1e-12
        )
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    angles = np.arccos(
        cosine
    )

    return float(
        np.mean(angles)
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("PixelSight SwinIR v2 Evaluation")
    print("=" * 60)

    print(
        "Device:",
        DEVICE
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ========================================================
    # Load Region 3
    # ========================================================

    print("\nLoading Region 3...")

    lr = np.load(
        LR_PATH
    ).astype(
        np.float32
    )

    hr = np.load(
        HR_PATH
    ).astype(
        np.float32
    )

    print(
        "LR shape:",
        lr.shape
    )

    print(
        "HR shape:",
        hr.shape
    )

    # ========================================================
    # Load model
    # ========================================================

    model = SwinIRResidual(
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

    print(
        "Loaded checkpoint epoch:",
        checkpoint["epoch"]
    )

    print(
        "Validation L1:",
        checkpoint["val_loss"]
    )

    # ========================================================
    # Storage
    # ========================================================

    all_predictions = []

    print("\nRunning inference...")

    # ========================================================
    # Inference
    # ========================================================

    with torch.no_grad():

        for i in range(
            len(lr)
        ):

            lr_patch = torch.from_numpy(
                lr[i]
            ).permute(
                2,
                0,
                1
            ).unsqueeze(
                0
            ).to(DEVICE)

            prediction = model(
                lr_patch
            )

            prediction = prediction.squeeze(
                0
            ).permute(
                1,
                2,
                0
            ).cpu().numpy()

            # Keep output in the valid reflectance range
            prediction = np.clip(
                prediction,
                0.0,
                1.0
            )

            all_predictions.append(
                prediction.astype(
                    np.float32
                )
            )

    predictions = np.stack(
        all_predictions
    )

    print(
        "Inference complete."
    )

    # ========================================================
    # Dataset information
    # ========================================================

    empty_mask = np.all(
        np.abs(hr) < 1e-8,
        axis=(1, 2, 3)
    )

    empty_count = int(
        np.sum(empty_mask)
    )

    content_mask = ~empty_mask

    content_count = int(
        np.sum(content_mask)
    )

    # ========================================================
    # ALL PIXELS
    # ========================================================

    squared_error = (
        predictions - hr
    ) ** 2

    global_mse = float(
        np.mean(
            squared_error
        )
    )

    global_psnr = peak_signal_noise_ratio(
        hr,
        predictions,
        data_range=1.0
    )

    # Calculate SSIM patch by patch
    all_ssim = []

    for i in range(
        len(hr)
    ):

        score = structural_similarity(
            hr[i],
            predictions[i],
            channel_axis=2,
            data_range=1.0
        )

        all_ssim.append(
            score
        )

    global_ssim = float(
        np.mean(all_ssim)
    )

    # ========================================================
    # CONTENT ONLY
    # ========================================================

    content_predictions = predictions[
        content_mask
    ]

    content_hr = hr[
        content_mask
    ]

    content_mse = float(
        np.mean(
            (
                content_predictions
                -
                content_hr
            ) ** 2
        )
    )

    content_psnr = peak_signal_noise_ratio(
        content_hr,
        content_predictions,
        data_range=1.0
    )

    content_ssim_values = []

    content_sam_values = []

    for prediction, target in zip(
        content_predictions,
        content_hr
    ):

        ssim = structural_similarity(
            target,
            prediction,
            channel_axis=2,
            data_range=1.0
        )

        content_ssim_values.append(
            ssim
        )

        sam = calculate_sam(
            prediction,
            target
        )

        if not np.isnan(sam):

            content_sam_values.append(
                sam
            )

    content_ssim = float(
        np.mean(
            content_ssim_values
        )
    )

    content_sam = float(
        np.mean(
            content_sam_values
        )
    )

    content_sam_degrees = (
        np.degrees(
            content_sam
        )
    )

    # ========================================================
    # EMPTY PATCH BEHAVIOR
    # ========================================================

    empty_predictions = predictions[
        empty_mask
    ]

    empty_hr = hr[
        empty_mask
    ]

    if empty_count > 0:

        empty_mse = float(
            np.mean(
                (
                    empty_predictions
                    -
                    empty_hr
                ) ** 2
            )
        )

        empty_max_abs = float(
            np.max(
                np.abs(
                    empty_predictions
                )
            )
        )

        empty_mean_abs = float(
            np.mean(
                np.abs(
                    empty_predictions
                )
            )
        )

    else:

        empty_mse = 0.0
        empty_max_abs = 0.0
        empty_mean_abs = 0.0

    # ========================================================
    # Print results
    # ========================================================

    print("\n")
    print("=" * 60)
    print("SWINIR v2 RESULTS — REGION 3")
    print("=" * 60)

    print("\nALL PIXELS")
    print("-" * 40)

    print(
        f"MSE  : {global_mse:.10f}"
    )

    print(
        f"PSNR : {global_psnr:.4f} dB"
    )

    print(
        f"SSIM : {global_ssim:.6f}"
    )

    print("\nCONTENT ONLY")
    print("-" * 40)

    print(
        f"MSE  : {content_mse:.10f}"
    )

    print(
        f"PSNR : {content_psnr:.4f} dB"
    )

    print(
        f"SSIM : {content_ssim:.6f}"
    )

    print(
        f"SAM  : {content_sam:.6f} radians"
    )

    print(
        f"SAM  : {content_sam_degrees:.4f} degrees"
    )

    print("\nEMPTY PATCH BEHAVIOR")
    print("-" * 40)

    print(
        f"Empty patches       : {empty_count}"
    )

    print(
        f"Empty MSE           : {empty_mse:.10f}"
    )

    print(
        f"Empty mean abs      : {empty_mean_abs:.10f}"
    )

    print(
        f"Empty max abs       : {empty_max_abs:.10f}"
    )

    print("\nDATASET")
    print("-" * 40)

    print(
        f"Total patches       : {len(hr)}"
    )

    print(
        f"Empty patches       : {empty_count}"
    )

    print(
        f"Content patches     : {content_count}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()