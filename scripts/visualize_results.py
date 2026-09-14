import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom

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

OUTPUT_DIR = Path("visual_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# Choose representative Region 3 content patches.
# These are non-empty patches.
PATCH_INDICES = [0, 1, 5, 10, 20]


# ============================================================
# Load data
# ============================================================

lr_data = np.load(LR_PATH)
hr_data = np.load(HR_PATH)


# ============================================================
# Load SwinIR
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

print("Loaded SwinIR checkpoint.")
print("Epoch:", checkpoint["epoch"])
print("Validation L1:", checkpoint["val_loss"])


# ============================================================
# RGB visualization helper
# ============================================================

def make_rgb(image):
    """
    Assumes:
        Band 0 = Red
        Band 1 = Green
        Band 2 = Blue

    Converts multispectral image into displayable RGB.
    """

    rgb = image[:, :, :3].copy()

    # Robust percentile stretch
    for c in range(3):

        low = np.percentile(rgb[:, :, c], 2)
        high = np.percentile(rgb[:, :, c], 98)

        if high > low:
            rgb[:, :, c] = (
                rgb[:, :, c] - low
            ) / (high - low)

    return np.clip(rgb, 0, 1)


# ============================================================
# Generate visual comparisons
# ============================================================

print("\nGenerating visual comparisons...")

with torch.no_grad():

    for index in PATCH_INDICES:

        lr = lr_data[index]
        hr = hr_data[index]

        # ----------------------------------------------------
        # Bicubic
        # ----------------------------------------------------

        bicubic = zoom(
            lr,
            zoom=(2, 2, 1),
            order=3
        ).astype(np.float32)

        bicubic = np.clip(
            bicubic,
            0,
            1
        )

        # ----------------------------------------------------
        # SwinIR
        # ----------------------------------------------------

        tensor = (
            torch.from_numpy(lr)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(DEVICE)
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):

            prediction = model(tensor)

        swinir = (
            prediction
            .squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        swinir = np.clip(
            swinir,
            0,
            1
        )

        # ----------------------------------------------------
        # RGB versions
        # ----------------------------------------------------

        lr_rgb = make_rgb(lr)
        bicubic_rgb = make_rgb(bicubic)
        swinir_rgb = make_rgb(swinir)
        hr_rgb = make_rgb(hr)

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(16, 4)
        )

        axes[0].imshow(lr_rgb)
        axes[0].set_title("LR 32×32")
        axes[0].axis("off")

        axes[1].imshow(bicubic_rgb)
        axes[1].set_title("Bicubic 64×64")
        axes[1].axis("off")

        axes[2].imshow(swinir_rgb)
        axes[2].set_title("SwinIR 64×64")
        axes[2].axis("off")

        axes[3].imshow(hr_rgb)
        axes[3].set_title("HR 64×64")
        axes[3].axis("off")

        fig.suptitle(
            f"PixelSight 2× Super-Resolution — Region 3 Patch {index}",
            fontsize=14
        )

        plt.tight_layout()

        output_path = (
            OUTPUT_DIR /
            f"patch_{index}_comparison.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print("Saved:", output_path)


print("\nDone.")
print("Visual results folder:", OUTPUT_DIR)