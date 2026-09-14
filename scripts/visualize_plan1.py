import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


# ============================================================
# Paths
# ============================================================

LR_PATH = ROOT / "dataset" / "region3" / "lr_patches.npy"
HR_PATH = ROOT / "dataset" / "region3" / "hr_patches.npy"

V2_MODEL_PATH = ROOT / "checkpoints" / "swinir_v2_best.pth"

FIGURES_DIR = ROOT / "results" / "plan1" / "figures"
QUALITATIVE_DIR = ROOT / "results" / "plan1" / "qualitative"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
QUALITATIVE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load v2 model
# ============================================================

import torch
from pixelsight.models.swinir_v2 import SwinIRResidual


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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
    V2_MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# Load Region 3
# ============================================================

print("Loading Region 3...")

lr = np.load(LR_PATH).astype(np.float32)
hr = np.load(HR_PATH).astype(np.float32)

print("LR:", lr.shape)
print("HR:", hr.shape)


# ============================================================
# Bicubic
# ============================================================

print("Generating bicubic predictions...")

bicubic = np.empty_like(hr)

for i in range(len(lr)):
    bicubic[i] = np.clip(
        zoom(
            lr[i],
            zoom=(2, 2, 1),
            order=3
        ),
        0.0,
        1.0
    )


# ============================================================
# SwinIR v2 predictions
# ============================================================

print("Generating SwinIR v2 predictions...")

v2_predictions = []

with torch.no_grad():

    for i in range(len(lr)):

        lr_patch = (
            torch.from_numpy(lr[i])
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(DEVICE)
        )

        prediction = model(lr_patch)

        prediction = (
            prediction
            .squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        prediction = np.clip(
            prediction,
            0.0,
            1.0
        )

        v2_predictions.append(
            prediction.astype(np.float32)
        )

v2_predictions = np.stack(v2_predictions)


# ============================================================
# Visualization helper
# ============================================================

def rgb_image(x):
    """
    Convert 4-channel B/G/R/NIR style data into
    a visible RGB composite using channels 0,1,2.
    """
    return np.clip(x[:, :, :3], 0, 1)


def save_image(
    image,
    title,
    filename
):

    plt.figure(figsize=(6, 6))

    plt.imshow(image)

    plt.title(title)

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Find a representative content patch
# ============================================================

content_mask = np.any(
    np.abs(hr) > 1e-8,
    axis=(1, 2, 3)
)

content_indices = np.where(content_mask)[0]

# Choose a middle content patch for reproducibility
content_index = int(
    content_indices[len(content_indices) // 2]
)

print(
    "Representative content patch:",
    content_index
)


# ============================================================
# Find an empty patch
# ============================================================

empty_mask = np.all(
    np.abs(hr) < 1e-8,
    axis=(1, 2, 3)
)

empty_indices = np.where(empty_mask)[0]

empty_index = int(
    empty_indices[len(empty_indices) // 2]
)

print(
    "Representative empty patch:",
    empty_index
)


# ============================================================
# Qualitative comparison
# ============================================================

idx = content_index

fig, axes = plt.subplots(
    1,
    4,
    figsize=(16, 4)
)

axes[0].imshow(rgb_image(lr[idx]))
axes[0].set_title("LR Input")

axes[1].imshow(rgb_image(bicubic[idx]))
axes[1].set_title("Bicubic ×2")

axes[2].imshow(rgb_image(v2_predictions[idx]))
axes[2].set_title("SwinIR v2")

axes[3].imshow(rgb_image(hr[idx]))
axes[3].set_title("HR Ground Truth")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    QUALITATIVE_DIR / "bicubic_vs_v2.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Empty patch comparison
# ============================================================

idx = empty_index

fig, axes = plt.subplots(
    1,
    3,
    figsize=(12, 4)
)

axes[0].imshow(rgb_image(lr[idx]))
axes[0].set_title("LR Empty Patch")

axes[1].imshow(rgb_image(v2_predictions[idx]))
axes[1].set_title("SwinIR v2")

axes[2].imshow(rgb_image(hr[idx]))
axes[2].set_title("HR Ground Truth")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    QUALITATIVE_DIR / "empty_patch_v2.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print()
print("Visualization complete.")
print()
print("Created:")
print(
    QUALITATIVE_DIR / "bicubic_vs_v2.png"
)
print(
    QUALITATIVE_DIR / "empty_patch_v2.png"
)