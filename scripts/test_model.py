import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import numpy as np

from pixelsight.models.swinir import SwinIR


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# Model
# ============================================================

model = SwinIR(
    in_channels=4,
    out_channels=4,
    scale=2,
    embed_dim=60,
    depths=6,
    num_heads=6,
    window_size=8
).to(device)


print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)


# ============================================================
# Load one real LR patch
# ============================================================

lr = np.load(
    "dataset/region1/lr_patches.npy"
)

patch = torch.from_numpy(
    lr[0]
).permute(
    2, 0, 1
).unsqueeze(
    0
).float()


patch = patch.to(device)


print(
    "Input shape:",
    patch.shape
)


# ============================================================
# Forward pass
# ============================================================

model.eval()

with torch.no_grad():

    output = model(patch)


print(
    "Output shape:",
    output.shape
)

print(
    "Output min:",
    output.min().item()
)

print(
    "Output max:",
    output.max().item()
)


# ============================================================
# VRAM
# ============================================================

if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    print(
        f"GPU memory allocated: {allocated:.3f} GB"
    )

    print(
        f"GPU memory reserved : {reserved:.3f} GB"
    )