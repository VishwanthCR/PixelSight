import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# P1 dataset
region = Path("dataset") / "region1"

# Load paired LR and HR patches
lr = np.load(region / "lr_patches.npy")
hr = np.load(region / "hr_patches.npy")

# Select first paired sample
idx = 0

lr_patch = lr[idx]
hr_patch = hr[idx]

print("LR shape:", lr_patch.shape)
print("HR shape:", hr_patch.shape)

# Bands 0, 1, 2 -> RGB
lr_rgb = lr_patch[:, :, :3]
hr_rgb = hr_patch[:, :, :3]

# Bicubic 2× upscale
lr_image = Image.fromarray(
    np.clip(lr_rgb * 255, 0, 255).astype(np.uint8)
)

lr_upscaled = lr_image.resize(
    (64, 64),
    Image.Resampling.BICUBIC
)

lr_upscaled = np.asarray(lr_upscaled) / 255.0

# Display comparison
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(lr_rgb)
axes[0].set_title("LR 32×32")

axes[1].imshow(lr_upscaled)
axes[1].set_title("Bicubic 2×")

axes[2].imshow(hr_rgb)
axes[2].set_title("HR 64×64")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig("patch_comparison.png", dpi=200)

plt.show()