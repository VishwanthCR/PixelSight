import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from pixelsight.models.swinir_v2 import SwinIRResidual


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


model = SwinIRResidual(
    in_channels=4,
    out_channels=4,
    scale=2,
    embed_dim=60,
    depths=6,
    num_heads=6,
    window_size=8
).to(device)

model.eval()

parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("Parameters:", parameters)


lr = np.load(
    ROOT / "dataset" / "region1" / "lr_patches.npy"
)

patch = torch.from_numpy(
    lr[0]
).permute(
    2, 0, 1
).unsqueeze(
    0
).float().to(device)


with torch.no_grad():
    output = model(patch)


print("Input shape:", patch.shape)
print("Output shape:", output.shape)
print("Output min:", output.min().item())
print("Output max:", output.max().item())

if torch.cuda.is_available():
    print(
        "GPU memory allocated:",
        round(
            torch.cuda.memory_allocated() / 1024**3,
            3
        ),
        "GB"
    )

    print(
        "GPU memory reserved :",
        round(
            torch.cuda.memory_reserved() / 1024**3,
            3
        ),
        "GB"
    )
