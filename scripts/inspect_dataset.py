import numpy as np
from pathlib import Path

dataset_dir = Path("dataset")

for region in sorted(dataset_dir.iterdir()):
    if not region.is_dir():
        continue

    lr_path = region / "lr_patches.npy"
    hr_path = region / "hr_patches.npy"

    print(f"\n{'=' * 50}")
    print(f"Region: {region.name}")
    print(f"{'=' * 50}")

    if not lr_path.exists() or not hr_path.exists():
        print("Missing LR or HR file!")
        continue

    lr = np.load(lr_path)
    hr = np.load(hr_path)

    print("LR shape :", lr.shape)
    print("HR shape :", hr.shape)

    print("LR dtype :", lr.dtype)
    print("HR dtype :", hr.dtype)

    print("LR min   :", lr.min())
    print("LR max   :", lr.max())
    print("LR mean  :", lr.mean())

    print("HR min   :", hr.min())
    print("HR max   :", hr.max())
    print("HR mean  :", hr.mean())

    print("LR NaNs  :", np.isnan(lr).sum())
    print("HR NaNs  :", np.isnan(hr).sum())