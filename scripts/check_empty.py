import numpy as np
from pathlib import Path

for region_name in ["region1", "region2", "region3"]:

    path = Path("dataset") / region_name / "hr_patches.npy"

    hr = np.load(path)

    # A patch is empty if every value in all 4 bands is zero
    empty = np.all(hr == 0, axis=(1, 2, 3))

    print("=" * 50)
    print(region_name)
    print("Total patches :", len(hr))
    print("Empty patches :", empty.sum())
    print("Content       :", (~empty).sum())
    print("Empty %       :", f"{empty.mean() * 100:.2f}%")