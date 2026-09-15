from pathlib import Path

import numpy as np


PATCH_ROOT = Path("dataset/plan2/patches")


def analyze_split(split):
    files = sorted(
        (PATCH_ROOT / split).glob("*.npz")
    )

    fractions = []

    for path in files:
        with np.load(path) as data:
            mask = data["mask"]
            fractions.append(float(mask.mean()))

    fractions = np.array(fractions)

    print(f"\n=== {split.upper()} ===")
    print(f"Number of patches: {len(fractions)}")

    if len(fractions) == 0:
        print("No patches found.")
        return

    print(f"Minimum valid:     {fractions.min() * 100:.2f}%")
    print(f"10th percentile:   {np.percentile(fractions, 10) * 100:.2f}%")
    print(f"25th percentile:   {np.percentile(fractions, 25) * 100:.2f}%")
    print(f"Median valid:      {np.median(fractions) * 100:.2f}%")
    print(f"75th percentile:   {np.percentile(fractions, 75) * 100:.2f}%")
    print(f"Maximum valid:     {fractions.max() * 100:.2f}%")

    print("\nPatch counts by quality:")

    for threshold in (0.50, 0.70, 0.80, 0.90, 0.95):
        count = np.sum(fractions >= threshold)
        percentage = count / len(fractions) * 100

        print(
            f"  >= {threshold * 100:.0f}% valid:"
            f" {count} ({percentage:.2f}%)"
        )


def main():
    for split in ("train", "validation", "test"):
        analyze_split(split)


if __name__ == "__main__":
    main()