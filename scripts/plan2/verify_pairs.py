from pathlib import Path

ROOT = Path("dataset/plan2/pairs")

for split in ("train", "validation", "test"):
    lr_count = len(list((ROOT / split / "lr").glob("*.npy")))
    hr_count = len(list((ROOT / split / "hr").glob("*.npy")))
    mask_count = len(list((ROOT / split / "mask").glob("*.npy")))

    print(
        f"{split.upper()}: "
        f"LR={lr_count}, "
        f"HR={hr_count}, "
        f"MASK={mask_count}"
    )

    if not (lr_count == hr_count == mask_count):
        raise RuntimeError(
            f"Count mismatch in {split}"
        )

print("\nAll pair counts match.")