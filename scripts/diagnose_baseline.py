import numpy as np
from pathlib import Path
from scipy.ndimage import zoom


REGION = "region3"

dataset_dir = Path("dataset") / REGION

lr = np.load(dataset_dir / "lr_patches.npy")
hr = np.load(dataset_dir / "hr_patches.npy")

print("LR:", lr.shape)
print("HR:", hr.shape)

mse_values = []
hr_variance = []
lr_variance = []
hr_nonzero = []
lr_nonzero = []

for i in range(len(lr)):

    prediction = zoom(
        lr[i],
        zoom=(2, 2, 1),
        order=3
    ).astype(np.float32)

    prediction = np.clip(prediction, 0, 1)

    error = hr[i] - prediction

    mse = np.mean(error ** 2)

    mse_values.append(mse)

    hr_variance.append(np.var(hr[i]))
    lr_variance.append(np.var(lr[i]))

    hr_nonzero.append(np.count_nonzero(hr[i]))
    lr_nonzero.append(np.count_nonzero(lr[i]))


mse_values = np.array(mse_values)
hr_variance = np.array(hr_variance)
lr_variance = np.array(lr_variance)
hr_nonzero = np.array(hr_nonzero)
lr_nonzero = np.array(lr_nonzero)


# ============================================================
# BASIC STATISTICS
# ============================================================

zero_mse = mse_values == 0
nonzero_mse = mse_values > 0

print("\n" + "=" * 60)
print("BASELINE DIAGNOSTIC")
print("=" * 60)

print(f"Total patches       : {len(lr)}")
print(f"Zero-MSE patches    : {zero_mse.sum()}")
print(f"Non-zero MSE        : {nonzero_mse.sum()}")

print()
print(f"Zero-MSE percentage : {zero_mse.mean() * 100:.2f}%")

print("\nHR variance:")
print("Min    :", hr_variance.min())
print("Max    :", hr_variance.max())
print("Mean   :", hr_variance.mean())

print("\nLR variance:")
print("Min    :", lr_variance.min())
print("Max    :", lr_variance.max())
print("Mean   :", lr_variance.mean())


# ============================================================
# CHECK FIRST ZERO-MSE PATCH
# ============================================================

zero_indices = np.where(zero_mse)[0]

if len(zero_indices) > 0:

    idx = zero_indices[0]

    print("\n" + "-" * 60)
    print("FIRST ZERO-MSE PATCH")
    print("-" * 60)

    print("Index:", idx)

    print("HR min:", hr[idx].min())
    print("HR max:", hr[idx].max())
    print("HR mean:", hr[idx].mean())
    print("HR variance:", hr_variance[idx])

    print("LR min:", lr[idx].min())
    print("LR max:", lr[idx].max())
    print("LR mean:", lr[idx].mean())
    print("LR variance:", lr_variance[idx])

    print("HR non-zero pixels:", hr_nonzero[idx])
    print("LR non-zero pixels:", lr_nonzero[idx])


# ============================================================
# CHECK FIRST NON-ZERO PATCH
# ============================================================

nonzero_indices = np.where(nonzero_mse)[0]

if len(nonzero_indices) > 0:

    idx = nonzero_indices[0]

    print("\n" + "-" * 60)
    print("FIRST NON-ZERO-MSE PATCH")
    print("-" * 60)

    print("Index:", idx)

    print("MSE:", mse_values[idx])

    print("HR min:", hr[idx].min())
    print("HR max:", hr[idx].max())
    print("HR mean:", hr[idx].mean())
    print("HR variance:", hr_variance[idx])

    print("LR min:", lr[idx].min())
    print("LR max:", lr[idx].max())
    print("LR mean:", lr[idx].mean())
    print("LR variance:", lr_variance[idx])


# ============================================================
# GLOBAL MSE / PSNR
# ============================================================

global_mse = np.mean(
    (hr - np.array([
        np.clip(
            zoom(x, zoom=(2, 2, 1), order=3),
            0,
            1
        )
        for x in lr
    ])) ** 2
)

global_psnr = 10 * np.log10(1.0 / global_mse)

print("\n" + "=" * 60)
print("GLOBAL METRICS")
print("=" * 60)

print(f"Global MSE  : {global_mse:.10f}")
print(f"Global PSNR : {global_psnr:.4f} dB")

print("=" * 60)