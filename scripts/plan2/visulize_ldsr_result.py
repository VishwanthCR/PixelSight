from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ================================================================
# Paths
# ================================================================

INPUT_TIF = Path(
    "dataset/plan2/geotiff/train_test_512_10m.tif"
)

SR_TIF = Path(
    "dataset/plan2/ldsr_s2_test/train_test_512_ldsr_s2_2p5m.tif"
)

OUTPUT_ROOT = Path(
    "results/plan2"
)

FIGURES_DIR = OUTPUT_ROOT / "figures"
QUALITATIVE_DIR = OUTPUT_ROOT / "qualitative"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
QUALITATIVE_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# Utility functions
# ================================================================

def normalize_for_display(image):
    """
    Percentile normalization for visualization only.
    Does NOT modify scientific data.
    """
    image = image.astype(np.float32)

    low = np.percentile(image, 2)
    high = np.percentile(image, 98)

    if high <= low:
        return np.zeros_like(image)

    image = (image - low) / (high - low)

    return np.clip(image, 0, 1)


def make_rgb(data):
    """
    Input:
        data shape = (4, H, W)

    Band order:
        B02, B03, B04, B08

    RGB:
        R = B04
        G = B03
        B = B02
    """

    blue = data[0]
    green = data[1]
    red = data[2]

    rgb = np.stack(
        [red, green, blue],
        axis=-1
    )

    return normalize_for_display(rgb)


def make_false_color(data):
    """
    False-color composite:

        R = B08 (NIR)
        G = B04 (Red)
        B = B03 (Green)
    """

    nir = data[3]
    red = data[2]
    green = data[1]

    false_color = np.stack(
        [nir, red, green],
        axis=-1
    )

    return normalize_for_display(false_color)


def resize_input_to_sr(input_data, output_shape):
    """
    Bicubic resize of the 10 m image to 2.5 m resolution
    purely for visualization comparison.

    This is NOT used as a scientific baseline here.
    """

    from scipy.ndimage import zoom

    scale_y = output_shape[0] / input_data.shape[1]
    scale_x = output_shape[1] / input_data.shape[2]

    resized = zoom(
        input_data,
        zoom=(1, scale_y, scale_x),
        order=3
    )

    return resized.astype(np.float32)


def save_figure(fig, path):
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {path}")


# ================================================================
# Load data
# ================================================================

print("=" * 70)
print("Loading Plan 2 LDSR-S2 results")
print("=" * 70)

with rasterio.open(INPUT_TIF) as src:

    input_data = src.read().astype(np.float32)

    input_transform = src.transform
    input_crs = src.crs
    input_bounds = src.bounds

    print()
    print("Input:")
    print(f"  Shape      : {input_data.shape}")
    print(f"  Resolution : {src.res}")
    print(f"  CRS        : {src.crs}")
    print(f"  Bounds     : {src.bounds}")
    print(f"  Dtype      : {src.dtypes}")


with rasterio.open(SR_TIF) as src:

    sr_data = src.read().astype(np.float32)

    sr_transform = src.transform
    sr_crs = src.crs
    sr_bounds = src.bounds

    print()
    print("LDSR-S2:")
    print(f"  Shape      : {sr_data.shape}")
    print(f"  Resolution : {src.res}")
    print(f"  CRS        : {src.crs}")
    print(f"  Bounds     : {src.bounds}")
    print(f"  Dtype      : {src.dtypes}")


# ================================================================
# Basic validation
# ================================================================

assert input_data.shape[0] == 4
assert sr_data.shape[0] == 4

assert sr_data.shape[1] == input_data.shape[1] * 4
assert sr_data.shape[2] == input_data.shape[2] * 4

print()
print("Resolution validation: PASSED")
print("4× spatial scaling: PASSED")

# ================================================================
# 1. RGB comparison
# ================================================================

print()
print("Creating RGB comparison...")

input_rgb = make_rgb(input_data)
sr_rgb = make_rgb(sr_data)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 7)
)

axes[0].imshow(input_rgb)
axes[0].set_title(
    "Sentinel-2 Input — 10 m",
    fontsize=14
)
axes[0].axis("off")

axes[1].imshow(sr_rgb)
axes[1].set_title(
    "LDSR-S2 Output — 2.5 m",
    fontsize=14
)
axes[1].axis("off")

fig.suptitle(
    "Plan 2: Sentinel-2 → LDSR-S2 4× Super-Resolution",
    fontsize=16
)

save_figure(
    fig,
    FIGURES_DIR / "input_vs_ldsr_rgb.png"
)


# ================================================================
# 2. False-color comparison
# ================================================================

print()
print("Creating false-color comparison...")

input_fc = make_false_color(input_data)
sr_fc = make_false_color(sr_data)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 7)
)

axes[0].imshow(input_fc)
axes[0].set_title(
    "Sentinel-2 False Color — 10 m",
    fontsize=14
)
axes[0].axis("off")

axes[1].imshow(sr_fc)
axes[1].set_title(
    "LDSR-S2 False Color — 2.5 m",
    fontsize=14
)
axes[1].axis("off")

fig.suptitle(
    "Plan 2: NIR False-Color Visualization",
    fontsize=16
)

save_figure(
    fig,
    FIGURES_DIR / "input_vs_ldsr_false_color.png"
)


# ================================================================
# 3. Individual band comparison
# ================================================================

print()
print("Creating individual band comparison...")

band_names = [
    "B02",
    "B03",
    "B04",
    "B08"
]

fig, axes = plt.subplots(
    4,
    2,
    figsize=(12, 20)
)

for i, band in enumerate(band_names):

    input_band = normalize_for_display(
        input_data[i]
    )

    sr_band = normalize_for_display(
        sr_data[i]
    )

    axes[i, 0].imshow(
        input_band,
        cmap="gray"
    )

    axes[i, 0].set_title(
        f"{band} — 10 m",
        fontsize=13
    )

    axes[i, 0].axis("off")

    axes[i, 1].imshow(
        sr_band,
        cmap="gray"
    )

    axes[i, 1].set_title(
        f"{band} — LDSR-S2 2.5 m",
        fontsize=13
    )

    axes[i, 1].axis("off")


fig.suptitle(
    "Plan 2: Individual Spectral Band Comparison",
    fontsize=16
)

save_figure(
    fig,
    FIGURES_DIR / "band_comparison.png"
)


# ================================================================
# 4. Spectral statistics
# ================================================================

print()
print("Calculating spectral statistics...")

input_means = []
sr_means = []

input_stds = []
sr_stds = []

input_mins = []
sr_mins = []

input_maxs = []
sr_maxs = []

for i in range(4):

    input_band = input_data[i]
    sr_band = sr_data[i]

    input_means.append(
        float(np.mean(input_band))
    )

    sr_means.append(
        float(np.mean(sr_band))
    )

    input_stds.append(
        float(np.std(input_band))
    )

    sr_stds.append(
        float(np.std(sr_band))
    )

    input_mins.append(
        float(np.min(input_band))
    )

    sr_mins.append(
        float(np.min(sr_band))
    )

    input_maxs.append(
        float(np.max(input_band))
    )

    sr_maxs.append(
        float(np.max(sr_band))
    )


# Mean comparison

x = np.arange(4)
width = 0.35

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.bar(
    x - width / 2,
    input_means,
    width,
    label="10 m Input"
)

ax.bar(
    x + width / 2,
    sr_means,
    width,
    label="LDSR-S2 2.5 m"
)

ax.set_xticks(x)
ax.set_xticklabels(band_names)

ax.set_ylabel(
    "Mean reflectance"
)

ax.set_title(
    "Mean Spectral Response: Input vs LDSR-S2"
)

ax.legend()

save_figure(
    fig,
    FIGURES_DIR / "spectral_mean_comparison.png"
)


# Standard deviation comparison

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.bar(
    x - width / 2,
    input_stds,
    width,
    label="10 m Input"
)

ax.bar(
    x + width / 2,
    sr_stds,
    width,
    label="LDSR-S2 2.5 m"
)

ax.set_xticks(x)
ax.set_xticklabels(band_names)

ax.set_ylabel(
    "Standard deviation"
)

ax.set_title(
    "Spectral Variation: Input vs LDSR-S2"
)

ax.legend()

save_figure(
    fig,
    FIGURES_DIR / "spectral_std_comparison.png"
)


# ================================================================
# 5. Zoomed RGB comparison
# ================================================================

print()
print("Creating zoomed RGB comparison...")

# Central quarter of the image
h, w = sr_data.shape[1:]

crop_size = min(
    1024,
    h // 2,
    w // 2
)

row_start = (h - crop_size) // 2
col_start = (w - crop_size) // 2

sr_crop = sr_data[
    :,
    row_start:row_start + crop_size,
    col_start:col_start + crop_size
]

# Corresponding input crop
input_crop_size = crop_size // 4

input_h, input_w = input_data.shape[1:]

input_row_start = (
    input_h - input_crop_size
) // 2

input_col_start = (
    input_w - input_crop_size
) // 2

input_crop = input_data[
    :,
    input_row_start:
    input_row_start + input_crop_size,
    input_col_start:
    input_col_start + input_crop_size
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 7)
)

axes[0].imshow(
    make_rgb(input_crop)
)

axes[0].set_title(
    "10 m Input — Zoomed",
    fontsize=14
)

axes[0].axis("off")

axes[1].imshow(
    make_rgb(sr_crop)
)

axes[1].set_title(
    "2.5 m LDSR-S2 — Zoomed",
    fontsize=14
)

axes[1].axis("off")

fig.suptitle(
    "Plan 2: Zoomed Spatial Detail",
    fontsize=16
)

save_figure(
    fig,
    QUALITATIVE_DIR / "rgb_comparison_zoom.png"
)


# ================================================================
# 6. Zoomed false-color
# ================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 7)
)

axes[0].imshow(
    make_false_color(input_crop)
)

axes[0].set_title(
    "10 m Input — False Color",
    fontsize=14
)

axes[0].axis("off")

axes[1].imshow(
    make_false_color(sr_crop)
)

axes[1].set_title(
    "2.5 m LDSR-S2 — False Color",
    fontsize=14
)

axes[1].axis("off")

fig.suptitle(
    "Plan 2: Zoomed NIR Spatial Detail",
    fontsize=16
)

save_figure(
    fig,
    QUALITATIVE_DIR / "false_color_zoom.png"
)


# ================================================================
# 7. Per-band zoom
# ================================================================

fig, axes = plt.subplots(
    4,
    2,
    figsize=(12, 20)
)

for i, band in enumerate(band_names):

    axes[i, 0].imshow(
        normalize_for_display(
            input_crop[i]
        ),
        cmap="gray"
    )

    axes[i, 0].set_title(
        f"{band} — 10 m",
        fontsize=13
    )

    axes[i, 0].axis("off")

    axes[i, 1].imshow(
        normalize_for_display(
            sr_crop[i]
        ),
        cmap="gray"
    )

    axes[i, 1].set_title(
        f"{band} — 2.5 m LDSR-S2",
        fontsize=13
    )

    axes[i, 1].axis("off")


fig.suptitle(
    "Plan 2: Zoomed Multispectral Detail",
    fontsize=16
)

save_figure(
    fig,
    QUALITATIVE_DIR / "band_detail_comparison.png"
)


# ================================================================
# 8. Spatial gradient/detail visualization
# ================================================================

print()
print("Creating spatial-detail visualization...")

# Use RGB luminance

sr_rgb_raw = np.stack(
    [
        sr_data[2],
        sr_data[1],
        sr_data[0]
    ],
    axis=-1
)

input_rgb_raw = np.stack(
    [
        input_data[2],
        input_data[1],
        input_data[0]
    ],
    axis=-1
)

sr_luma = (
    0.299 * sr_rgb_raw[:, :, 0]
    + 0.587 * sr_rgb_raw[:, :, 1]
    + 0.114 * sr_rgb_raw[:, :, 2]
)

input_luma = (
    0.299 * input_rgb_raw[:, :, 0]
    + 0.587 * input_rgb_raw[:, :, 1]
    + 0.114 * input_rgb_raw[:, :, 2]
)

from scipy.ndimage import sobel

sr_edge_x = sobel(
    sr_luma,
    axis=1
)

sr_edge_y = sobel(
    sr_luma,
    axis=0
)

sr_edges = np.sqrt(
    sr_edge_x ** 2
    + sr_edge_y ** 2
)

input_edge_x = sobel(
    input_luma,
    axis=1
)

input_edge_y = sobel(
    input_luma,
    axis=0
)

input_edges = np.sqrt(
    input_edge_x ** 2
    + input_edge_y ** 2
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 7)
)

axes[0].imshow(
    normalize_for_display(input_edges),
    cmap="gray"
)

axes[0].set_title(
    "10 m Input — Spatial Detail",
    fontsize=14
)

axes[0].axis("off")

axes[1].imshow(
    normalize_for_display(sr_edges),
    cmap="gray"
)

axes[1].set_title(
    "2.5 m LDSR-S2 — Spatial Detail",
    fontsize=14
)

axes[1].axis("off")

fig.suptitle(
    "Plan 2: Spatial Detail / Edge Visualization",
    fontsize=16
)

save_figure(
    fig,
    FIGURES_DIR / "spatial_detail_comparison.png"
)


# ================================================================
# 9. Save statistics CSV
# ================================================================

print()
print("Saving statistics...")

statistics_path = (
    OUTPUT_ROOT / "plan2_spectral_statistics.csv"
)

with open(
    statistics_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "band,"
        "input_min,input_max,input_mean,input_std,"
        "sr_min,sr_max,sr_mean,sr_std\n"
    )

    for i, band in enumerate(band_names):

        f.write(
            f"{band},"
            f"{input_mins[i]:.8f},"
            f"{input_maxs[i]:.8f},"
            f"{input_means[i]:.8f},"
            f"{input_stds[i]:.8f},"
            f"{sr_mins[i]:.8f},"
            f"{sr_maxs[i]:.8f},"
            f"{sr_means[i]:.8f},"
            f"{sr_stds[i]:.8f}\n"
        )

print(
    f"Saved: {statistics_path}"
)


# ================================================================
# Final summary
# ================================================================

print()
print("=" * 70)
print("PLAN 2 VISUALIZATION COMPLETE")
print("=" * 70)

print()
print("Figures:")
print(
    f"  {FIGURES_DIR}"
)

print()
print("Qualitative:")
print(
    f"  {QUALITATIVE_DIR}"
)

print()
print("Generated:")
print("  1. RGB comparison")
print("  2. False-color comparison")
print("  3. Individual band comparison")
print("  4. Spectral mean comparison")
print("  5. Spectral standard deviation comparison")
print("  6. Zoomed RGB comparison")
print("  7. Zoomed false-color comparison")
print("  8. Zoomed multispectral comparison")
print("  9. Spatial-detail / edge comparison")
print(" 10. Spectral statistics CSV")

print()
print("Done.")