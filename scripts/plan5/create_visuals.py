from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PixelSight — Plan 5 Visualization Pack
# ============================================================

ROOT = Path(".")
P5 = ROOT / "results" / "plan5"
P3 = ROOT / "results" / "plan3"
OUT = P5 / "figures"

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load P5 report
# ============================================================

with open(P5 / "final_metrics.json", "r", encoding="utf-8") as f:
    report = json.load(f)


# ============================================================
# P3 UNCERTAINTY
# ============================================================

uncertainty = np.load(
    P3 / "uncertainty_map.npy"
).astype(np.float32)

uncertainty = np.nan_to_num(
    uncertainty,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# ============================================================
# 1. UNCERTAINTY MAP
# ============================================================

plt.figure(figsize=(8, 7))

plt.imshow(
    uncertainty,
    cmap="inferno"
)

plt.colorbar(
    label="Pixel uncertainty"
)

plt.title(
    "PixelSight — Diffusion Uncertainty Map"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    OUT / "01_uncertainty_map.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. UNCERTAINTY DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    uncertainty.ravel(),
    bins=80
)

plt.axvline(
    np.percentile(uncertainty, 90),
    linestyle="--",
    label="90th percentile"
)

plt.xlabel(
    "Pixel uncertainty"
)

plt.ylabel(
    "Number of pixels"
)

plt.title(
    "Distribution of PixelSight Uncertainty"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT / "02_uncertainty_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# 3. URBAN SPATIAL DETAIL
# ============================================================

urban = report["p4"]["urban"]

labels = [
    "Native 10m",
    "Bicubic 2.5m",
    "LDSR 2.5m"
]

values = [
    urban["native_10m_gradient_energy"],
    urban["bicubic_2p5m_gradient_energy"],
    urban["ldsr_2p5m_gradient_energy"]
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    labels,
    values
)

plt.ylabel(
    "Gradient energy"
)

plt.title(
    "Spatial Detail Comparison"
)

for bar, value in zip(bars, values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.6f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    OUT / "03_urban_spatial_detail.png",
    dpi=200
)

plt.close()


# ============================================================
# 4. CROP NDVI
# ============================================================

crop = report["p4"]["crop_ndvi"]

labels = [
    "Bicubic",
    "LDSR"
]

mae = [
    crop["bicubic_ndvi_mae"],
    crop["ldsr_ndvi_mae"]
]

rmse = [
    crop["bicubic_ndvi_rmse"],
    crop["ldsr_ndvi_rmse"]
]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    mae,
    width,
    label="MAE"
)

plt.bar(
    x + width / 2,
    rmse,
    width,
    label="RMSE"
)

plt.xticks(
    x,
    labels
)

plt.ylabel(
    "NDVI consistency error"
)

plt.title(
    "Vegetation Spectral Consistency"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT / "04_crop_ndvi_consistency.png",
    dpi=200
)

plt.close()


# ============================================================
# 5. UNCERTAINTY VS ERROR
# ============================================================

ue = report[
    "p3"
]["uncertainty_error_relationship"]

labels = [
    "Low uncertainty",
    "High uncertainty"
]

values = [
    ue["low_uncertainty_error_rate"] * 100,
    ue["high_uncertainty_error_rate"] * 100
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    labels,
    values
)

plt.ylabel(
    "Downstream error rate (%)"
)

plt.title(
    "Uncertainty as a Reliability Signal"
)

for bar, value in zip(bars, values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    OUT / "05_uncertainty_vs_error.png",
    dpi=200
)

plt.close()


# ============================================================
# 6. SEGMENTATION
# ============================================================

seg = report[
    "p4"
]["segmentation"]

metrics = [
    "pixel_accuracy",
    "mIoU",
    "dice",
    "precision",
    "recall"
]

metric_names = [
    "Accuracy",
    "mIoU",
    "Dice",
    "Precision",
    "Recall"
]

native = [
    seg["native"][m]
    for m in metrics
]

ldsr = [
    seg["ldsr"][m]
    for m in metrics
]

x = np.arange(len(metrics))

plt.figure(figsize=(10, 5))

plt.bar(
    x - width / 2,
    native,
    width,
    label="Native 10m"
)

plt.bar(
    x + width / 2,
    ldsr,
    width,
    label="LDSR 2.5m"
)

plt.xticks(
    x,
    metric_names
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0,
    1
)

plt.title(
    "Segmentation: Native vs Direct LDSR Transfer"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT / "06_segmentation_comparison.png",
    dpi=200
)

plt.close()


# ============================================================
# 7. MASTER RESULTS TABLE
# ============================================================

table = OUT / "07_master_results.txt"

with open(table, "w", encoding="utf-8") as f:

    f.write("=" * 80 + "\n")
    f.write("PIXELSIGHT — MASTER RESULTS\n")
    f.write("=" * 80 + "\n\n")

    f.write("P2 — SUPER-RESOLUTION\n")
    f.write("-" * 80 + "\n")
    f.write("Input resolution       : 10 m\n")
    f.write("Output resolution      : 2.5 m equivalent\n")
    f.write("Scale factor           : 4x\n")
    f.write("Spectral bands         : 4 (B02/B03/B04/B08)\n")
    f.write("Diffusion steps        : 100\n\n")

    f.write("P3 — UNCERTAINTY\n")
    f.write("-" * 80 + "\n")
    f.write(
        f"Mean uncertainty       : "
        f"{report['p3']['uncertainty_statistics']['mean']:.6f}\n"
    )

    f.write(
        f"Median uncertainty     : "
        f"{report['p3']['uncertainty_statistics']['median']:.6f}\n"
    )

    f.write(
        f"90th percentile        : "
        f"{report['p3']['uncertainty_statistics']['p90']:.6f}\n"
    )

    f.write(
        f"Maximum uncertainty    : "
        f"{report['p3']['uncertainty_statistics']['max']:.6f}\n"
    )

    f.write(
        f"Low-U error rate       : "
        f"{ue['low_uncertainty_error_rate'] * 100:.2f}%\n"
    )

    f.write(
        f"High-U error rate      : "
        f"{ue['high_uncertainty_error_rate'] * 100:.2f}%\n"
    )

    f.write(
        f"Difference             : "
        f"{ue['difference_percentage_points']:.2f} percentage points\n"
    )

    f.write(
        f"Relative increase      : "
        f"{ue['relative_increase_percent']:.2f}%\n\n"
    )

    f.write("P4 — URBAN\n")
    f.write("-" * 80 + "\n")

    f.write(
        f"Native gradient energy : "
        f"{urban['native_10m_gradient_energy']:.6f}\n"
    )

    f.write(
        f"Bicubic gradient       : "
        f"{urban['bicubic_2p5m_gradient_energy']:.6f}\n"
    )

    f.write(
        f"LDSR gradient          : "
        f"{urban['ldsr_2p5m_gradient_energy']:.6f}\n"
    )

    f.write(
        f"LDSR vs bicubic        : "
        f"{urban['ldsr_vs_bicubic_percent']:.2f}%\n"
    )

    f.write(
        f"LDSR/bicubic ratio     : "
        f"{urban['ldsr_vs_bicubic_ratio']:.2f}x\n\n"
    )

    f.write("P4 — CROP\n")
    f.write("-" * 80 + "\n")

    f.write(
        f"Bicubic NDVI MAE      : "
        f"{crop['bicubic_ndvi_mae']:.8f}\n"
    )

    f.write(
        f"LDSR NDVI MAE         : "
        f"{crop['ldsr_ndvi_mae']:.8f}\n"
    )

    f.write(
        f"MAE improvement        : "
        f"{crop['mae_improvement_percent']:.2f}%\n"
    )

    f.write(
        f"Bicubic NDVI RMSE     : "
        f"{crop['bicubic_ndvi_rmse']:.8f}\n"
    )

    f.write(
        f"LDSR NDVI RMSE        : "
        f"{crop['ldsr_ndvi_rmse']:.8f}\n"
    )

    f.write(
        f"RMSE improvement      : "
        f"{crop['rmse_improvement_percent']:.2f}%\n\n"
    )

    f.write("P4 — SEGMENTATION\n")
    f.write("-" * 80 + "\n")

    for m, name in zip(metrics, metric_names):

        f.write(
            f"{name:<15}: "
            f"Native={seg['native'][m]:.4f} | "
            f"LDSR={seg['ldsr'][m]:.4f}\n"
        )

    f.write("\n")
    f.write("IMPORTANT LIMITATIONS\n")
    f.write("-" * 80 + "\n")
    f.write(
        "1. LDSR output is a super-resolved representation, not observed "
        "2.5 m satellite imagery.\n"
    )

    f.write(
        "2. WorldCover provides 10 m-scale proxy labels.\n"
    )

    f.write(
        "3. The U-Net was trained at 10 m and directly applied to 2.5 m LDSR output.\n"
    )

    f.write(
        "4. No disaster-specific ground-truth dataset was available locally.\n"
    )

    f.write(
        "5. Independent true high-resolution validation has not yet been completed.\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("PixelSight — P5 Visualization Pack COMPLETE")
print("=" * 70)

print()
print("Created:")

for path in sorted(OUT.iterdir()):
    print(f"  {path}")