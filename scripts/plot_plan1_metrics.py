from pathlib import Path
import csv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT / "results" / "plan1" / "tables" / "plan1_comparison.csv"
OUT_DIR = ROOT / "results" / "plan1" / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

with open(CSV_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

content = [
    r for r in rows
    if r["subset"] == "Content Only"
]

models = [r["model"] for r in content]

psnr = [float(r["PSNR_dB"]) for r in content]
ssim = [float(r["SSIM"]) for r in content]
sam = [float(r["SAM_deg"]) for r in content]


def make_plot(values, ylabel, title, filename):
    plt.figure(figsize=(8, 5))

    bars = plt.bar(models, values)

    plt.ylabel(ylabel)
    plt.title(title)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


make_plot(
    psnr,
    "PSNR (dB)",
    "Plan 1 — Content-Only PSNR",
    "psnr_comparison.png"
)

make_plot(
    ssim,
    "SSIM",
    "Plan 1 — Content-Only SSIM",
    "ssim_comparison.png"
)

make_plot(
    sam,
    "SAM (degrees)",
    "Plan 1 — Content-Only Spectral Angle",
    "sam_comparison.png"
)

print("Created:")
print(OUT_DIR / "psnr_comparison.png")
print(OUT_DIR / "ssim_comparison.png")
print(OUT_DIR / "sam_comparison.png")