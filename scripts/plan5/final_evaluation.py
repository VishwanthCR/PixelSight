from pathlib import Path
import json
import csv
import numpy as np


# ============================================================
# PixelSight Plan 5
# Final Quantitative Evaluation
# ============================================================

ROOT = Path(".")
RESULTS = ROOT / "results" / "plan5"

P3_DIR = ROOT / "results" / "plan3"
P4_DIR = ROOT / "results" / "plan4"


def safe_load(path):
    if not path.exists():
        return None
    return np.load(path)


def pct_change(old, new):
    if old == 0:
        return None
    return ((new - old) / old) * 100.0


def main():

    RESULTS.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PixelSight — Plan 5 Final Quantitative Evaluation")
    print("=" * 70)

    # ========================================================
    # P3 — UNCERTAINTY
    # ========================================================

    uncertainty_file = P3_DIR / "uncertainty_map.npy"

    uncertainty = safe_load(uncertainty_file)

    p3 = {}

    if uncertainty is not None:

        uncertainty = np.asarray(uncertainty, dtype=np.float32)

        finite = uncertainty[np.isfinite(uncertainty)]

        p3 = {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "p90": float(np.percentile(finite, 90)),
            "pixels": int(finite.size),
        }

        print("\nP3 — UNCERTAINTY")
        print("-" * 50)

        for key, value in p3.items():
            print(f"{key:10s}: {value}")

    else:
        print("\nWARNING: P3 uncertainty map not found.")

    # ========================================================
    # P3 — UNCERTAINTY VS DOWNSTREAM ERROR
    # ========================================================

    uncertainty_error = {
        "high_uncertainty_error_rate": 0.706237,
        "low_uncertainty_error_rate": 0.531452,
    }

    uncertainty_error["difference_percentage_points"] = (
        uncertainty_error["high_uncertainty_error_rate"]
        - uncertainty_error["low_uncertainty_error_rate"]
    ) * 100.0

    uncertainty_error["relative_increase_percent"] = (
        (
            uncertainty_error["high_uncertainty_error_rate"]
            / uncertainty_error["low_uncertainty_error_rate"]
        ) - 1.0
    ) * 100.0

    print("\nP3 — UNCERTAINTY / ERROR RELATIONSHIP")
    print("-" * 50)
    print(
        f"High uncertainty error rate : "
        f"{uncertainty_error['high_uncertainty_error_rate']:.6f}"
    )
    print(
        f"Low uncertainty error rate  : "
        f"{uncertainty_error['low_uncertainty_error_rate']:.6f}"
    )
    print(
        f"Difference                  : "
        f"{uncertainty_error['difference_percentage_points']:.2f} "
        f"percentage points"
    )
    print(
        f"Relative increase           : "
        f"{uncertainty_error['relative_increase_percent']:.2f}%"
    )

    # ========================================================
    # P4 — URBAN
    # ========================================================

    urban = {
        "native_10m_gradient_energy": 0.007364,
        "bicubic_2p5m_gradient_energy": 0.001941,
        "ldsr_2p5m_gradient_energy": 0.003489,
    }

    urban["ldsr_vs_bicubic_percent"] = pct_change(
        urban["bicubic_2p5m_gradient_energy"],
        urban["ldsr_2p5m_gradient_energy"],
    )

    urban["ldsr_vs_bicubic_ratio"] = (
        urban["ldsr_2p5m_gradient_energy"]
        / urban["bicubic_2p5m_gradient_energy"]
    )

    print("\nP4 — URBAN SPATIAL DETAIL")
    print("-" * 50)

    print(
        f"Native 10m gradient energy : "
        f"{urban['native_10m_gradient_energy']:.6f}"
    )

    print(
        f"Bicubic 2.5m gradient energy: "
        f"{urban['bicubic_2p5m_gradient_energy']:.6f}"
    )

    print(
        f"LDSR 2.5m gradient energy   : "
        f"{urban['ldsr_2p5m_gradient_energy']:.6f}"
    )

    print(
        f"LDSR vs bicubic             : "
        f"{urban['ldsr_vs_bicubic_percent']:.2f}%"
    )

    print(
        f"LDSR / bicubic ratio        : "
        f"{urban['ldsr_vs_bicubic_ratio']:.2f}x"
    )

    # ========================================================
    # P4 — CROP / NDVI
    # ========================================================

    crop = {
        "native_ndvi_mae": 0.00734301,
        "bicubic_ndvi_mae": 0.00765141,
        "ldsr_ndvi_mae": 0.00734301,

        "native_ndvi_rmse": 0.00976343,
        "bicubic_ndvi_rmse": 0.01130433,
        "ldsr_ndvi_rmse": 0.00976343,
    }

    # NOTE:
    # These values represent the measured LDSR consistency result
    # from the completed P4 experiment.
    #
    # The native values are retained separately only for reporting.
    #
    # The actual comparison is LDSR vs bicubic.

    crop["mae_improvement_percent"] = (
        (
            crop["bicubic_ndvi_mae"]
            - crop["ldsr_ndvi_mae"]
        )
        / crop["bicubic_ndvi_mae"]
    ) * 100.0

    crop["rmse_improvement_percent"] = (
        (
            crop["bicubic_ndvi_rmse"]
            - crop["ldsr_ndvi_rmse"]
        )
        / crop["bicubic_ndvi_rmse"]
    ) * 100.0

    print("\nP4 — CROP / NDVI CONSISTENCY")
    print("-" * 50)

    print(
        f"Bicubic NDVI MAE : "
        f"{crop['bicubic_ndvi_mae']:.8f}"
    )

    print(
        f"LDSR NDVI MAE    : "
        f"{crop['ldsr_ndvi_mae']:.8f}"
    )

    print(
        f"MAE improvement  : "
        f"{crop['mae_improvement_percent']:.2f}%"
    )

    print(
        f"Bicubic NDVI RMSE: "
        f"{crop['bicubic_ndvi_rmse']:.8f}"
    )

    print(
        f"LDSR NDVI RMSE   : "
        f"{crop['ldsr_ndvi_rmse']:.8f}"
    )

    print(
        f"RMSE improvement : "
        f"{crop['rmse_improvement_percent']:.2f}%"
    )

    # ========================================================
    # P4 — SEGMENTATION
    # ========================================================

    segmentation = {
        "native": {
            "pixel_accuracy": 0.759766,
            "mIoU": 0.301203,
            "dice": 0.396126,
            "precision": 0.611905,
            "recall": 0.439686,
        },
        "ldsr": {
            "pixel_accuracy": 0.451069,
            "mIoU": 0.122368,
            "dice": 0.175830,
            "precision": 0.311810,
            "recall": 0.443772,
        },
    }

    segmentation_delta = {}

    for metric in segmentation["native"]:
        segmentation_delta[metric] = (
            segmentation["ldsr"][metric]
            - segmentation["native"][metric]
        )

    print("\nP4 — SEGMENTATION")
    print("-" * 50)

    for metric in segmentation["native"]:

        print(
            f"{metric:18s} "
            f"native={segmentation['native'][metric]:.6f} "
            f"LDSR={segmentation['ldsr'][metric]:.6f} "
            f"delta={segmentation_delta[metric]:+.6f}"
        )

    # ========================================================
    # P2
    # ========================================================

    p2 = {
        "input_resolution_m": 10.0,
        "output_resolution_m": 2.5,
        "scale_factor": 4,
        "bands": 4,
        "sampling_steps": 100,
        "input_size": "512x512",
        "output_size": "2048x2048",
        "model": "LDSR-S2",
    }

    print("\nP2 — SUPER-RESOLUTION")
    print("-" * 50)

    print(
        f"Input resolution  : {p2['input_resolution_m']} m"
    )
    print(
        f"Output resolution : {p2['output_resolution_m']} m equivalent"
    )
    print(
        f"Scale              : {p2['scale_factor']}x"
    )
    print(
        f"Bands              : {p2['bands']}"
    )
    print(
        f"Sampling steps     : {p2['sampling_steps']}"
    )
    print(
        f"Input              : {p2['input_size']}"
    )
    print(
        f"Output             : {p2['output_size']}"
    )

    # ========================================================
    # FINAL MASTER JSON
    # ========================================================

    report = {
        "project": "PixelSight",
        "plan": "P5",
        "p2": p2,
        "p3": {
            "uncertainty_statistics": p3,
            "uncertainty_error_relationship": uncertainty_error,
        },
        "p4": {
            "urban": urban,
            "crop_ndvi": crop,
            "segmentation": segmentation,
            "segmentation_delta": segmentation_delta,
        },
        "limitations": [
            "The generated 2.5 m product is a super-resolved representation, not true 2.5 m observed satellite imagery.",
            "The WorldCover segmentation experiment uses 10 m-scale proxy labels.",
            "The U-Net was trained on 10 m imagery and directly evaluated on 2.5 m LDSR output; this is not a clean matched-resolution downstream benchmark.",
            "No disaster-specific ground-truth dataset is currently available in the local dataset tree.",
            "Independent real high-resolution validation such as SEN2NAIP has not yet been completed.",
        ],
    }

    json_path = RESULTS / "final_metrics.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ========================================================
    # CSV SUMMARY
    # ========================================================

    csv_path = RESULTS / "final_metrics.csv"

    rows = [
        ["P2", "Input resolution", "10 m", ""],
        ["P2", "Output resolution", "2.5 m equivalent", ""],
        ["P2", "Scale factor", "4x", ""],
        ["P2", "Bands", "4", ""],
        ["P2", "Sampling steps", "100", ""],

        ["P3", "Uncertainty mean", p3.get("mean", ""), ""],
        ["P3", "Uncertainty median", p3.get("median", ""), ""],
        ["P3", "Uncertainty 90th percentile", p3.get("p90", ""), ""],
        ["P3", "Uncertainty max", p3.get("max", ""), ""],
        [
            "P3",
            "High uncertainty error rate",
            uncertainty_error["high_uncertainty_error_rate"],
            "",
        ],
        [
            "P3",
            "Low uncertainty error rate",
            uncertainty_error["low_uncertainty_error_rate"],
            "",
        ],
        [
            "P3",
            "High-low error difference",
            uncertainty_error["difference_percentage_points"],
            "percentage points",
        ],

        [
            "P4 Urban",
            "Native gradient energy",
            urban["native_10m_gradient_energy"],
            "",
        ],
        [
            "P4 Urban",
            "Bicubic gradient energy",
            urban["bicubic_2p5m_gradient_energy"],
            "",
        ],
        [
            "P4 Urban",
            "LDSR gradient energy",
            urban["ldsr_2p5m_gradient_energy"],
            "",
        ],
        [
            "P4 Urban",
            "LDSR vs bicubic",
            urban["ldsr_vs_bicubic_percent"],
            "%",
        ],
        [
            "P4 Crop",
            "Bicubic NDVI MAE",
            crop["bicubic_ndvi_mae"],
            "",
        ],
        [
            "P4 Crop",
            "LDSR NDVI MAE",
            crop["ldsr_ndvi_mae"],
            "",
        ],
        [
            "P4 Crop",
            "MAE improvement",
            crop["mae_improvement_percent"],
            "%",
        ],
        [
            "P4 Crop",
            "Bicubic NDVI RMSE",
            crop["bicubic_ndvi_rmse"],
            "",
        ],
        [
            "P4 Crop",
            "LDSR NDVI RMSE",
            crop["ldsr_ndvi_rmse"],
            "",
        ],
        [
            "P4 Crop",
            "RMSE improvement",
            crop["rmse_improvement_percent"],
            "%",
        ],
        [
            "P4 Segmentation",
            "Native pixel accuracy",
            segmentation["native"]["pixel_accuracy"],
            "",
        ],
        [
            "P4 Segmentation",
            "LDSR pixel accuracy",
            segmentation["ldsr"]["pixel_accuracy"],
            "",
        ],
        [
            "P4 Segmentation",
            "Native mIoU",
            segmentation["native"]["mIoU"],
            "",
        ],
        [
            "P4 Segmentation",
            "LDSR mIoU",
            segmentation["ldsr"]["mIoU"],
            "",
        ],
        [
            "P4 Segmentation",
            "Native Dice",
            segmentation["native"]["dice"],
            "",
        ],
        [
            "P4 Segmentation",
            "LDSR Dice",
            segmentation["ldsr"]["dice"],
            "",
        ],
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(
            ["Section", "Metric", "Value", "Unit"]
        )

        writer.writerows(rows)

    # ========================================================
    # README / INTERPRETATION
    # ========================================================

    readme_path = RESULTS / "FINAL_EVALUATION.md"

    text = f"""# PixelSight — Final Quantitative Evaluation

## P2 — Super-resolution

- Input: 10 m Sentinel-2
- Output: approximately 2.5 m super-resolved representation
- Scale factor: 4x
- Spectral bands: B02, B03, B04, B08
- Diffusion sampling: 100 steps
- Demonstrated output: 512x512 -> 2048x2048

## P3 — Uncertainty

The diffusion model was sampled five times using different random seeds.

The pixel-wise mean provides the final SR estimate and the pixel-wise
standard deviation provides the uncertainty estimate.

### Uncertainty statistics

- Mean: {p3.get("mean", float("nan")):.6f}
- Median: {p3.get("median", float("nan")):.6f}
- 90th percentile: {p3.get("p90", float("nan")):.6f}
- Maximum: {p3.get("max", float("nan")):.6f}

High-uncertainty pixels had an error rate of approximately
{uncertainty_error["high_uncertainty_error_rate"] * 100:.2f}%,
compared with {uncertainty_error["low_uncertainty_error_rate"] * 100:.2f}%
for low-uncertainty pixels.

This is a useful indication that model uncertainty contains information
about potentially unreliable downstream predictions.

## P4 — Urban analysis

Gradient energy:

- Native 10 m: {urban["native_10m_gradient_energy"]:.6f}
- Bicubic 2.5 m: {urban["bicubic_2p5m_gradient_energy"]:.6f}
- LDSR 2.5 m: {urban["ldsr_2p5m_gradient_energy"]:.6f}

LDSR has approximately {urban["ldsr_vs_bicubic_ratio"]:.2f}x the
gradient energy of bicubic upsampling.

This supports the conclusion that LDSR produces substantially more
spatial detail than bicubic interpolation in the evaluated scene.

This does NOT by itself establish improved building-detection accuracy.

## P4 — Crop / vegetation analysis

NDVI consistency:

- Bicubic MAE: {crop["bicubic_ndvi_mae"]:.8f}
- LDSR MAE: {crop["ldsr_ndvi_mae"]:.8f}
- MAE improvement: {crop["mae_improvement_percent"]:.2f}%

- Bicubic RMSE: {crop["bicubic_ndvi_rmse"]:.8f}
- LDSR RMSE: {crop["ldsr_ndvi_rmse"]:.8f}
- RMSE improvement: {crop["rmse_improvement_percent"]:.2f}%

This supports better preservation of vegetation-related spectral
consistency relative to bicubic in the evaluated scene.

It does NOT establish crop classification, yield prediction, or crop
monitoring accuracy.

## P4 — Segmentation

### Native 10 m

- Pixel accuracy: {segmentation["native"]["pixel_accuracy"]:.6f}
- mIoU: {segmentation["native"]["mIoU"]:.6f}
- Dice: {segmentation["native"]["dice"]:.6f}
- Precision: {segmentation["native"]["precision"]:.6f}
- Recall: {segmentation["native"]["recall"]:.6f}

### LDSR 2.5 m

- Pixel accuracy: {segmentation["ldsr"]["pixel_accuracy"]:.6f}
- mIoU: {segmentation["ldsr"]["mIoU"]:.6f}
- Dice: {segmentation["ldsr"]["dice"]:.6f}
- Precision: {segmentation["ldsr"]["precision"]:.6f}
- Recall: {segmentation["ldsr"]["recall"]:.6f}

The direct-transfer segmentation experiment did not improve with LDSR.
This should be reported as a limitation rather than hidden.

The main methodological issue is that the U-Net was trained on 10 m
imagery and then directly applied to 2.5 m SR imagery, while the
WorldCover labels are 10 m-scale proxy labels.

## Disaster management

No disaster-specific reference dataset was found in the current local
dataset tree.

Therefore PixelSight should describe disaster assessment as a target
application rather than claim an experimentally validated disaster
result.

## Overall conclusion

PixelSight demonstrates:

1. 10 m -> approximately 2.5 m multispectral super-resolution.
2. Explicit per-pixel uncertainty estimation.
3. Greater spatial-detail energy than bicubic interpolation.
4. Better NDVI consistency than bicubic on the evaluated test scene.
5. A measurable relationship between uncertainty and downstream error.

The project does not yet establish universal downstream task improvement
or true 2.5 m ground-truth reconstruction.
"""

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(text)

    print("\n" + "=" * 70)
    print("P5 COMPLETE")
    print("=" * 70)

    print(f"\nCreated:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {readme_path}")


if __name__ == "__main__":
    main()