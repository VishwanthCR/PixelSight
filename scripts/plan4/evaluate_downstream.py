from pathlib import Path
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from pixelsight.models.segmentation.unet import create_unet


# ============================================================
# PixelSight Plan 4
# Downstream Segmentation Evaluation
# ============================================================

CHECKPOINT = Path(
    "checkpoints/segmentation/unet_worldcover_best.pth"
)

NATIVE_IMAGE = Path(
    "dataset/segmentation/patches/train/images/"
    "r00000_c00000.npy"
)

LABEL = Path(
    "dataset/segmentation/patches/train/labels/"
    "r00000_c00000.npy"
)

SR_IMAGE = Path(
    "results/plan3/mean_sr.npy"
)

UNCERTAINTY = Path(
    "results/plan3/uncertainty_map.npy"
)

OUTPUT_DIR = Path("results/plan4")

NUM_CLASSES = 7
IGNORE_INDEX = 255
BASE_CHANNELS = 32

CLASS_NAMES = [
    "Tree",
    "Shrubland",
    "Grassland",
    "Cropland",
    "Built-up",
    "Bare",
    "Water",
]


# ============================================================
# Load model
# ============================================================

def load_model(device):

    print("\n=== Loading trained U-Net ===")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model = create_unet(
        in_channels=4,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch')}"
    )

    print(
        f"Validation loss: "
        f"{checkpoint.get('validation_loss'):.6f}"
    )

    print("U-Net loaded successfully.")

    return model


# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict(model, image, device):

    tensor = torch.from_numpy(
        image
    ).float().unsqueeze(0).to(device)

    logits = model(tensor)

    prediction = torch.argmax(
        logits,
        dim=1,
    )

    return prediction.squeeze(0).cpu().numpy()


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    prediction,
    target,
    num_classes=NUM_CLASSES,
):

    valid = target != IGNORE_INDEX

    pred = prediction[valid]
    true = target[valid]

    pixel_accuracy = float(
        np.mean(pred == true)
    )

    ious = []
    dices = []
    precisions = []
    recalls = []

    confusion = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    for c in range(num_classes):

        true_c = true == c
        pred_c = pred == c

        tp = np.sum(
            true_c & pred_c
        )

        fp = np.sum(
            (~true_c) & pred_c
        )

        fn = np.sum(
            true_c & (~pred_c)
        )

        union = tp + fp + fn

        if union > 0:
            iou = tp / union
        else:
            iou = np.nan

        denominator = (
            2 * tp + fp + fn
        )

        if denominator > 0:
            dice = (
                2 * tp
                / denominator
            )
        else:
            dice = np.nan

        precision_denominator = (
            tp + fp
        )

        if precision_denominator > 0:
            precision = (
                tp
                / precision_denominator
            )
        else:
            precision = np.nan

        recall_denominator = (
            tp + fn
        )

        if recall_denominator > 0:
            recall = (
                tp
                / recall_denominator
            )
        else:
            recall = np.nan

        ious.append(float(iou))
        dices.append(float(dice))
        precisions.append(float(precision))
        recalls.append(float(recall))

        confusion[c, :] = np.bincount(
            pred[true == c],
            minlength=num_classes,
        )

    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": float(
            np.nanmean(ious)
        ),
        "mean_dice": float(
            np.nanmean(dices)
        ),
        "mean_precision": float(
            np.nanmean(precisions)
        ),
        "mean_recall": float(
            np.nanmean(recalls)
        ),
        "per_class_iou": ious,
        "per_class_dice": dices,
        "per_class_precision": precisions,
        "per_class_recall": recalls,
        "confusion_matrix": confusion,
    }


# ============================================================
# Print metrics
# ============================================================

def print_metrics(title, metrics):

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    print(
        f"Pixel Accuracy : "
        f"{metrics['pixel_accuracy']:.6f}"
    )

    print(
        f"Mean IoU       : "
        f"{metrics['mean_iou']:.6f}"
    )

    print(
        f"Mean Dice      : "
        f"{metrics['mean_dice']:.6f}"
    )

    print(
        f"Mean Precision : "
        f"{metrics['mean_precision']:.6f}"
    )

    print(
        f"Mean Recall    : "
        f"{metrics['mean_recall']:.6f}"
    )

    print("\nPer-class metrics:")

    print(
        f"{'Class':<14}"
        f"{'IoU':>10}"
        f"{'Dice':>10}"
        f"{'Prec':>10}"
        f"{'Recall':>10}"
    )

    for i, name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{name:<14}"
            f"{metrics['per_class_iou'][i]:>10.4f}"
            f"{metrics['per_class_dice'][i]:>10.4f}"
            f"{metrics['per_class_precision'][i]:>10.4f}"
            f"{metrics['per_class_recall'][i]:>10.4f}"
        )


# ============================================================
# Save confusion matrix
# ============================================================

def save_confusion_matrix(
    matrix,
    path,
    title,
):

    plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(matrix)

    plt.colorbar(
        label="Pixel count"
    )

    plt.xticks(
        range(NUM_CLASSES),
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(NUM_CLASSES),
        CLASS_NAMES,
    )

    plt.xlabel(
        "Predicted class"
    )

    plt.ylabel(
        "Reference class"
    )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Save prediction visualization
# ============================================================

def save_prediction_map(
    prediction,
    path,
    title,
):

    plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        prediction,
        vmin=0,
        vmax=NUM_CLASSES - 1,
    )

    plt.colorbar(
        ticks=range(NUM_CLASSES),
        label="Class",
    )

    plt.title(title)
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("PixelSight Plan 4")
    print("Downstream Segmentation Evaluation")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Verify files
    # --------------------------------------------------------

    required = [
        CHECKPOINT,
        NATIVE_IMAGE,
        LABEL,
        SR_IMAGE,
        UNCERTAINTY,
    ]

    print("\nChecking required files...")

    for path in required:

        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}"
            )

        print(
            f"Found: {path}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    native = np.load(
        NATIVE_IMAGE
    ).astype(np.float32)

    label = np.load(
        LABEL
    ).astype(np.uint8)

    sr = np.load(
        SR_IMAGE
    ).astype(np.float32)

    uncertainty = np.load(
        UNCERTAINTY
    ).astype(np.float32)

    print("\nData shapes:")

    print(
        "Native:",
        native.shape,
    )

    print(
        "Label:",
        label.shape,
    )

    print(
        "SR:",
        sr.shape,
    )

    print(
        "Uncertainty:",
        uncertainty.shape,
    )

    # --------------------------------------------------------
    # Validate native image
    # --------------------------------------------------------

    if native.shape != (
        4,
        128,
        128,
    ):
        raise ValueError(
            f"Unexpected native shape: "
            f"{native.shape}"
        )

    # --------------------------------------------------------
    # Validate SR
    # --------------------------------------------------------

    if sr.shape != (
        512,
        512,
        4,
    ):
        raise ValueError(
            f"Unexpected SR shape: "
            f"{sr.shape}"
        )

    # --------------------------------------------------------
    # Validate uncertainty
    # --------------------------------------------------------

    if uncertainty.shape != (
        512,
        512,
    ):
        raise ValueError(
            f"Unexpected uncertainty shape: "
            f"{uncertainty.shape}"
        )

    # --------------------------------------------------------
    # Convert SR HWC -> CHW
    # --------------------------------------------------------

    sr_chw = sr.transpose(
        2,
        0,
        1,
    )

    # --------------------------------------------------------
    # Upsample 10m WorldCover reference
    # --------------------------------------------------------
    #
    # Each 10m reference pixel becomes a 4x4 block.
    #
    # This is a PROXY reference, NOT genuine 2.5m ground truth.
    # --------------------------------------------------------

    label_sr = np.repeat(
        np.repeat(
            label,
            4,
            axis=0,
        ),
        4,
        axis=1,
    )

    print(
        "\nProxy SR reference:",
        label_sr.shape,
    )

    # --------------------------------------------------------
    # Load U-Net
    # --------------------------------------------------------

    model = load_model(device)

    # --------------------------------------------------------
    # Native prediction
    # --------------------------------------------------------

    print(
        "\n=== Native 10m prediction ==="
    )

    native_prediction = predict(
        model,
        native,
        device,
    )

    native_metrics = calculate_metrics(
        native_prediction,
        label,
    )

    print_metrics(
        "NATIVE 10m RESULTS",
        native_metrics,
    )

    # --------------------------------------------------------
    # SR prediction
    # --------------------------------------------------------

    print(
        "\n=== LDSR-S2 2.5m prediction ==="
    )

    sr_prediction = predict(
        model,
        sr_chw,
        device,
    )

    sr_metrics = calculate_metrics(
        sr_prediction,
        label_sr,
    )

    print_metrics(
        "LDSR-S2 2.5m RESULTS",
        sr_metrics,
    )

    # --------------------------------------------------------
    # Uncertainty analysis
    # --------------------------------------------------------

    valid = label_sr != IGNORE_INDEX

    error_map = (
        sr_prediction != label_sr
    )

    error_map &= valid

    print(
        "\n=== Uncertainty analysis ==="
    )

    valid_uncertainty = uncertainty[
        valid
    ]

    valid_errors = error_map[
        valid
    ]

    print(
        f"Mean uncertainty: "
        f"{valid_uncertainty.mean():.8f}"
    )

    print(
        f"Median uncertainty: "
        f"{np.median(valid_uncertainty):.8f}"
    )

    print(
        f"Maximum uncertainty: "
        f"{valid_uncertainty.max():.8f}"
    )

    error_rate = float(
        valid_errors.mean()
    )

    print(
        f"Overall SR error rate: "
        f"{error_rate:.6f}"
    )

    # --------------------------------------------------------
    # High uncertainty threshold
    # --------------------------------------------------------

    threshold = np.percentile(
        valid_uncertainty,
        90,
    )

    high_uncertainty = (
        uncertainty >= threshold
    )

    high_uncertainty &= valid

    low_uncertainty = (
        ~high_uncertainty
    ) & valid

    high_error_rate = float(
        error_map[
            high_uncertainty
        ].mean()
    )

    low_error_rate = float(
        error_map[
            low_uncertainty
        ].mean()
    )

    print(
        f"\n90th percentile uncertainty: "
        f"{threshold:.8f}"
    )

    print(
        f"High-uncertainty pixels: "
        f"{int(high_uncertainty.sum())}"
    )

    print(
        f"High-uncertainty error rate: "
        f"{high_error_rate:.6f}"
    )

    print(
        f"Low-uncertainty error rate: "
        f"{low_error_rate:.6f}"
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    def clean_metrics(metrics):

        result = dict(metrics)

        result.pop(
            "confusion_matrix",
            None,
        )

        return result

    comparison = {
        "native_10m": clean_metrics(
            native_metrics
        ),
        "ldsr_s2_2p5m_proxy": clean_metrics(
            sr_metrics
        ),
        "uncertainty_analysis": {
            "mean_uncertainty":
                float(valid_uncertainty.mean()),
            "median_uncertainty":
                float(np.median(valid_uncertainty)),
            "max_uncertainty":
                float(valid_uncertainty.max()),
            "overall_error_rate":
                error_rate,
            "p90_uncertainty":
                float(threshold),
            "high_uncertainty_error_rate":
                high_error_rate,
            "low_uncertainty_error_rate":
                low_error_rate,
        },
        "note": (
            "The LDSR-S2 2.5m evaluation uses "
            "10m WorldCover labels replicated "
            "4x4 as a proxy reference. This is "
            "not genuine 2.5m ground truth."
        ),
    }

    metrics_path = (
        OUTPUT_DIR
        / "downstream_comparison.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            comparison,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR
        / "native_prediction.npy",
        native_prediction,
    )

    np.save(
        OUTPUT_DIR
        / "sr_prediction.npy",
        sr_prediction,
    )

    np.save(
        OUTPUT_DIR
        / "sr_proxy_reference.npy",
        label_sr,
    )

    np.save(
        OUTPUT_DIR
        / "error_map.npy",
        error_map,
    )

    # --------------------------------------------------------
    # Visual outputs
    # --------------------------------------------------------

    save_prediction_map(
        native_prediction,
        OUTPUT_DIR
        / "native_prediction.png",
        "Native 10m U-Net Prediction",
    )

    save_prediction_map(
        sr_prediction,
        OUTPUT_DIR
        / "sr_prediction.png",
        "LDSR-S2 2.5m U-Net Prediction",
    )

    save_prediction_map(
        label_sr,
        OUTPUT_DIR
        / "sr_proxy_reference.png",
        "Upsampled WorldCover Proxy Reference",
    )

    save_prediction_map(
        error_map.astype(np.uint8),
        OUTPUT_DIR
        / "sr_error_map.png",
        "LDSR-S2 Segmentation Error Map",
    )

    save_confusion_matrix(
        native_metrics[
            "confusion_matrix"
        ],
        OUTPUT_DIR
        / "native_confusion_matrix.png",
        "Native 10m Confusion Matrix",
    )

    save_confusion_matrix(
        sr_metrics[
            "confusion_matrix"
        ],
        OUTPUT_DIR
        / "sr_confusion_matrix.png",
        "LDSR-S2 2.5m Confusion Matrix",
    )

    # --------------------------------------------------------
    # Uncertainty/error visualization
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        uncertainty
    )

    plt.colorbar(
        label="Uncertainty"
    )

    plt.title(
        "P3 LDSR-S2 Uncertainty"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "uncertainty_map.png",
        dpi=200,
    )

    plt.close()

    plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        error_map
    )

    plt.title(
        "Segmentation Errors"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "uncertainty_vs_error.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PLAN 4 COMPLETE")
    print("=" * 70)

    print(
        f"Native Mean IoU : "
        f"{native_metrics['mean_iou']:.6f}"
    )

    print(
        f"SR Mean IoU     : "
        f"{sr_metrics['mean_iou']:.6f}"
    )

    print(
        f"Native Mean Dice: "
        f"{native_metrics['mean_dice']:.6f}"
    )

    print(
        f"SR Mean Dice    : "
        f"{sr_metrics['mean_dice']:.6f}"
    )

    print(
        f"High-U error    : "
        f"{high_error_rate:.6f}"
    )

    print(
        f"Low-U error     : "
        f"{low_error_rate:.6f}"
    )

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()