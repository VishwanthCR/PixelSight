from pathlib import Path
import json

import numpy as np
import torch

from pixelsight.models.segmentation.unet import create_unet


# ============================================================
# PixelSight Plan 4
# Full Held-Out Test Set Evaluation
# ============================================================

CHECKPOINT = Path(
    "checkpoints/segmentation/unet_worldcover_aug_dice_best.pth"
)

TEST_IMAGES = Path(
    "dataset/segmentation/patches/test/images"
)

TEST_LABELS = Path(
    "dataset/segmentation/patches/test/labels"
)

OUTPUT_DIR = Path(
    "results/plan4/test_evaluation"
)

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

    return model


# ============================================================
# Metrics accumulator
# ============================================================

def update_confusion_matrix(
    confusion,
    prediction,
    target,
):

    valid = target != IGNORE_INDEX

    true = target[valid]
    pred = prediction[valid]

    for c in range(NUM_CLASSES):

        mask = true == c

        if np.any(mask):

            confusion[c] += np.bincount(
                pred[mask],
                minlength=NUM_CLASSES,
            )


# ============================================================
# Calculate final metrics
# ============================================================

def calculate_metrics(confusion):

    total = confusion.sum()

    correct = np.trace(confusion)

    pixel_accuracy = (
        correct / total
        if total > 0
        else np.nan
    )

    ious = []
    dices = []
    precisions = []
    recalls = []

    for c in range(NUM_CLASSES):

        tp = confusion[c, c]

        fp = (
            confusion[:, c].sum()
            - tp
        )

        fn = (
            confusion[c, :].sum()
            - tp
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
                2 * tp / denominator
            )
        else:
            dice = np.nan

        precision_denominator = (
            tp + fp
        )

        if precision_denominator > 0:
            precision = (
                tp / precision_denominator
            )
        else:
            precision = np.nan

        recall_denominator = (
            tp + fn
        )

        if recall_denominator > 0:
            recall = (
                tp / recall_denominator
            )
        else:
            recall = np.nan

        ious.append(float(iou))
        dices.append(float(dice))
        precisions.append(float(precision))
        recalls.append(float(recall))

    return {
        "pixel_accuracy": float(pixel_accuracy),
        "mean_iou": float(np.nanmean(ious)),
        "mean_dice": float(np.nanmean(dices)),
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
        "confusion_matrix": confusion.tolist(),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("PixelSight Plan 4")
    print("FULL HELD-OUT TEST SET EVALUATION")
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
    # Check files
    # --------------------------------------------------------

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    if not TEST_IMAGES.exists():
        raise FileNotFoundError(
            f"Test images not found: {TEST_IMAGES}"
        )

    if not TEST_LABELS.exists():
        raise FileNotFoundError(
            f"Test labels not found: {TEST_LABELS}"
        )

    # --------------------------------------------------------
    # Find test patches
    # --------------------------------------------------------

    image_files = sorted(
        TEST_IMAGES.glob("*.npy")
    )

    print()
    print(
        f"Test image patches: "
        f"{len(image_files):,}"
    )

    if len(image_files) == 0:
        raise RuntimeError(
            "No test image patches found."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = load_model(device)

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )

    total_valid_pixels = 0

    for index, image_path in enumerate(
        image_files,
        start=1,
    ):

        label_path = (
            TEST_LABELS
            / image_path.name
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label for "
                f"{image_path.name}"
            )

        image = np.load(
            image_path
        ).astype(np.float32)

        label = np.load(
            label_path
        ).astype(np.uint8)

        if image.shape != (
            4,
            128,
            128,
        ):
            raise ValueError(
                f"Unexpected image shape "
                f"{image_path.name}: "
                f"{image.shape}"
            )

        if label.shape != (
            128,
            128,
        ):
            raise ValueError(
                f"Unexpected label shape "
                f"{label_path.name}: "
                f"{label.shape}"
            )

        tensor = (
            torch.from_numpy(image)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        with torch.no_grad():

            logits = model(tensor)

            prediction = (
                torch.argmax(
                    logits,
                    dim=1,
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )

        valid_pixels = (
            label != IGNORE_INDEX
        )

        total_valid_pixels += int(
            valid_pixels.sum()
        )

        update_confusion_matrix(
            confusion,
            prediction,
            label,
        )

        if (
            index % 250 == 0
            or index == len(image_files)
        ):

            print(
                f"Processed "
                f"{index:,}/"
                f"{len(image_files):,} patches"
            )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        confusion
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FULL TEST SET RESULTS")
    print("=" * 70)

    print(
        f"Test patches    : "
        f"{len(image_files):,}"
    )

    print(
        f"Valid pixels    : "
        f"{total_valid_pixels:,}"
    )

    print(
        f"Pixel Accuracy  : "
        f"{metrics['pixel_accuracy']:.6f}"
    )

    print(
        f"Mean IoU        : "
        f"{metrics['mean_iou']:.6f}"
    )

    print(
        f"Mean Dice       : "
        f"{metrics['mean_dice']:.6f}"
    )

    print(
        f"Mean Precision  : "
        f"{metrics['mean_precision']:.6f}"
    )

    print(
        f"Mean Recall     : "
        f"{metrics['mean_recall']:.6f}"
    )

    print()
    print("=" * 70)
    print("PER-CLASS RESULTS")
    print("=" * 70)

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

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        "Rows = reference"
    )

    print(
        "Columns = prediction"
    )

    print()

    print(
        " " * 15
        + " ".join(
            f"{name[:8]:>10}"
            for name in CLASS_NAMES
        )
    )

    for i, name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{name:<15}"
            + " ".join(
                f"{value:>10,}"
                for value in confusion[i]
            )
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": 21,
        "test_patches": len(image_files),
        "valid_pixels": total_valid_pixels,
        "metrics": metrics,
        "note": (
            "Evaluation performed on the held-out "
            "test split using the existing 50-epoch "
            "U-Net checkpoint. No training data is "
            "used for metric calculation."
        ),
    }

    with open(
        OUTPUT_DIR / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    np.save(
        OUTPUT_DIR / "test_confusion_matrix.npy",
        confusion,
    )

    print()
    print("=" * 70)
    print("TEST EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Saved to:\n"
        f"{OUTPUT_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()