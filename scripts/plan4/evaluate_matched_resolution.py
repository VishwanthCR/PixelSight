from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F

from pixelsight.models.segmentation.unet import create_unet


# ============================================================
# PixelSight Plan 4
# Matched-Resolution Downstream Evaluation
# ============================================================

CHECKPOINT = Path(
    "checkpoints/segmentation/unet_worldcover_best.pth"
)

NATIVE_IMAGE = Path(
    "dataset/segmentation/patches/train/images/r00000_c00000.npy"
)

LABEL = Path(
    "dataset/segmentation/patches/train/labels/r00000_c00000.npy"
)

LDSR_IMAGE = Path(
    "results/plan3/mean_sr.npy"
)

OUTPUT_DIR = Path("results/plan4/matched_resolution")

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
# Model
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
# Prediction
# ============================================================

@torch.no_grad()
def predict(model, image, device):

    tensor = torch.from_numpy(
        image
    ).float().unsqueeze(0).to(device)

    logits = model(tensor)

    return torch.argmax(
        logits,
        dim=1,
    ).squeeze(0).cpu().numpy()


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    prediction,
    target,
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
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )

    for c in range(NUM_CLASSES):

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

        iou = (
            tp / union
            if union > 0
            else np.nan
        )

        denominator = (
            2 * tp + fp + fn
        )

        dice = (
            2 * tp / denominator
            if denominator > 0
            else np.nan
        )

        precision_denominator = (
            tp + fp
        )

        precision = (
            tp / precision_denominator
            if precision_denominator > 0
            else np.nan
        )

        recall_denominator = (
            tp + fn
        )

        recall = (
            tp / recall_denominator
            if recall_denominator > 0
            else np.nan
        )

        ious.append(float(iou))
        dices.append(float(dice))
        precisions.append(float(precision))
        recalls.append(float(recall))

        confusion[c, :] = np.bincount(
            pred[true == c],
            minlength=NUM_CLASSES,
        )

    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": float(np.nanmean(ious)),
        "mean_dice": float(np.nanmean(dices)),
        "mean_precision": float(np.nanmean(precisions)),
        "mean_recall": float(np.nanmean(recalls)),
        "per_class_iou": ious,
        "per_class_dice": dices,
        "per_class_precision": precisions,
        "per_class_recall": recalls,
        "confusion_matrix": confusion.tolist(),
    }


# ============================================================
# Print
# ============================================================

def print_metrics(name, metrics):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

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

    print()
    print(
        f"{'Class':<14}"
        f"{'IoU':>10}"
        f"{'Dice':>10}"
        f"{'Prec':>10}"
        f"{'Recall':>10}"
    )

    for i, name in enumerate(CLASS_NAMES):

        print(
            f"{name:<14}"
            f"{metrics['per_class_iou'][i]:>10.4f}"
            f"{metrics['per_class_dice'][i]:>10.4f}"
            f"{metrics['per_class_precision'][i]:>10.4f}"
            f"{metrics['per_class_recall'][i]:>10.4f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("PixelSight Plan 4")
    print("Matched-Resolution Downstream Evaluation")
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

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    native = np.load(
        NATIVE_IMAGE
    ).astype(np.float32)

    label = np.load(
        LABEL
    ).astype(np.uint8)

    ldsr = np.load(
        LDSR_IMAGE
    ).astype(np.float32)

    print()
    print("Input shapes:")
    print("  Native :", native.shape)
    print("  Label  :", label.shape)
    print("  LDSR   :", ldsr.shape)

    if native.shape != (4, 128, 128):
        raise ValueError(
            f"Unexpected native shape: {native.shape}"
        )

    if label.shape != (128, 128):
        raise ValueError(
            f"Unexpected label shape: {label.shape}"
        )

    if ldsr.shape != (512, 512, 4):
        raise ValueError(
            f"Unexpected LDSR shape: {ldsr.shape}"
        )

    # --------------------------------------------------------
    # Convert LDSR HWC -> CHW
    # --------------------------------------------------------

    ldsr_chw = ldsr.transpose(
        2,
        0,
        1,
    )

    # --------------------------------------------------------
    # Bicubic baseline
    #
    # Native 128x128 -> 512x512
    # --------------------------------------------------------

    native_tensor = torch.from_numpy(
        native
    ).unsqueeze(0)

    bicubic_tensor = F.interpolate(
        native_tensor,
        size=(512, 512),
        mode="bicubic",
        align_corners=False,
    )

    bicubic = (
        bicubic_tensor
        .squeeze(0)
        .numpy()
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The existing U-Net was trained at 128x128.
    #
    # Therefore we downsample both 2.5m representations
    # back to 128x128 before feeding them into that model.
    #
    # This tests spectral/spatial information preservation
    # without introducing a 4x input-size mismatch.
    # --------------------------------------------------------

    ldsr_tensor = torch.from_numpy(
        ldsr_chw
    ).unsqueeze(0)

    bicubic_10m_tensor = F.interpolate(
        bicubic_tensor,
        size=(128, 128),
        mode="area",
    )

    ldsr_10m_tensor = F.interpolate(
        ldsr_tensor,
        size=(128, 128),
        mode="area",
    )

    bicubic_matched = (
        bicubic_10m_tensor
        .squeeze(0)
        .numpy()
        .astype(np.float32)
    )

    ldsr_matched = (
        ldsr_10m_tensor
        .squeeze(0)
        .numpy()
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # Save intermediate representations
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR / "bicubic_2p5m.npy",
        bicubic,
    )

    np.save(
        OUTPUT_DIR / "bicubic_matched_10m.npy",
        bicubic_matched,
    )

    np.save(
        OUTPUT_DIR / "ldsr_matched_10m.npy",
        ldsr_matched,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model(device)

    # --------------------------------------------------------
    # Native
    # --------------------------------------------------------

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
        "NATIVE 10m",
        native_metrics,
    )

    # --------------------------------------------------------
    # Bicubic
    # --------------------------------------------------------

    bicubic_prediction = predict(
        model,
        bicubic_matched,
        device,
    )

    bicubic_metrics = calculate_metrics(
        bicubic_prediction,
        label,
    )

    print_metrics(
        "BICUBIC → 2.5m → MATCHED BACK TO 10m",
        bicubic_metrics,
    )

    # --------------------------------------------------------
    # LDSR
    # --------------------------------------------------------

    ldsr_prediction = predict(
        model,
        ldsr_matched,
        device,
    )

    ldsr_metrics = calculate_metrics(
        ldsr_prediction,
        label,
    )

    print_metrics(
        "LDSR-S2 → 2.5m → MATCHED BACK TO 10m",
        ldsr_metrics,
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    comparison = {

        "native_10m": native_metrics,

        "bicubic_2p5m_matched": bicubic_metrics,

        "ldsr_s2_2p5m_matched": ldsr_metrics,

        "ldsr_minus_bicubic": {
            "pixel_accuracy":
                ldsr_metrics["pixel_accuracy"]
                - bicubic_metrics["pixel_accuracy"],

            "mean_iou":
                ldsr_metrics["mean_iou"]
                - bicubic_metrics["mean_iou"],

            "mean_dice":
                ldsr_metrics["mean_dice"]
                - bicubic_metrics["mean_dice"],

            "mean_precision":
                ldsr_metrics["mean_precision"]
                - bicubic_metrics["mean_precision"],

            "mean_recall":
                ldsr_metrics["mean_recall"]
                - bicubic_metrics["mean_recall"],
        },

        "methodology_note": (
            "Both 2.5m representations are reduced to "
            "the original 128x128 spatial dimensions "
            "before inference because the existing U-Net "
            "was trained at 128x128. WorldCover remains "
            "a 10m reference and is not genuine 2.5m "
            "ground truth."
        ),
    }

    with open(
        OUTPUT_DIR / "matched_resolution_metrics.json",
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
        OUTPUT_DIR / "native_prediction.npy",
        native_prediction,
    )

    np.save(
        OUTPUT_DIR / "bicubic_prediction.npy",
        bicubic_prediction,
    )

    np.save(
        OUTPUT_DIR / "ldsr_prediction.npy",
        ldsr_prediction,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MATCHED-RESOLUTION EXPERIMENT COMPLETE")
    print("=" * 70)

    print()
    print("LDSR vs Bicubic:")

    for key, value in comparison[
        "ldsr_minus_bicubic"
    ].items():

        print(
            f"  {key:<18}: "
            f"{value:+.6f}"
        )

    print()
    print(
        f"Results saved to:\n"
        f"{OUTPUT_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()