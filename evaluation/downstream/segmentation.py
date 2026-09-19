"""
Downstream segmentation evaluation.

Scientific conventions
----------------------
* Metrics are per-class (IoU, F1, Dice, precision, recall) and aggregate
  (mIoU, mDice, accuracy).
* When WorldCover 10 m labels are used as proxy labels for a 2.5 m SR
  evaluation, this must be explicitly documented.  The label_is_proxy flag
  must be True and methodological_note must explain the limitation.
* Do NOT modify labels to improve metrics.
* Negative results must be preserved and reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Result containers (re-exported from results.py to keep downstream self-contained)
# ---------------------------------------------------------------------------

@dataclass
class SegmentationResult:
    """Per-class and aggregate segmentation metrics."""

    # Aggregate
    pixel_accuracy: float | None = None
    mean_iou: float | None = None
    mean_dice: float | None = None
    mean_precision: float | None = None
    mean_recall: float | None = None

    # Per-class
    per_class_iou: dict[str, float] = field(default_factory=dict)
    per_class_dice: dict[str, float] = field(default_factory=dict)
    per_class_precision: dict[str, float] = field(default_factory=dict)
    per_class_recall: dict[str, float] = field(default_factory=dict)
    per_class_f1: dict[str, float] = field(default_factory=dict)

    # Provenance
    representation: str = ""       # "native_10m" | "bicubic_2p5m" | "pixelsight_2p5m"
    num_classes: int = 0
    class_names: list[str] = field(default_factory=list)
    ignore_index: int = 255
    label_type: str = ""
    label_is_proxy: bool = True
    methodological_note: str = ""


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _iou(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else float("nan")


def _dice(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else float("nan")


def _precision(tp: int, fp: int) -> float:
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")


def _recall(tp: int, fn: int) -> float:
    return float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")


def _f1(tp: int, fp: int, fn: int) -> float:
    p = _precision(tp, fp)
    r = _recall(tp, fn)
    if np.isnan(p) or np.isnan(r) or (p + r) == 0:
        return float("nan")
    return float(2 * p * r / (p + r))


def segmentation_metrics(
    prediction: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None = None,
    num_classes: int | None = None,
    ignore_index: int = 255,
    representation: str = "",
    label_is_proxy: bool = True,
    label_type: str = "worldcover_10m_proxy",
    methodological_note: str = "",
) -> SegmentationResult:
    """Compute segmentation metrics.

    Parameters
    ----------
    prediction : np.ndarray, shape (H, W)
        Predicted class labels (integer).
    labels : np.ndarray, shape (H, W)
        Ground-truth class labels.
    class_names : list[str], optional
        Human-readable class names.
    num_classes : int, optional
        Number of classes. Inferred from labels if not provided.
    ignore_index : int
        Label value to exclude from all metrics (default 255).
    representation : str
        Which representation produced this prediction.
    label_is_proxy : bool
        True if labels are proxy (e.g. 10 m WorldCover replicated to 2.5 m).
    label_type : str
        Description of the label source.
    methodological_note : str
        Free-text note about methodological limitations.

    Returns
    -------
    SegmentationResult
    """
    prediction = np.asarray(prediction).ravel()
    labels = np.asarray(labels).ravel()

    valid_mask = labels != ignore_index
    pred_valid = prediction[valid_mask]
    lab_valid = labels[valid_mask]

    if num_classes is None:
        num_classes = int(lab_valid.max()) + 1 if len(lab_valid) > 0 else 0

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    result = SegmentationResult(
        representation=representation,
        num_classes=num_classes,
        class_names=class_names,
        ignore_index=ignore_index,
        label_type=label_type,
        label_is_proxy=label_is_proxy,
        methodological_note=methodological_note,
    )

    if len(pred_valid) == 0:
        return result

    # Pixel accuracy
    result.pixel_accuracy = float((pred_valid == lab_valid).mean())

    # Per-class
    ious, dices, precs, recs, f1s = [], [], [], [], []
    for cls_id in range(num_classes):
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        actual = lab_valid == cls_id
        predicted = pred_valid == cls_id
        tp = int(np.logical_and(actual, predicted).sum())
        fp = int(np.logical_and(~actual, predicted).sum())
        fn = int(np.logical_and(actual, ~predicted).sum())
        iou_v = _iou(tp, fp, fn)
        dice_v = _dice(tp, fp, fn)
        prec_v = _precision(tp, fp)
        rec_v = _recall(tp, fn)
        f1_v = _f1(tp, fp, fn)
        result.per_class_iou[name] = iou_v
        result.per_class_dice[name] = dice_v
        result.per_class_precision[name] = prec_v
        result.per_class_recall[name] = rec_v
        result.per_class_f1[name] = f1_v
        ious.append(iou_v)
        dices.append(dice_v)
        precs.append(prec_v)
        recs.append(rec_v)
        f1s.append(f1_v)

    # Aggregate (mean over classes, excluding NaN)
    def _safe_mean(vals: list[float]) -> float | None:
        finite = [v for v in vals if not np.isnan(v)]
        return float(np.mean(finite)) if finite else None

    result.mean_iou = _safe_mean(ious)
    result.mean_dice = _safe_mean(dices)
    result.mean_precision = _safe_mean(precs)
    result.mean_recall = _safe_mean(recs)

    return result


def aggregate_segmentation(
    results: list[SegmentationResult],
) -> dict[str, float | None]:
    """Average aggregate metrics across multiple tiles."""
    fields = ["pixel_accuracy", "mean_iou", "mean_dice", "mean_precision", "mean_recall"]
    out: dict[str, float | None] = {}
    for f in fields:
        vals = [getattr(r, f) for r in results if getattr(r, f) is not None]
        out[f] = float(np.mean(vals)) if vals else None
    return out
