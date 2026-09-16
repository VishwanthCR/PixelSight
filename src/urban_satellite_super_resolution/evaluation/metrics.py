"""Research metrics for image, urban, change, and uncertainty quality."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def image_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction, target = np.asarray(prediction, dtype=np.float32), np.asarray(target, dtype=np.float32)
    return {
        "psnr": float(peak_signal_noise_ratio(target, prediction, data_range=1.0)),
        "ssim": float(structural_similarity(target, prediction, channel_axis=0, data_range=1.0)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "sam_degrees": spectral_angle_metric(prediction, target),
    }


def spectral_angle_metric(prediction: np.ndarray, target: np.ndarray) -> float:
    pred, true = np.moveaxis(prediction, 0, -1).reshape(-1, prediction.shape[0]), np.moveaxis(target, 0, -1).reshape(-1, target.shape[0])
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1)
    valid = denominator > 1e-8
    cosine = np.sum(pred[valid] * true[valid], axis=1) / denominator[valid]
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).mean()) if np.any(valid) else float("nan")


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray, classes: int = 5, ignore_index: int = 255) -> dict[str, object]:
    valid = target != ignore_index
    per_class = {}
    for class_id in range(classes):
        actual, predicted = (target == class_id) & valid, (prediction == class_id) & valid
        tp, fp, fn = np.logical_and(actual, predicted).sum(), np.logical_and(~actual, predicted).sum(), np.logical_and(actual, ~predicted).sum()
        per_class[str(class_id)] = {"iou": float(tp / max(tp + fp + fn, 1)), "f1": float(2 * tp / max(2 * tp + fp + fn, 1)), "precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1))}
    return {"per_class": per_class, "accuracy": float((prediction[valid] == target[valid]).mean()) if np.any(valid) else float("nan")}


def uncertainty_metrics(confidence: np.ndarray, errors: np.ndarray, correct: np.ndarray) -> dict[str, float | list[dict[str, float]]]:
    confidence, errors, correct = np.asarray(confidence).ravel(), np.asarray(errors).ravel(), np.asarray(correct).ravel().astype(bool)
    if confidence.size == 0:
        raise ValueError("Uncertainty arrays are empty")
    correlation = float(np.corrcoef(confidence, errors)[0, 1]) if confidence.size > 1 else float("nan")
    bins = np.linspace(0.0, 1.0, 11)
    calibration = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        selected = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if selected.any():
            calibration.append({"confidence": float(confidence[selected].mean()), "accuracy": float(correct[selected].mean()), "count": int(selected.sum())})
    ece = float(sum(abs(item["confidence"] - item["accuracy"]) * item["count"] for item in calibration) / confidence.size)
    return {"expected_calibration_error": ece, "uncertainty_error_correlation": correlation, "calibration_bins": calibration}
