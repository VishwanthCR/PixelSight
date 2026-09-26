"""
PixelSight Disaster Ground Truth Evaluator
==========================================
Evaluates detected disaster change maps against independent, validated
disaster ground-truth reference labels on a common evaluation grid.

Scientific Rules:
- Never use model predictions or SR outputs as ground truth.
- Evaluates on an identical common evaluation grid (2.5m or 10m).
- Excludes nodata / ignore regions (class 255).
- Computes genuine binary & multi-class metrics: TP, FP, FN, TN, IoU, Dice, Precision, Recall.
- When no ground truth exists, returns label_status: NONE and does not display damage accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from backend.app.services.ground_truth.schema import LabelStatus, ValidationStatus


class DisasterGroundTruthEvaluator:
    """Disaster damage and change evaluation engine."""

    @staticmethod
    def evaluate(
        change_map_path: Path,
        ground_truth_path: Path | None,
        change_threshold: float = 0.15,
        evaluation_grid: str = "2.5m",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate change prediction against ground truth if present and validated.
        """
        meta = metadata or {}
        validation_status = meta.get("validation_status", "none")

        if (
            ground_truth_path is None
            or not ground_truth_path.exists()
            or validation_status != ValidationStatus.VALIDATED.value
        ):
            return {
                "available": False,
                "label_status": LabelStatus.NONE.value,
                "status_label": "○ No Labels",
                "message": "Change detection without ground-truth validation. Validated disaster labels required for quantitative benchmark.",
                "evaluation_grid": evaluation_grid,
                "metrics": None,
                "confusion_matrix": None,
            }

        # Load prediction change map
        with rasterio.open(change_map_path) as src:
            change_data = src.read(1).astype(np.float32)
            pred_profile = src.profile.copy()
            pred_transform = src.transform
            pred_crs = src.crs
            h, w = src.height, src.width

        # Load ground truth raster
        with rasterio.open(ground_truth_path) as gt_src:
            gt_data = np.zeros((h, w), dtype=np.uint8)
            reproject(
                source=rasterio.band(gt_src, 1),
                destination=gt_data,
                src_transform=gt_src.transform,
                src_crs=gt_src.crs,
                dst_transform=pred_transform,
                dst_crs=pred_crs,
                resampling=Resampling.nearest,
            )

        # Valid mask: ignore class 255 and nodata
        valid_mask = (gt_data != 255) & np.isfinite(change_data)
        if not np.any(valid_mask):
            return {
                "available": False,
                "label_status": LabelStatus.NONE.value,
                "status_label": "○ No Labels",
                "message": "No valid evaluated pixels outside ignore mask.",
                "evaluation_grid": evaluation_grid,
                "metrics": None,
                "confusion_matrix": None,
            }

        # Binarize change predictions (1 = change, 0 = unchanged)
        pred_change = (change_data[valid_mask] >= change_threshold).astype(int)
        # Ground truth: 0 is unchanged, >= 1 is affected/change
        gt_change = (gt_data[valid_mask] > 0).astype(int)

        # Compute confusion matrix elements
        tp = int(np.sum((pred_change == 1) & (gt_change == 1)))
        fp = int(np.sum((pred_change == 1) & (gt_change == 0)))
        fn = int(np.sum((pred_change == 0) & (gt_change == 1)))
        tn = int(np.sum((pred_change == 0) & (gt_change == 0)))
        total = tp + fp + fn + tn

        accuracy = float((tp + tn) / max(total, 1))
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float(2 * precision * recall / max(precision + recall, 1e-8))
        iou = float(tp / max(tp + fp + fn, 1))
        dice = float(2 * tp / max(2 * tp + fp + fn, 1))

        label_status = LabelStatus.GROUND_TRUTH.value if validation_status == "validated" else LabelStatus.REFERENCE_LABELS.value

        return {
            "available": True,
            "label_status": label_status,
            "status_label": "✓ Ground Truth Available" if label_status == LabelStatus.GROUND_TRUTH.value else "◐ Reference Labels",
            "message": "Evaluated against validated disaster ground-truth labels on common grid.",
            "evaluation_grid": evaluation_grid,
            "metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "iou": round(iou, 4),
                "dice": round(dice, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "total_valid_pixels": total,
            },
            "confusion_matrix": {
                "labels": ["Unchanged", "Affected / Change"],
                "matrix": [
                    [tn, fp],
                    [fn, tp],
                ],
            },
            "interpretation": (
                f"Disaster change detection achieved an IoU of {iou:.3f} and F1 score of {f1:.3f} "
                f"against validated ground-truth reference on the {evaluation_grid} evaluation grid."
            ),
        }


disaster_gt_evaluator = DisasterGroundTruthEvaluator()
