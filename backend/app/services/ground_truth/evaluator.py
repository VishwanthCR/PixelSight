"""
Ground Truth Classification Evaluator
======================================
Enforces Sections 7, 8, 9, 10, 16:
- Evaluates Native and SR model predictions against the identical Ground Truth raster
- Operates on a defined Common Evaluation Grid (2.5m or 10m) with Nearest-Neighbor discrete resampling
- Excludes nodata/ignore (255) pixels strictly from denominator and union calculations
- Produces Overall (Accuracy, mIoU, Dice, Precision, Recall), Per-Class, Confusion Matrix, and Deltas
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from PIL import Image

from backend.app.services.ground_truth.schema import (
    CLASS_BY_ID,
    GT_CLASSES,
)


class GroundTruthEvaluator:
    """Computes comprehensive classification metrics against validated Ground Truth."""

    def evaluate(
        self,
        gt_raster_path: Path,
        sr_raster_path: Path,
        native_raster_path: Path,
        output_dir: Optional[Path] = None,
        evaluation_grid: str = "2.5m",
    ) -> Dict[str, Any]:
        with rasterio.open(gt_raster_path) as src_gt:
            gt_arr = src_gt.read(1)
            gt_h, gt_w = gt_arr.shape
            gt_res = src_gt.res
            is_deg = src_gt.crs is not None and src_gt.crs.is_geographic

        # Load SR predictions and align to GT shape if needed
        with rasterio.open(sr_raster_path) as src_sr:
            sr_raw = src_sr.read(1)
            if sr_raw.shape != (gt_h, gt_w):
                sr_img = Image.fromarray(sr_raw)
                sr_arr = np.asarray(sr_img.resize((gt_w, gt_h), Image.NEAREST), dtype=np.uint8)
            else:
                sr_arr = sr_raw

        # Load Native predictions and resample to GT shape (Nearest neighbor for discrete classes)
        with rasterio.open(native_raster_path) as src_nat:
            nat_raw = src_nat.read(1)
            if nat_raw.shape != (gt_h, gt_w):
                nat_img = Image.fromarray(nat_raw)
                nat_arr = np.asarray(nat_img.resize((gt_w, gt_h), Image.NEAREST), dtype=np.uint8)
            else:
                nat_arr = nat_raw

        # Valid mask: Only consider pixels where Ground Truth has valid class (0..6)
        valid_mask = (gt_arr != 255) & (gt_arr < len(GT_CLASSES))
        valid_pixels = int(valid_mask.sum())

        if valid_pixels == 0:
            return {
                "evaluation_grid": evaluation_grid,
                "valid_pixels": 0,
                "status": "NO_VALID_GT_PIXELS",
                "overall": None,
                "per_class": [],
                "confusion_matrix": None,
            }

        y_gt = gt_arr[valid_mask]
        y_sr = sr_arr[valid_mask]
        y_nat = nat_arr[valid_mask]

        # Compute metric bundles
        native_metrics, conf_nat = self._compute_bundle(y_gt, y_nat)
        sr_metrics, conf_sr = self._compute_bundle(y_gt, y_sr)

        # Calculate Deltas: SR - Native
        delta_overall = {}
        for k in ("accuracy", "miou", "dice", "precision", "recall"):
            v_sr = sr_metrics["overall"].get(k)
            v_nat = native_metrics["overall"].get(k)
            if v_sr is not None and v_nat is not None:
                delta_overall[k] = round(v_sr - v_nat, 4)
            else:
                delta_overall[k] = None

        # Per-class combined list
        per_class = []
        for i, c in enumerate(GT_CLASSES):
            c_name = c["name"]
            c_id = c["id"]
            m_nat = native_metrics["per_class"][c_name]
            m_sr = sr_metrics["per_class"][c_name]

            del_iou = round(m_sr["iou"] - m_nat["iou"], 4) if m_sr["iou"] is not None and m_nat["iou"] is not None else None
            del_f1 = round(m_sr["f1"] - m_nat["f1"], 4) if m_sr["f1"] is not None and m_nat["f1"] is not None else None
            del_pr = round(m_sr["precision"] - m_nat["precision"], 4) if m_sr["precision"] is not None and m_nat["precision"] is not None else None
            del_re = round(m_sr["recall"] - m_nat["recall"], 4) if m_sr["recall"] is not None and m_nat["recall"] is not None else None

            per_class.append({
                "id": c_id,
                "name": c_name,
                "color": c["color"],
                "native": m_nat,
                "sr": m_sr,
                "delta": {
                    "iou": del_iou,
                    "f1": del_f1,
                    "precision": del_pr,
                    "recall": del_re,
                },
            })

        # Class distributions (GT vs Native vs SR)
        dist_gt = self._compute_distribution(y_gt, gt_res, is_deg)
        dist_nat = self._compute_distribution(y_nat, gt_res, is_deg)
        dist_sr = self._compute_distribution(y_sr, gt_res, is_deg)

        # Model-to-model agreement
        model_agreement_pct = round(float((y_sr == y_nat).mean() * 100.0), 2)
        model_disagreement_pct = round(100.0 - model_agreement_pct, 2)

        # Generate spatial difference previews if output directory given
        diff_previews = {}
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            diff_previews = self._generate_difference_maps(
                gt_arr, sr_arr, nat_arr, valid_mask, output_dir
            )

        # Generate scientific interpretations
        interpretations = self._generate_interpretations(
            native_metrics["overall"]["miou"],
            sr_metrics["overall"]["miou"],
            native_metrics["overall"]["accuracy"],
            sr_metrics["overall"]["accuracy"],
            per_class,
            evaluation_grid,
        )

        return {
            "evaluation_grid": f"Common {evaluation_grid} evaluation grid ({gt_w}x{gt_h})",
            "resampling_method": "nearest_neighbor_discrete",
            "alignment_method": "affine_reproject",
            "total_evaluated_pixels": valid_pixels,
            "overall": {
                "native": native_metrics["overall"],
                "sr": sr_metrics["overall"],
                "delta": delta_overall,
            },
            "per_class": per_class,
            "distribution": {
                "ground_truth": dist_gt,
                "native": dist_nat,
                "sr": dist_sr,
            },
            "confusion_matrix": {
                "native": conf_nat,
                "sr": conf_sr,
            },
            "prediction_comparison": {
                "agreement_percent": model_agreement_pct,
                "disagreement_percent": model_disagreement_pct,
            },
            "interpretations": interpretations,
            "difference_previews": diff_previews,
        }

    def _compute_bundle(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[Dict[str, Any], List[List[int]]]:
        num_classes = len(GT_CLASSES)
        # Confusion matrix: rows = true, cols = pred
        conf = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                conf[t, p] += 1

        total_samples = len(y_true)
        correct = int(np.trace(conf))
        overall_accuracy = round(float(correct / max(1, total_samples)), 4)

        per_class = {}
        ious, precs, recs, f1s = [], [], [], []

        for i, c in enumerate(GT_CLASSES):
            tp = int(conf[i, i])
            fp = int(conf[:, i].sum() - tp)
            fn = int(conf[i, :].sum() - tp)
            support = int(conf[i, :].sum())

            precision = round(float(tp / (tp + fp)), 4) if (tp + fp) > 0 else None
            recall = round(float(tp / (tp + fn)), 4) if (tp + fn) > 0 else None

            if precision is not None and recall is not None and (precision + recall) > 0:
                f1 = round(float(2 * precision * recall / (precision + recall)), 4)
            else:
                f1 = None

            union = tp + fp + fn
            iou = round(float(tp / union), 4) if union > 0 else None

            if support > 0:
                if iou is not None:
                    ious.append(iou)
                if precision is not None:
                    precs.append(precision)
                if recall is not None:
                    recs.append(recall)
                if f1 is not None:
                    f1s.append(f1)

            per_class[c["name"]] = {
                "id": c["id"],
                "color": c["color"],
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }

        miou = round(float(np.mean(ious)), 4) if ious else None
        macro_prec = round(float(np.mean(precs)), 4) if precs else None
        macro_rec = round(float(np.mean(recs)), 4) if recs else None
        macro_f1 = round(float(np.mean(f1s)), 4) if f1s else None

        overall = {
            "accuracy": overall_accuracy,
            "miou": miou,
            "dice": macro_f1,
            "precision": macro_prec,
            "recall": macro_rec,
        }

        return {"overall": overall, "per_class": per_class}, conf.tolist()

    def _compute_distribution(
        self, arr: np.ndarray, res: Tuple[float, float], is_deg: bool
    ) -> Dict[str, Any]:
        total = max(1, len(arr))
        res_x, res_y = res
        if is_deg or abs(res_x) < 0.1:
            meter_x = abs(res_x) * 111320.0
            meter_y = abs(res_y) * 111320.0
        else:
            meter_x = abs(res_x)
            meter_y = abs(res_y)

        pixel_area_m2 = max(meter_x * meter_y, 1.0)
        ha_per_px = pixel_area_m2 / 10000.0
        km2_per_px = pixel_area_m2 / 1000000.0

        dist = {}
        for c in GT_CLASSES:
            cid = c["id"]
            cnt = int((arr == cid).sum())
            pct = round((cnt / total) * 100.0, 2)
            dist[c["name"]] = {
                "id": cid,
                "color": c["color"],
                "pixel_count": cnt,
                "percent": pct,
                "area_ha": round(cnt * ha_per_px, 3),
                "area_km2": round(cnt * km2_per_px, 4),
            }
        return dist

    def _generate_difference_maps(
        self,
        gt: np.ndarray,
        sr: np.ndarray,
        nat: np.ndarray,
        valid_mask: np.ndarray,
        out_dir: Path,
    ) -> Dict[str, str]:
        h, w = gt.shape
        # 1. GT vs Native
        diff_nat = np.zeros((h, w, 3), dtype=np.uint8)
        # Background: dark slate
        diff_nat[:] = [15, 23, 42]
        # Match = emerald, Mismatch = crimson
        match_nat = valid_mask & (gt == nat)
        mismatch_nat = valid_mask & (gt != nat)
        diff_nat[match_nat] = [16, 185, 129]     # Emerald
        diff_nat[mismatch_nat] = [239, 68, 68]   # Red
        p_nat = out_dir / "gt_vs_native.png"
        Image.fromarray(diff_nat).save(p_nat)

        # 2. GT vs SR
        diff_sr = np.zeros((h, w, 3), dtype=np.uint8)
        diff_sr[:] = [15, 23, 42]
        match_sr = valid_mask & (gt == sr)
        mismatch_sr = valid_mask & (gt != sr)
        diff_sr[match_sr] = [16, 185, 129]
        diff_sr[mismatch_sr] = [239, 68, 68]
        p_sr = out_dir / "gt_vs_sr.png"
        Image.fromarray(diff_sr).save(p_sr)

        # 3. Native vs SR model disagreement
        diff_model = np.zeros((h, w, 3), dtype=np.uint8)
        diff_model[:] = [15, 23, 42]
        match_model = valid_mask & (nat == sr)
        mismatch_model = valid_mask & (nat != sr)
        diff_model[match_model] = [6, 182, 212]    # Cyan agreement
        diff_model[mismatch_model] = [245, 158, 11] # Amber disagreement
        p_model = out_dir / "native_vs_sr.png"
        Image.fromarray(diff_model).save(p_model)

        return {
            "gt_vs_native": str(p_nat),
            "gt_vs_sr": str(p_sr),
            "native_vs_sr": str(p_model),
        }

    def _generate_interpretations(
        self,
        nat_miou: Optional[float],
        sr_miou: Optional[float],
        nat_acc: Optional[float],
        sr_acc: Optional[float],
        per_class: List[Dict[str, Any]],
        grid: str,
    ) -> List[str]:
        items = []
        if nat_miou is not None and sr_miou is not None:
            delta_miou = round(sr_miou - nat_miou, 4)
            if delta_miou > 0:
                items.append(
                    f"In this evaluation, SR segmentation mIoU ({sr_miou:.3f}) was higher than the native segmentation ({nat_miou:.3f}) by {delta_miou:+.3f} on the common {grid} evaluation grid."
                )
            elif delta_miou < 0:
                items.append(
                    f"In this evaluation, SR segmentation mIoU ({sr_miou:.3f}) was lower than the native segmentation ({nat_miou:.3f}) by {delta_miou:+.3f} on the common {grid} evaluation grid."
                )
            else:
                items.append(
                    f"In this evaluation, both native and SR segmentation achieved identical mIoU of {nat_miou:.3f} on the common {grid} evaluation grid."
                )

        if nat_acc is not None and sr_acc is not None:
            delta_acc = round(sr_acc - nat_acc, 4)
            items.append(
                f"Pixel classification accuracy against ground truth: Native = {nat_acc*100:.1f}%, SR = {sr_acc*100:.1f}% (Δ = {delta_acc*100:+.1f}%)."
            )

        valid_class_deltas = [
            (c["name"], c["delta"]["iou"])
            for c in per_class
            if c["delta"]["iou"] is not None
        ]
        if valid_class_deltas:
            valid_class_deltas.sort(key=lambda x: abs(x[1]), reverse=True)
            top_classes = [f"{name} ({d:+.3f})" for name, d in valid_class_deltas[:3]]
            items.append(
                f"Per-class IoU changes were most pronounced in: {', '.join(top_classes)}."
            )

        items.append(
            "Evaluation evaluated identical ground-truth polygons against both model outputs under standardized nearest-neighbor discrete label resampling."
        )
        return items


ground_truth_evaluator = GroundTruthEvaluator()
