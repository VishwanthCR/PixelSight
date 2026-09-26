"""
Ground Truth Subsystem Service
===============================
Manages vector ground-truth lifecycle for PixelSight:
- Workspace initialization & vector GeoJSON storage
- Validation & review workflow (draft -> review -> validated)
- Rasterization onto common evaluation grid
- Dual-model evaluation against identical ground truth
- Export of vector, raster, tabular metrics, and metadata
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.services.ground_truth.schema import (
    AnnotationCollection,
    GroundTruthMetadata,
    LabelStatus,
    ValidationReport,
    ValidationStatus,
    GT_CLASSES,
)
from backend.app.services.ground_truth.validator import annotation_validator
from backend.app.services.ground_truth.rasterizer import ground_truth_rasterizer
from backend.app.services.ground_truth.evaluator import ground_truth_evaluator


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_GT_ROOT = PROJECT_ROOT / "data" / "ground_truth"


class GroundTruthService:
    def __init__(self, storage_root: Optional[Path] = None):
        self.storage_root = storage_root or DEFAULT_GT_ROOT
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def _get_job_gt_dir(self, job_id: str, job_dir: Optional[Path] = None) -> Path:
        if job_dir is not None:
            d = Path(job_dir) / "ground_truth"
            d.mkdir(parents=True, exist_ok=True)
            return d
        d = self.storage_root / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_ground_truth_info(self, job_id: str, job_dir: Optional[Path] = None) -> Dict[str, Any]:
        gt_dir = self._get_job_gt_dir(job_id, job_dir)
        meta_path = gt_dir / "metadata.json"
        geojson_path = gt_dir / "annotations.geojson"
        raster_path = gt_dir / "ground_truth.tif"
        preview_path = gt_dir / "ground_truth.png"

        metadata = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                pass

        geojson = {"type": "FeatureCollection", "features": []}
        if geojson_path.exists():
            try:
                with open(geojson_path, "r", encoding="utf-8") as f:
                    geojson = json.load(f)
            except Exception:
                pass

        has_raster = raster_path.exists()
        has_annotations = len(geojson.get("features", [])) > 0
        val_status = metadata.get("validation_status", ValidationStatus.DRAFT.value)
        is_validated = val_status == ValidationStatus.VALIDATED.value

        label_status = LabelStatus.NONE.value
        if is_validated and has_raster:
            label_status = LabelStatus.GROUND_TRUTH.value
        elif has_raster:
            label_status = LabelStatus.REFERENCE_LABELS.value

        return {
            "job_id": job_id,
            "has_annotations": has_annotations,
            "has_raster": has_raster,
            "total_features": len(geojson.get("features", [])),
            "validation_status": val_status,
            "label_status": label_status,
            "metadata": metadata,
            "annotations": geojson,
            "files": {
                "geojson": str(geojson_path) if geojson_path.exists() else None,
                "raster": str(raster_path) if raster_path.exists() else None,
                "preview": str(preview_path) if preview_path.exists() else None,
            },
        }

    def save_annotations(
        self,
        job_id: str,
        geojson_data: Dict[str, Any],
        annotator: str = "Expert Annotator",
        notes: Optional[str] = None,
        job_dir: Optional[Path] = None,
        aoi_bbox: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        gt_dir = self._get_job_gt_dir(job_id, job_dir)
        geojson_path = gt_dir / "annotations.geojson"
        meta_path = gt_dir / "metadata.json"

        # Timestamp features
        now_iso = datetime.now(timezone.utc).isoformat()
        features = geojson_data.get("features", [])
        for feat in features:
            props = feat.setdefault("properties", {})
            props.setdefault("annotator", annotator)
            props.setdefault("created_at", now_iso)
            if notes and "notes" not in props:
                props["notes"] = notes

        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)

        # Update metadata
        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass

        meta["job_id"] = job_id
        meta.setdefault("source", "Expert Vector Annotation")
        meta.setdefault("created_at", now_iso)
        meta["updated_at"] = now_iso
        meta["annotator"] = annotator
        if aoi_bbox:
            meta["aoi"] = aoi_bbox
        # Resets status to draft on modification unless re-validated
        if meta.get("validation_status") == ValidationStatus.VALIDATED.value:
            meta["validation_status"] = ValidationStatus.REVIEW.value
            meta["label_status"] = LabelStatus.REFERENCE_LABELS.value
        else:
            meta.setdefault("validation_status", ValidationStatus.DRAFT.value)
            meta.setdefault("label_status", LabelStatus.NONE.value)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return self.get_ground_truth_info(job_id, job_dir)

    def validate_annotations(
        self,
        job_id: str,
        job_dir: Optional[Path] = None,
        aoi_bbox: Optional[List[float]] = None,
    ) -> ValidationReport:
        gt_dir = self._get_job_gt_dir(job_id, job_dir)
        geojson_path = gt_dir / "annotations.geojson"
        meta_path = gt_dir / "metadata.json"

        if not geojson_path.exists():
            report = annotation_validator.validate({"features": []}, aoi_bbox=aoi_bbox)
        else:
            with open(geojson_path, "r", encoding="utf-8") as f:
                geojson_data = json.load(f)
            report = annotation_validator.validate(geojson_data, aoi_bbox=aoi_bbox)

        # Store validation report into metadata
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["validation_report"] = report.model_dump()
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass

        return report

    def update_review_status(
        self,
        job_id: str,
        status: ValidationStatus,
        reviewer: Optional[str] = None,
        job_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Transitions review workflow: draft -> review -> validated.
        Only 'validated' can be marked as GROUND_TRUTH.
        """
        gt_dir = self._get_job_gt_dir(job_id, job_dir)
        meta_path = gt_dir / "metadata.json"

        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass

        meta["validation_status"] = status.value
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        if reviewer:
            meta["reviewer"] = reviewer

        if status == ValidationStatus.VALIDATED:
            meta["label_status"] = LabelStatus.GROUND_TRUTH.value
        elif status == ValidationStatus.REVIEW:
            meta["label_status"] = LabelStatus.REFERENCE_LABELS.value
        else:
            meta["label_status"] = LabelStatus.NONE.value

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return self.get_ground_truth_info(job_id, job_dir)

    def rasterize_and_evaluate(
        self,
        job_id: str,
        job_dir: Path,
        evaluation_grid: str = "2.5m",
        auto_validate: bool = False,
    ) -> Dict[str, Any]:
        """
        Burns annotations to ground_truth.tif and runs dual model evaluation
        against Native segmentation and SR segmentation.
        """
        gt_dir = self._get_job_gt_dir(job_id, job_dir)
        geojson_path = gt_dir / "annotations.geojson"
        meta_path = gt_dir / "metadata.json"
        output_tif = gt_dir / "ground_truth.tif"

        if not geojson_path.exists():
            raise FileNotFoundError(f"No vector annotations found for job {job_id}.")

        with open(geojson_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        # Template raster: prefer HR reference or SR tif for 2.5m grid
        sr_tif_candidates = [
            job_dir / "application" / "urban" / "segmentation_sr.tif",
            job_dir / "super_resolution" / "sr.tif",
            job_dir / "reference" / "aligned_reference.tif",
        ]
        native_tif_candidates = [
            job_dir / "application" / "urban" / "segmentation_native.tif",
            job_dir / "preprocessing" / "normalized.tif",
        ]

        template_sr = next((p for p in sr_tif_candidates if p.exists()), None)
        template_nat = next((p for p in native_tif_candidates if p.exists()), None)

        template_path = template_sr if evaluation_grid == "2.5m" else template_nat
        if not template_path:
            template_path = template_sr or template_nat

        # Extract AOI if available
        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass
        aoi_bbox = meta.get("aoi")

        # Rasterize
        stats = ground_truth_rasterizer.rasterize_to_grid(
            geojson_data=geojson_data,
            output_tif_path=output_tif,
            template_raster_path=template_path,
            aoi_bbox=aoi_bbox,
            target_resolution_m=2.5 if evaluation_grid == "2.5m" else 10.0,
            target_dims=(512, 512) if evaluation_grid == "2.5m" else (128, 128),
        )

        # Run evaluation if native & SR segmentation rasters are present
        sr_seg = job_dir / "application" / "urban" / "segmentation_sr.tif"
        nat_seg = job_dir / "application" / "urban" / "segmentation_native.tif"

        # Auto-trigger urban pipeline if segmentation is missing
        if not sr_seg.exists() or not nat_seg.exists():
            try:
                from backend.app.services.urban import run_urban_pipeline
                cand_sr = job_dir / "super_resolution" / "sr.tif"
                cand_nat = job_dir / "preprocessing" / "normalized.tif"
                if cand_sr.exists() and cand_nat.exists():
                    run_urban_pipeline(cand_nat, cand_sr, job_dir)
            except Exception:
                pass

        eval_report = {}
        if sr_seg.exists() and nat_seg.exists():
            eval_report = ground_truth_evaluator.evaluate(
                gt_raster_path=output_tif,
                sr_raster_path=sr_seg,
                native_raster_path=nat_seg,
                output_dir=gt_dir / "previews",
                evaluation_grid=evaluation_grid,
            )
            # Write metrics json
            with open(gt_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
                json.dump(eval_report, f, indent=2)

        # Update metadata
        meta["coverage_percentage"] = stats["coverage_percentage"]
        meta["ignored_percentage"] = stats["ignored_percentage"]
        meta["class_area_statistics"] = stats["class_area_statistics"]
        meta["evaluation_grid"] = evaluation_grid
        meta["dimensions"] = stats["dimensions"]
        meta["crs"] = stats["crs"]
        meta["bounds"] = stats["bounds"]
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()

        if auto_validate:
            meta["validation_status"] = ValidationStatus.VALIDATED.value
            meta["label_status"] = LabelStatus.GROUND_TRUTH.value

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        res = self.get_ground_truth_info(job_id, job_dir)
        res["raster_statistics"] = stats
        res["evaluation"] = eval_report
        return res

    def export_data(
        self,
        job_id: str,
        export_format: str,
        job_dir: Optional[Path] = None,
    ) -> Tuple[bytes, str, str]:
        """
        Exports GT assets in requested format:
        - geojson: vector annotations
        - geotiff: ground_truth.tif
        - csv: class area & metric comparison CSV
        - confusion_matrix: JSON
        - metadata: metadata.json
        """
        gt_dir = self._get_job_gt_dir(job_id, job_dir)
        export_format = export_format.lower()

        if export_format == "geojson":
            p = gt_dir / "annotations.geojson"
            if not p.exists():
                return b'{"type":"FeatureCollection","features":[]}', "application/geo+json", f"{job_id}_ground_truth.geojson"
            return p.read_bytes(), "application/geo+json", f"{job_id}_ground_truth.geojson"

        elif export_format in ("geotiff", "tif"):
            p = gt_dir / "ground_truth.tif"
            if not p.exists():
                raise FileNotFoundError("Ground truth raster has not been generated yet.")
            return p.read_bytes(), "image/tiff", f"{job_id}_ground_truth.tif"

        elif export_format == "metadata":
            p = gt_dir / "metadata.json"
            data = p.read_bytes() if p.exists() else b"{}"
            return data, "application/json", f"{job_id}_ground_truth_metadata.json"

        elif export_format == "confusion_matrix":
            p = gt_dir / "evaluation_report.json"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    rep = json.load(f)
                cm = rep.get("confusion_matrix", {})
            else:
                cm = {"native": None, "sr": None}
            return json.dumps(cm, indent=2).encode("utf-8"), "application/json", f"{job_id}_confusion_matrix.json"

        elif export_format == "csv":
            # Generate comprehensive per-class and distribution CSV
            p = gt_dir / "evaluation_report.json"
            p_meta = gt_dir / "metadata.json"
            out = io.StringIO()
            writer = csv.writer(out)

            writer.writerow(["PixelSight Scientific Ground Truth Evaluation Report"])
            writer.writerow(["Job ID", job_id])
            writer.writerow(["Exported At", datetime.now(timezone.utc).isoformat()])
            writer.writerow([])

            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    rep = json.load(f)
                ov = rep.get("overall", {})
                writer.writerow(["Overall Evaluation Metrics vs Ground Truth"])
                writer.writerow(["Metric", "Native (10m)", "PixelSight SR (~2.5m)", "Delta (SR - Native)"])
                for m in ("accuracy", "miou", "dice", "precision", "recall"):
                    n_v = ov.get("native", {}).get(m, "—")
                    s_v = ov.get("sr", {}).get(m, "—")
                    d_v = ov.get("delta", {}).get(m, "—")
                    writer.writerow([m.upper(), n_v, s_v, d_v])
                writer.writerow([])

                writer.writerow(["Per-Class Metrics vs Ground Truth"])
                writer.writerow(["Class ID", "Class Name", "Support (px)", "Native IoU", "SR IoU", "Delta IoU", "Native F1", "SR F1", "Delta F1"])
                for c in rep.get("per_class", []):
                    writer.writerow([
                        c.get("id"),
                        c.get("name"),
                        c.get("native", {}).get("support", 0),
                        c.get("native", {}).get("iou", "—"),
                        c.get("sr", {}).get("iou", "—"),
                        c.get("delta", {}).get("iou", "—"),
                        c.get("native", {}).get("f1", "—"),
                        c.get("sr", {}).get("f1", "—"),
                        c.get("delta", {}).get("f1", "—"),
                    ])
            else:
                writer.writerow(["No evaluated raster report available."])

            return out.getvalue().encode("utf-8"), "text/csv", f"{job_id}_ground_truth_report.csv"

        raise ValueError(f"Unsupported export format: {export_format}")


ground_truth_service = GroundTruthService()
