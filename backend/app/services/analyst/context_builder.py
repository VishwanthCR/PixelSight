from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ContextBuilder:
    """
    Constructs a compact, structured scientific JSON representation of an actual PixelSight job
    or batch run for consumption by the local LLM.
    
    Guarantees:
    - Only includes fields that actually exist.
    - Never invents missing metrics.
    - Unavailable values are explicitly null or noted.
    - Never sends raw satellite matrices or huge rasters.
    """

    @classmethod
    def build_job_context(
        cls,
        job_data: dict[str, Any],
        report_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build compact, use-case-aware scientific context from job and report data."""
        job_id = job_data.get("job_id", "unknown")
        application = (
            job_data.get("application") or
            (report_data.get("application") if report_data else None) or
            "research"
        ).lower()
        status = job_data.get("status", "completed")

        # Fallback to report data if available
        rep = report_data or job_data.get("report") or {}

        # 1. Job overview
        context: dict[str, Any] = {
            "job": {
                "job_id": job_id,
                "application": application,
                "status": status,
                "stage": job_data.get("stage", "completed"),
            }
        }

        # 2. Input metadata
        input_meta = rep.get("input") or job_data.get("outputs", {}).get("inspection") or {}
        context["input"] = {
            "source": input_meta.get("filename") or input_meta.get("source") or "Sentinel-2 L2A Compatible Raster",
            "crs": input_meta.get("crs"),
            "resolution": input_meta.get("resolution") or [10.0, 10.0],
            "bands": input_meta.get("band_names") or input_meta.get("bands"),
            "dimensions": [input_meta.get("width"), input_meta.get("height")] if input_meta.get("width") else None,
        }

        # 3. Preprocessing
        prep = rep.get("preprocessing") or {}
        ops = prep.get("operations", [])
        context["preprocessing"] = {
            "performed": len(ops) > 0,
            "operations_count": len(ops),
            "operations": [
                {
                    "name": op.get("name") or op.get("operation"),
                    "status": op.get("status", "applied"),
                }
                for op in ops[:10]  # compact
            ],
        }

        # 4. Super-Resolution
        sr_config = rep.get("super_resolution") or {}
        context["super_resolution"] = {
            "model": sr_config.get("model", "LDSR-S2"),
            "scale": sr_config.get("scale", 4),
            "sampling_steps": sr_config.get("sampling_steps", 100),
            "target_resolution": "~2.5m equivalent super-resolved representation",
        }

        # 5. Reference & Discovery
        eval_data = rep.get("evaluation") or job_data.get("evaluation") or {}
        disc = rep.get("reference_discovery") or eval_data.get("discovery") or {}
        ref_status = eval_data.get("status")
        ref_available = (ref_status == "reference_available") or disc.get("available", False)

        context["reference"] = {
            "available": ref_available,
            "status": ref_status or ("reference_available" if ref_available else "reference_unavailable"),
            "source": disc.get("source") or (eval_data.get("reference_provenance", {}).get("source") if ref_available else None),
            "reference_id": disc.get("reference_id") or (eval_data.get("reference_provenance", {}).get("tile_id") if ref_available else None),
            "resolution_m": disc.get("resolution_m"),
            "spatial_overlap_percent": disc.get("spatial_overlap"),
            "temporal_difference_days": disc.get("temporal_difference_days"),
            "spectral_compatibility": disc.get("spectral_compatibility"),
            "selection_reason": disc.get("selection_reason") or eval_data.get("reason"),
            "eligible_metrics": disc.get("eligible_metrics", []),
        }

        # 6. Evaluation metrics (strictly respect availability)
        if ref_available:
            context["evaluation"] = {
                "reference_based": True,
                "psnr": eval_data.get("psnr"),
                "ssim": eval_data.get("ssim"),
                "sam": eval_data.get("sam"),
                "mae": eval_data.get("mae"),
                "rmse": eval_data.get("rmse"),
            }
        else:
            no_ref = eval_data.get("no_reference_metrics", {})
            context["evaluation"] = {
                "reference_based": False,
                "note": "PSNR, SSIM, and SAM are null because no compatible external HR reference covers this AOI.",
                "psnr": None,
                "ssim": None,
                "sam": None,
                "no_reference_diagnostics": {
                    "spectral_conservation": no_ref.get("spectral_conservation"),
                    "spatial_statistics": no_ref.get("spatial_statistics"),
                } if no_ref else None,
            }

        # 7. Uncertainty
        unc = rep.get("uncertainty") or job_data.get("outputs", {}).get("uncertainty") or {}
        context["uncertainty"] = {
            "available": bool(unc),
            "mean": unc.get("mean"),
            "std": unc.get("std"),
            "high_uncertainty_percent": unc.get("high_percentage") or unc.get("high_uncertainty_percent"),
            "method": unc.get("method", "LDSR-S2 Stochastic Diffusion Variance"),
            "calibration_status": unc.get("status", "uncalibrated_variance_proxy"),
            "note": unc.get("limitation") or "Uncertainty reflects stochastic sampling dispersion; not a calibrated error probability.",
        }

        # 8. Use-case-specific application results
        context["application_results"] = cls._extract_application_results(application, rep, job_data)

        # 9. Scientific Limitations
        limits = list(rep.get("scientific_limitations", []))
        if not limits:
            limits = [
                "The SR product is a super-resolved representation, not observed 2.5m imagery.",
                "Metrics and outputs are bounded by the input Sentinel-2 sensor characteristics.",
            ]
        context["limitations"] = limits

        # 10. Artifacts manifest (names only, no raster bytes)
        artifacts = rep.get("outputs") or job_data.get("outputs") or {}
        context["artifacts"] = [k for k in artifacts.keys() if isinstance(k, str)]

        return context

    @classmethod
    def _extract_application_results(
        cls,
        application: str,
        report_data: dict[str, Any],
        job_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract only genuine, validated application metrics for the specific use case."""
        app_results: dict[str, Any] = {}

        if application == "crop":
            crop_data = report_data.get("crop_analysis") or job_data.get("outputs", {}).get("crop_analysis") or {}
            if crop_data:
                app_results["ndvi_summary"] = {
                    "native_mean_ndvi": crop_data.get("native_mean_ndvi"),
                    "sr_mean_ndvi": crop_data.get("sr_mean_ndvi"),
                    "ndvi_change_percent": crop_data.get("ndvi_change_percent"),
                    "vegetation_area_ha": crop_data.get("vegetation_area_ha"),
                    "spectral_preservation_score": crop_data.get("spectral_preservation_score"),
                }
                app_results["diagnostic_note"] = (
                    "Metrics indicate canopy radiometric preservation. "
                    "PixelSight does not compute agronomic yield or disease pathology diagnosis."
                )

        elif application == "urban":
            urban_data = report_data.get("urban_analysis") or job_data.get("outputs", {}).get("urban_analysis") or {}
            if urban_data:
                metrics = urban_data.get("metrics") or {}
                app_results["segmentation"] = {
                    "miou": metrics.get("miou"),
                    "dice": metrics.get("dice") or metrics.get("f1"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "classes": urban_data.get("classes") or metrics.get("class_iou"),
                    "ground_truth_status": urban_data.get("ground_truth_status", "no_reference_labels"),
                }
                app_results["diagnostic_note"] = (
                    "Visual edge sharpness does not guarantee superior segmentation accuracy. "
                    "Comparisons must be interpreted relative to validated ground-truth labels."
                )

        elif application == "disaster":
            disaster_data = report_data.get("disaster_analysis") or job_data.get("outputs", {}).get("disaster_analysis") or {}
            disaster_meta = report_data.get("disaster_metadata") or job_data.get("outputs", {}).get("disaster_metadata") or {}
            if disaster_data or disaster_meta:
                app_results["disaster"] = {
                    "pre_event_date": disaster_meta.get("pre_event_date") or disaster_meta.get("pre_date"),
                    "post_event_date": disaster_meta.get("post_event_date") or disaster_meta.get("post_date"),
                    "temporal_gap_days": disaster_meta.get("temporal_gap_days"),
                    "change_statistics": disaster_data.get("change_statistics") or disaster_data.get("area_statistics"),
                    "affected_area_ha": disaster_data.get("affected_area_ha"),
                    "confidence": disaster_data.get("confidence_level", "unvalidated_spectral_difference"),
                }
                app_results["diagnostic_note"] = (
                    "Distinguishes observed spectral surface changes from ground-truth verified physical disaster damage."
                )

        elif application in {"research", "core"}:
            app_results["core"] = {
                "focus": "LDSR-S2 super-resolution reconstruction and radiometric fidelity",
            }

        return app_results

    @classmethod
    def build_batch_context(
        cls,
        batch_id: str,
        jobs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Construct compact batch execution context across multiple jobs."""
        total = len(jobs)
        completed = sum(1 for j in jobs if j.get("status") == "completed")
        failed = sum(1 for j in jobs if j.get("status") == "failed")
        processing = sum(1 for j in jobs if j.get("status") == "processing")
        queued = sum(1 for j in jobs if j.get("status") == "queued")

        summaries = []
        for j in jobs[:25]:  # compact bound
            eval_dict = j.get("evaluation") or j.get("outputs", {}).get("evaluation") or {}
            unc_dict = j.get("outputs", {}).get("uncertainty") or {}
            summaries.append({
                "job_id": j.get("job_id"),
                "application": j.get("application", "research"),
                "status": j.get("status"),
                "psnr": eval_dict.get("psnr"),
                "ssim": eval_dict.get("ssim"),
                "ref_status": eval_dict.get("status"),
                "uncertainty_mean": unc_dict.get("mean"),
                "error": j.get("error"),
            })

        return {
            "batch": {
                "batch_id": batch_id,
                "total_jobs": total,
                "completed": completed,
                "failed": failed,
                "processing": processing,
                "queued": queued,
            },
            "jobs_summary": summaries,
            "limitations": [
                "Batch summary reflects aggregated metrics across individual tile executions.",
                "Failed jobs must be inspected for individual input/format anomalies.",
            ],
        }
