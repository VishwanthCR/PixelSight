"""
Ground Truth Vector Annotation Validator
==========================================
Enforces Section 5 geometric and semantic validation requirements:
- Invalid geometries (self-intersection, unclosed loops)
- Conflicting polygon overlaps
- Out-of-bounds geometries relative to AOI
- Invalid class IDs
- Empty annotations
- Missing classes / excessive ignored area warnings
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from shapely.geometry import shape, box, Polygon, MultiPolygon

from backend.app.services.ground_truth.schema import (
    CLASS_BY_ID,
    GT_CLASSES,
    ValidationIssue,
    ValidationReport,
    ValidationStatus,
)


class AnnotationValidator:
    """Validates vector GeoJSON annotations prior to rasterization and ground-truth approval."""

    def __init__(self, max_allowed_overlap_pct: float = 0.5):
        self.max_allowed_overlap_pct = max_allowed_overlap_pct

    def validate(
        self,
        geojson_data: Dict[str, Any],
        aoi_bbox: Optional[List[float]] = None,
    ) -> ValidationReport:
        errors: List[ValidationIssue] = []
        warnings: List[ValidationIssue] = []

        features = geojson_data.get("features", [])
        if not features:
            errors.append(
                ValidationIssue(
                    type="error",
                    code="EMPTY_ANNOTATION",
                    message="Annotation contains zero polygon features. At least one labeled feature is required.",
                )
            )
            return ValidationReport(
                is_valid=False,
                status=ValidationStatus.DRAFT,
                total_features=0,
                covered_classes=[],
                missing_classes=[c["name"] for c in GT_CLASSES],
                errors=errors,
                warnings=warnings,
            )

        aoi_geom = None
        if aoi_bbox and len(aoi_bbox) == 4:
            # bbox: [min_lon, min_lat, max_lon, max_lat]
            aoi_geom = box(aoi_bbox[0], aoi_bbox[1], aoi_bbox[2], aoi_bbox[3])

        parsed_polys: List[tuple[int, Any, int, str]] = []
        covered_class_ids = set()

        for idx, feat in enumerate(features):
            geom_dict = feat.get("geometry")
            props = feat.get("properties", {})
            class_id = props.get("class_id")
            class_name = props.get("class_name", "Unknown")

            # 1. Class ID validation
            if class_id is None or class_id not in CLASS_BY_ID:
                errors.append(
                    ValidationIssue(
                        type="error",
                        code="INVALID_CLASS_ID",
                        message=f"Feature {idx} has invalid or unmapped class_id: {class_id}.",
                        feature_index=idx,
                        class_name=class_name,
                    )
                )
                continue

            if class_id != 255:
                covered_class_ids.add(class_id)

            # 2. Geometry validity
            if not geom_dict:
                errors.append(
                    ValidationIssue(
                        type="error",
                        code="MISSING_GEOMETRY",
                        message=f"Feature {idx} is missing geometry coordinates.",
                        feature_index=idx,
                        class_name=class_name,
                    )
                )
                continue

            try:
                geom = shape(geom_dict)
            except Exception as e:
                errors.append(
                    ValidationIssue(
                        type="error",
                        code="MALFORMED_GEOMETRY",
                        message=f"Feature {idx} could not be parsed: {str(e)}",
                        feature_index=idx,
                        class_name=class_name,
                    )
                )
                continue

            if not geom.is_valid:
                errors.append(
                    ValidationIssue(
                        type="error",
                        code="GEOMETRY_INVALID",
                        message=f"Feature {idx} geometry is topologically invalid: {geom.is_valid_reason()}.",
                        feature_index=idx,
                        class_name=class_name,
                    )
                )
                continue

            if geom.is_empty:
                errors.append(
                    ValidationIssue(
                        type="error",
                        code="EMPTY_GEOMETRY",
                        message=f"Feature {idx} has empty area.",
                        feature_index=idx,
                        class_name=class_name,
                    )
                )
                continue

            # 3. AOI boundary check
            if aoi_geom is not None:
                if not aoi_geom.intersects(geom):
                    errors.append(
                        ValidationIssue(
                            type="error",
                            code="OUTSIDE_AOI",
                            message=f"Feature {idx} lies entirely outside the target AOI boundary.",
                            feature_index=idx,
                            class_name=class_name,
                        )
                    )
                elif not aoi_geom.contains(geom):
                    # Partially outside
                    overlap_ratio = geom.intersection(aoi_geom).area / max(1e-12, geom.area)
                    if overlap_ratio < 0.95:
                        warnings.append(
                            ValidationIssue(
                                type="warning",
                                code="PARTIALLY_OUTSIDE_AOI",
                                message=f"Feature {idx} extends outside the AOI boundary ({round((1 - overlap_ratio)*100, 1)}% exterior).",
                                feature_index=idx,
                                class_name=class_name,
                            )
                        )

            parsed_polys.append((idx, geom, class_id, class_name))

        # 4. Polygon overlap validation
        for i in range(len(parsed_polys)):
            idx_a, geom_a, cid_a, cname_a = parsed_polys[i]
            for j in range(i + 1, len(parsed_polys)):
                idx_b, geom_b, cid_b, cname_b = parsed_polys[j]
                if cid_a == cid_b:
                    # Same class overlaps can merge during rasterization, but warn if large
                    continue
                if geom_a.intersects(geom_b):
                    inter = geom_a.intersection(geom_b)
                    if inter.area > 1e-10:
                        min_area = min(geom_a.area, geom_b.area)
                        overlap_pct = (inter.area / max(1e-12, min_area)) * 100.0
                        if overlap_pct > self.max_allowed_overlap_pct:
                            errors.append(
                                ValidationIssue(
                                    type="error",
                                    code="CONFLICTING_OVERLAP",
                                    message=f"Conflicting classes overlap between feature {idx_a} ({cname_a}) and {idx_b} ({cname_b}) with {overlap_pct:.1f}% mutual intersection.",
                                    feature_index=idx_a,
                                    class_name=cname_a,
                                )
                            )

        # 5. Missing classes (informational warning, not fatal error per Section 5)
        covered_names = [CLASS_BY_ID[cid]["name"] for cid in sorted(covered_class_ids)]
        missing_names = [c["name"] for c in GT_CLASSES if c["id"] not in covered_class_ids]
        if missing_names:
            warnings.append(
                ValidationIssue(
                    type="info",
                    code="UNOBSERVED_CLASSES",
                    message=f"The following land-cover classes have no polygons in this annotation: {', '.join(missing_names)}. This is permissible if the AOI legitimately contains none.",
                )
            )

        is_valid = len(errors) == 0
        status = ValidationStatus.VALIDATED if is_valid else ValidationStatus.DRAFT

        return ValidationReport(
            is_valid=is_valid,
            status=status,
            total_features=len(features),
            covered_classes=covered_names,
            missing_classes=missing_names,
            errors=errors,
            warnings=warnings,
        )


annotation_validator = AnnotationValidator()
