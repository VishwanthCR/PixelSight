"""
Reference Discovery Service
===========================
Orchestrates automatic discovery of external HR reference products for:
1. Map AOI bounding boxes / GeoJSON polygons, and
2. Georeferenced uploaded satellite GeoTIFF files.

Gracefully falls back to no-reference evaluation when no suitable reference exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .registry import ReferenceRegistry, ReferenceTile, ReferenceSource
from .spatial_match import SpatialMatcher, SpatialMatchResult
from .temporal_match import TemporalMatcher, TemporalMatchResult
from .spectral_match import SpectralMatcher, SpectralMatchResult
from .downloader import ReferenceDownloader

try:
    import rasterio
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False


@dataclass
class CandidateEvaluation:
    """Internal candidate evaluation record."""
    source_name: str
    tile_id: str
    tile_name: str
    spatial_result: SpatialMatchResult
    temporal_result: TemporalMatchResult
    spectral_result: SpectralMatchResult
    resolution_m: float
    is_eligible: bool
    score: float
    reasons: list[str]


@dataclass
class ReferenceDiscoveryResult:
    """Standardized result returned by reference discovery."""
    available: bool = False
    source: str | None = None
    reference_id: str | None = None
    reference_path: str | None = None
    resolution_m: float | None = None
    spatial_overlap: float = 0.0
    temporal_difference_days: int | None = None
    temporal_match_status: str = "UNKNOWN"
    spectral_compatibility: str = "INCOMPATIBLE"
    match_status: str = "UNAVAILABLE"
    selection_reason: str = "No compatible HR reference found."
    limitations: list[str] = field(default_factory=list)
    eligible_metrics: list[str] = field(default_factory=list)
    ineligible_metrics: dict[str, str] = field(default_factory=dict)
    band_mapping: dict[str, int | None] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    candidates_searched: int = 0
    candidate_summary: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReferenceDiscoveryService:
    """High-level discovery service for PixelSight."""

    def __init__(
        self,
        registry: ReferenceRegistry | None = None,
        downloader: ReferenceDownloader | None = None,
    ) -> None:
        self.registry = registry or ReferenceRegistry()
        self.downloader = downloader or ReferenceDownloader(self.registry.cache_dir)
        self.spatial_matcher = SpatialMatcher(min_overlap_pct=self.registry.min_spatial_overlap_pct)
        self.temporal_matcher = TemporalMatcher(
            exact_days=self.registry.temporal_thresholds_days.get("exact", 0),
            close_days=self.registry.temporal_thresholds_days.get("close", 7),
            acceptable_with_caveat_days=self.registry.temporal_thresholds_days.get("acceptable_with_caveat", 30),
        )
        self.spectral_matcher = SpectralMatcher()

    def discover_for_geotiff(
        self,
        image_path: str | Path,
        acquisition_date: str | datetime | None = None,
        aoi: list[float] | tuple[float, float, float, float] | None = None,
        source_preference: str | None = None,
        job_dir: Path | None = None,
    ) -> ReferenceDiscoveryResult:
        """Extract geospatial metadata from uploaded GeoTIFF and run reference discovery."""
        image_path = Path(image_path)
        suffix = image_path.suffix.lower()

        # Rule 6: Strictly reject non-geospatial formats
        if suffix in (".png", ".jpg", ".jpeg"):
            return ReferenceDiscoveryResult(
                available=False,
                match_status="NON_GEOSPATIAL",
                selection_reason="Geospatial metadata is unavailable for this uploaded image format (PNG/JPEG).",
                limitations=["Standard image formats do not carry georeferencing coordinate systems."],
            )

        if not _RASTERIO_AVAILABLE:
            return ReferenceDiscoveryResult(
                available=False,
                match_status="RASTERIO_UNAVAILABLE",
                selection_reason="Rasterio library unavailable to parse GeoTIFF metadata.",
            )

        if not image_path.exists():
            return ReferenceDiscoveryResult(
                available=False,
                match_status="FILE_NOT_FOUND",
                selection_reason=f"File does not exist: {image_path}",
            )

        # 1. Resolve acquisition date from parameter, sidecar JSON, job directory, tags, or filename
        acq_date = acquisition_date

        # Check job_dir metadata (copernicus_request.json or source_metadata.json)
        cand_job_dirs = []
        if job_dir:
            cand_job_dirs.append(Path(job_dir))
        cand_job_dirs.append(image_path.parent.parent)
        cand_job_dirs.append(image_path.parent)

        if not acq_date:
            for jd in cand_job_dirs:
                if not jd.exists():
                    continue
                for fname in ("copernicus_request.json", "source_metadata.json", "metadata.json"):
                    meta_p = jd / fname if (jd / fname).exists() else jd / "input" / fname
                    if meta_p.exists():
                        try:
                            with open(meta_p, "r", encoding="utf-8") as f:
                                mdata = json.load(f)
                            found_d = (
                                mdata.get("acquisition_date")
                                or mdata.get("date")
                                or mdata.get("datetime")
                                or mdata.get("time_range", {}).get("from")
                            )
                            if found_d:
                                acq_date = found_d
                                break
                        except Exception:
                            pass
                if acq_date:
                    break

        try:
            with rasterio.open(image_path) as src:
                crs = str(src.crs) if src.crs else "EPSG:4326"
                bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
                resolution = float(src.res[0])
                if resolution < 0.1:
                    resolution = resolution * 111320.0
                bands = [src.descriptions[i] or f"B{i+1}" for i in range(src.count)]

                # Look for acquisition date in TIFF tags if not yet resolved
                if not acq_date:
                    tags = src.tags()
                    acq_date = (
                        tags.get("DATETIME")
                        or tags.get("ACQUISITION_DATE")
                        or tags.get("TIFFTAG_DATETIME")
                        or tags.get("acquisition_time")
                    )
        except Exception as e:
            return ReferenceDiscoveryResult(
                available=False,
                match_status="INVALID_GEOTIFF",
                selection_reason=f"Failed to inspect GeoTIFF metadata: {e}",
            )

        # Fallback to filename date extraction if still missing
        if not acq_date:
            import re
            m = re.search(r"(?:^|[^0-9])(20\d{2})[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])(?:[^0-9]|$)", image_path.name)
            if m:
                acq_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

        target_aoi = aoi if aoi is not None else bounds

        return self.discover(
            aoi=target_aoi,
            aoi_crs=crs,
            acquisition_date=acq_date,
            input_resolution_m=resolution,
            source_preference=source_preference,
        )

    def discover(
        self,
        aoi: list[float] | tuple[float, float, float, float] | dict[str, Any],
        aoi_crs: str = "EPSG:4326",
        acquisition_date: str | datetime | None = None,
        input_resolution_m: float = 10.0,
        source_preference: str | None = None,
    ) -> ReferenceDiscoveryResult:
        """Search registered catalogs for the best eligible external HR reference."""
        if input_resolution_m < 0.1:
            input_resolution_m = input_resolution_m * 111320.0
        candidates = self.registry.get_all_tiles()
        if source_preference:
            candidates = [t for t in candidates if t.dataset == source_preference]

        if not candidates:
            return ReferenceDiscoveryResult(
                available=False,
                match_status="NO_CANDIDATES",
                selection_reason="No candidate reference datasets registered for this query.",
                limitations=["No catalog entries matched the search filter."],
            )

        evaluated: list[CandidateEvaluation] = []

        for tile in candidates:
            # 1. Spatial Matching
            spat_res = self.spatial_matcher.compute_match(
                aoi=aoi,
                candidate_bbox=tile.bbox,
                candidate_geometry=tile.geometry,
                aoi_crs=aoi_crs,
                candidate_crs=tile.metadata.get("crs", "EPSG:4326"),
            )

            # 2. Resolution Check
            res_ok = (
                tile.resolution_m <= self.registry.max_acceptable_resolution_m
                and tile.resolution_m < input_resolution_m
            )

            # 3. Spectral Matching
            spec_res = self.spectral_matcher.evaluate(
                reference_bands=tile.bands,
                reference_band_count=len(tile.bands),
            )

            # 4. Temporal Matching
            ref_dt = tile.datetime
            if not ref_dt:
                ref_dt = tile.metadata.get("datetime") or tile.metadata.get("acquisition_date") or tile.id
            temp_res = self.temporal_matcher.evaluate(
                target_date=acquisition_date,
                reference_date=ref_dt,
            )

            is_eligible = (
                spat_res.intersects
                and res_ok
                and spec_res.compatibility != "INCOMPATIBLE"
                and temp_res.is_acceptable
            )

            reasons: list[str] = []
            if not spat_res.intersects:
                reasons.append(f"Insufficient spatial overlap ({spat_res.overlap_pct:.1f}% < {self.registry.min_spatial_overlap_pct}%).")
            if not res_ok:
                reasons.append(f"Resolution ({tile.resolution_m}m) does not satisfy requirements.")
            if spec_res.compatibility == "INCOMPATIBLE":
                reasons.append("Spectral bands incompatible.")
            if not temp_res.is_acceptable:
                reasons.append(f"Temporal delta ({temp_res.temporal_difference_days} days) exceeds threshold.")

            # Transparent internal ranking score (Section 7)
            # Factors: spatial overlap (45%), temporal proximity (25%), resolution match (15%), spectral match (15%)
            overlap_score = spat_res.overlap_pct / 100.0
            temp_score = 1.0 if temp_res.temporal_difference_days is None else max(0.0, 1.0 - (temp_res.temporal_difference_days / 60.0))
            spec_score = 1.0 if spec_res.compatibility == "FULL" else (0.7 if spec_res.compatibility == "RGB_ONLY" else 0.4)
            res_score = 1.0 if tile.resolution_m <= 2.5 else 0.8

            composite_score = (
                0.45 * overlap_score
                + 0.25 * temp_score
                + 0.15 * spec_score
                + 0.15 * res_score
            ) if is_eligible else 0.0

            evaluated.append(
                CandidateEvaluation(
                    source_name=tile.dataset,
                    tile_id=tile.id,
                    tile_name=tile.name,
                    spatial_result=spat_res,
                    temporal_result=temp_res,
                    spectral_result=spec_res,
                    resolution_m=tile.resolution_m,
                    is_eligible=is_eligible,
                    score=composite_score,
                    reasons=reasons,
                )
            )

        candidate_summaries = [
            {
                "tile_id": ev.tile_id,
                "dataset": ev.source_name,
                "overlap_pct": round(ev.spatial_result.overlap_pct, 1),
                "temporal_days": ev.temporal_result.temporal_difference_days,
                "temporal_status": ev.temporal_result.temporal_match_status,
                "spectral": ev.spectral_result.compatibility,
                "eligible": ev.is_eligible,
                "score": round(ev.score, 3),
            }
            for ev in evaluated
        ]

        eligible_candidates = [ev for ev in evaluated if ev.is_eligible]
        if not eligible_candidates:
            return ReferenceDiscoveryResult(
                available=False,
                match_status="NO_ELIGIBLE_REFERENCE",
                selection_reason="No compatible HR reference was found for this AOI.",
                limitations=[
                    "No registered reference tile satisfied spatial overlap, temporal proximity, and spectral criteria.",
                    "Gracefully falling back to no-reference scientific evaluation.",
                ],
                candidates_searched=len(evaluated),
                candidate_summary=candidate_summaries,
            )

        # Select highest composite score
        best = max(eligible_candidates, key=lambda ev: ev.score)
        tile_obj = next(t for t in candidates if t.id == best.tile_id)

        # Retrieve file through downloader / cache (Section 6)
        ref_path = self.downloader.retrieve_reference(
            uri_or_path=tile_obj.uri,
            source=tile_obj.dataset,
            reference_id=tile_obj.id,
            bbox=tile_obj.bbox,
            resolution_m=tile_obj.resolution_m,
        )

        # Section 6: Candidate is available ONLY if reference raster can be loaded
        file_exists = bool(ref_path and Path(ref_path).exists())
        if not file_exists:
            return ReferenceDiscoveryResult(
                available=False,
                source=tile_obj.dataset,
                reference_id=tile_obj.id,
                reference_path=None,
                resolution_m=tile_obj.resolution_m,
                spatial_overlap=round(best.spatial_result.overlap_pct, 2),
                temporal_difference_days=best.temporal_result.temporal_difference_days,
                temporal_match_status=best.temporal_result.temporal_match_status,
                spectral_compatibility=best.spectral_result.compatibility,
                match_status="REFERENCE_FILE_UNAVAILABLE",
                selection_reason=f"Reference candidate {tile_obj.id} was selected but local raster file could not be loaded.",
                limitations=[
                    f"Selected reference tile {tile_obj.id} could not be resolved from {tile_obj.uri}.",
                    "Falling back to no-reference scientific evaluation.",
                ],
                candidates_searched=len(evaluated),
                candidate_summary=candidate_summaries,
            )

        limitations = [
            f"External HR reference from {tile_obj.provider or tile_obj.dataset}.",
            f"Ground sampling distance: {tile_obj.resolution_m}m nominal resolution.",
        ]
        if best.temporal_result.temporal_match_status in ("ACCEPTABLE_WITH_CAVEAT", "UNKNOWN"):
            limitations.append(
                f"Temporal match status is {best.temporal_result.temporal_match_status} "
                f"({best.temporal_result.temporal_difference_days or 'unknown'} days difference). "
                "Vegetation phenology or illumination changes may impact exact radiometric comparison."
            )
        if best.spectral_result.compatibility == "RGB_ONLY":
            limitations.append("Reference imagery is RGB only. Spectral Angle Mapper (SAM) is withheld.")

        if best.temporal_result.temporal_difference_days == 0:
            temporal_phrase = "same acquisition date"
        elif best.temporal_result.temporal_difference_days is not None:
            temporal_phrase = f"{best.temporal_result.temporal_difference_days} days temporal separation"
        else:
            temporal_phrase = f"temporal proximity ({best.temporal_result.temporal_match_status})"

        spec_label = "FULL_VNIR" if best.spectral_result.compatibility == "FULL" else best.spectral_result.compatibility
        selection_reason = (
            f"Selected because it provides {spec_label} compatibility, "
            f"{best.spatial_result.overlap_pct:.0f}% AOI overlap, "
            f"{temporal_phrase}, and {tile_obj.resolution_m}m spatial resolution."
        )

        provenance = {
            "source": tile_obj.dataset,
            "tile_id": tile_obj.id,
            "tile_name": tile_obj.name,
            "sensor": tile_obj.sensor,
            "provider": tile_obj.provider,
            "provenance_statement": tile_obj.provenance,
            "reference_datetime": tile_obj.datetime or ref_dt,
            "cloud_cover_pct": tile_obj.cloud_cover_pct,
            "cached": True if ref_path and self.downloader.is_cached(
                self.downloader.generate_cache_key(tile_obj.dataset, tile_obj.id, tile_obj.bbox, tile_obj.resolution_m)
            ) else False,
        }

        return ReferenceDiscoveryResult(
            available=True,
            source=tile_obj.dataset,
            reference_id=tile_obj.id,
            reference_path=str(ref_path) if ref_path else None,
            resolution_m=tile_obj.resolution_m,
            spatial_overlap=round(best.spatial_result.overlap_pct, 2),
            temporal_difference_days=best.temporal_result.temporal_difference_days,
            temporal_match_status=best.temporal_result.temporal_match_status,
            spectral_compatibility=spec_label,
            match_status="ELIGIBLE",
            selection_reason=selection_reason,
            limitations=limitations,
            eligible_metrics=best.spectral_result.eligible_metrics,
            ineligible_metrics=best.spectral_result.ineligible_metrics,
            band_mapping=best.spectral_result.band_mapping,
            provenance=provenance,
            candidates_searched=len(evaluated),
            candidate_summary=candidate_summaries,
        )
