"""
PixelSight Reference Source Registry
====================================
Loads, parses, and provides access to external HR reference source configurations
and their corresponding geographic index catalogs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ReferenceTile:
    """Individual tile/product in a reference dataset index."""
    id: str
    name: str
    dataset: str
    bbox: list[float]  # [west, south, east, north] in EPSG:4326
    geometry: dict[str, Any]  # GeoJSON geometry
    datetime: str | None
    resolution_m: float
    scale_factor: int = 4
    bands: list[str] = field(default_factory=lambda: ["B02", "B03", "B04", "B08"])
    cloud_cover_pct: float = 0.0
    sensor: str = ""
    provider: str = ""
    provenance: str = ""
    uri: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceSource:
    """Metadata and catalog for a configured reference dataset."""
    name: str
    type: str  # "external_hr_reference" | "benchmark"
    description: str
    resolution_m: float
    scale_factor: int
    enabled: bool
    crs: str
    bands: list[str]
    geographic_index_path: Path
    requires_alignment: bool
    provenance: str
    tiles: list[ReferenceTile] = field(default_factory=list)


class ReferenceRegistry:
    """Manages reference dataset configurations and index lookup."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.root_dir = Path(__file__).resolve().parents[4]  # repo root
        self.config_path = (
            Path(config_path)
            if config_path
            else self.root_dir / "configs" / "evaluation" / "reference_sources.yaml"
        )
        self.cache_dir = self.root_dir / "results" / "cache" / "reference"
        self.min_spatial_overlap_pct = 50.0
        self.max_acceptable_resolution_m = 5.0
        self.target_scale_factor = 4
        self.temporal_thresholds_days = {
            "exact": 0,
            "close": 7,
            "acceptable_with_caveat": 30,
        }
        self.sources: dict[str, ReferenceSource] = {}
        self.load()

    def load(self) -> None:
        """Parse yaml configuration and load index files."""
        if not self.config_path.exists():
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            return

        settings = cfg.get("settings", {})
        if "cache_dir" in settings:
            self.cache_dir = self.root_dir / settings["cache_dir"]
        self.min_spatial_overlap_pct = float(settings.get("min_spatial_overlap_pct", 50.0))
        self.max_acceptable_resolution_m = float(settings.get("max_acceptable_resolution_m", 5.0))
        self.target_scale_factor = int(settings.get("target_scale_factor", 4))
        self.temporal_thresholds_days = settings.get(
            "temporal_thresholds_days", self.temporal_thresholds_days
        )

        for src_cfg in cfg.get("sources", []):
            name = src_cfg.get("name")
            if not name or not src_cfg.get("enabled", True):
                continue

            index_rel = src_cfg.get("geographic_index", "")
            index_path = self.root_dir / index_rel if index_rel else Path()
            tiles = self._load_index_tiles(index_path, name)

            self.sources[name] = ReferenceSource(
                name=name,
                type=src_cfg.get("type", "external_hr_reference"),
                description=src_cfg.get("description", ""),
                resolution_m=float(src_cfg.get("resolution_m", 2.5)),
                scale_factor=int(src_cfg.get("scale_factor", 4)),
                enabled=bool(src_cfg.get("enabled", True)),
                crs=src_cfg.get("crs", "EPSG:4326"),
                bands=list(src_cfg.get("bands", ["B02", "B03", "B04", "B08"])),
                geographic_index_path=index_path,
                requires_alignment=bool(src_cfg.get("requires_alignment", True)),
                provenance=src_cfg.get("provenance", ""),
                tiles=tiles,
            )

    def _load_index_tiles(self, index_path: Path, dataset_name: str) -> list[ReferenceTile]:
        """Load tiles from an index JSON file."""
        if not index_path.exists():
            return []
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []

        tiles: list[ReferenceTile] = []
        raw_tiles = data.get("tiles", [])
        for item in raw_tiles:
            tile_id = item.get("id")
            if not tile_id:
                continue
            tiles.append(
                ReferenceTile(
                    id=tile_id,
                    name=item.get("name", tile_id),
                    dataset=dataset_name,
                    bbox=item.get("bbox", [0.0, 0.0, 0.0, 0.0]),
                    geometry=item.get("geometry", {}),
                    datetime=item.get("datetime"),
                    resolution_m=float(item.get("resolution_m", 2.5)),
                    scale_factor=int(item.get("scale_factor", 4)),
                    bands=list(item.get("bands", ["B02", "B03", "B04", "B08"])),
                    cloud_cover_pct=float(item.get("cloud_cover_pct", 0.0)),
                    sensor=item.get("sensor", ""),
                    provider=item.get("provider", ""),
                    provenance=item.get("provenance", ""),
                    uri=item.get("uri", ""),
                    metadata=item.get("metadata", {}),
                )
            )
        return tiles

    def get_source(self, name: str) -> ReferenceSource | None:
        return self.sources.get(name)

    def list_sources(self) -> list[ReferenceSource]:
        return list(self.sources.values())

    def get_all_tiles(self) -> list[ReferenceTile]:
        all_tiles: list[ReferenceTile] = []
        for src in self.sources.values():
            all_tiles.extend(src.tiles)
        return all_tiles

    def register_custom_source(self, source: ReferenceSource) -> None:
        """Allow registering sources programmatically for testing/plugins."""
        self.sources[source.name] = source
