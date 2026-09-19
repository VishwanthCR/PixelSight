"""
PixelSight Application Registry
==============================
Configuration-driven discovery and metadata registry for PixelSight applications:
- research: Core 4x super-resolution & uncertainty
- crop: Crop health and NDVI monitoring
- urban: Urban land-cover segmentation and built-up analysis
- disaster: Pre/post-event temporal change detection and affected area mapping
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs" / "applications"


class ApplicationRegistry:
    def __init__(self, configs_dir: Path = CONFIGS_DIR) -> None:
        self.configs_dir = configs_dir
        self._cache: dict[str, dict[str, Any]] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.configs_dir.exists():
            return
        for config_path in self.configs_dir.glob("*.yaml"):
            try:
                content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                if isinstance(content, dict) and "application" in content:
                    self._cache[content["application"]] = content
            except Exception as exc:
                print(f"[Warning] Failed loading application config {config_path.name}: {exc}")

    def list_applications(self) -> list[dict[str, Any]]:
        """Return list of public application definitions for API/frontend."""
        apps = []
        ordered_keys = ["research", "crop", "urban", "disaster"]
        # Return in canonical order
        for key in ordered_keys:
            if key in self._cache:
                data = self._cache[key]
                apps.append({
                    "id": key,
                    "name": data.get("name", key.capitalize()),
                    "description": data.get("description", ""),
                    "scientific_status": data.get("scientific_status", "Research"),
                    "bands": data.get("bands", {}),
                    "mode": data.get("mode", "single_image"),
                })
        # Any other configured apps
        for key, data in self._cache.items():
            if key not in ordered_keys:
                apps.append({
                    "id": key,
                    "name": data.get("name", key.capitalize()),
                    "description": data.get("description", ""),
                    "scientific_status": data.get("scientific_status", "Research"),
                    "bands": data.get("bands", {}),
                    "mode": data.get("mode", "single_image"),
                })
        return apps

    def get_application(self, app_id: str) -> dict[str, Any] | None:
        return self._cache.get(app_id)

    def is_valid_application(self, app_id: str) -> bool:
        return app_id in self._cache

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "version": "1.0.0",
            "model": "LDSR-S2",
            "scale": 4,
            "sampling_steps": 100,
            "tile_size": 128,
            "supported_bands": ["B02", "B03", "B04", "B08"],
            "applications": self.list_applications(),
        }


# Singleton instance
registry = ApplicationRegistry()
