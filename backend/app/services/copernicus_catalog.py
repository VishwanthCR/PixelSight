import datetime
from typing import Any, Dict, List, Optional
import requests

from backend.app.config import COPERNICUS_CATALOG_URL
from backend.app.services.aoi import parse_bbox_from_geometry
from backend.app.services.copernicus_auth import copernicus_auth, CopernicusAuthError


class CopernicusCatalogError(Exception):
    """Raised when Copernicus Catalog / STAC search fails."""
    pass


class CopernicusCatalogService:
    def __init__(self, catalog_url: str = COPERNICUS_CATALOG_URL):
        self.catalog_url = catalog_url

    def search_scenes(
        self,
        aoi: Any,
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 20.0,
        collection: str = "sentinel-2-l2a",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Searches Copernicus Sentinel Hub Catalog for Sentinel-2 L2A scenes matching AOI and date range.
        Returns candidate scenes sorted with lowest cloud cover first or latest date.
        """
        token = copernicus_auth.get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/geo+json, application/json, */*",
        }

        bbox = parse_bbox_from_geometry(aoi)

        # Format dates properly as ISO 8601 interval
        s_date = start_date.strip()
        e_date = end_date.strip()
        if "T" not in s_date:
            s_date = f"{s_date}T00:00:00Z"
        if "T" not in e_date:
            e_date = f"{e_date}T23:59:59Z"

        payload: Dict[str, Any] = {
            "collections": [collection],
            "datetime": f"{s_date}/{e_date}",
            "bbox": bbox,
            "limit": max(1, min(limit, 50)),
        }

        try:
            response = requests.post(
                self.catalog_url,
                json=payload,
                headers=headers,
                timeout=25,
            )
        except requests.RequestException as exc:
            raise CopernicusCatalogError(f"Network error querying Copernicus Catalog: {exc}") from exc

        if response.status_code != 200:
            raise CopernicusCatalogError(
                f"Copernicus Catalog search failed with status {response.status_code}: {response.text[:200]}"
            )

        data = response.json()
        features = data.get("features", [])

        candidate_scenes: List[Dict[str, Any]] = []
        for feat in features:
            props = feat.get("properties", {})
            cloud_val = props.get("eo:cloud_cover", props.get("cloudCover", 0.0))
            try:
                cloud_pct = round(float(cloud_val), 2)
            except (ValueError, TypeError):
                cloud_pct = 0.0

            if max_cloud_cover is not None and cloud_pct > max_cloud_cover:
                continue

            scene_id = feat.get("id", "")
            dt = props.get("datetime", "")

            candidate_scenes.append({
                "id": scene_id,
                "datetime": dt,
                "cloud_cover": cloud_pct,
                "collection": "Sentinel-2 L2A",
                "bbox": feat.get("bbox", bbox),
                "geometry": feat.get("geometry"),
                "platform": props.get("platform", "Sentinel-2"),
            })

        # Sort candidate scenes by acquisition date descending
        candidate_scenes.sort(key=lambda s: s.get("datetime", ""), reverse=True)
        return candidate_scenes

    def select_best_scene(
        self,
        scenes: List[Dict[str, Any]],
        criterion: str = "lowest_cloud_cover",
    ) -> Optional[Dict[str, Any]]:
        """
        Explicitly chooses the best scene based on clear defined criteria:
        - 'lowest_cloud_cover': Minimum cloud cover %, tie-breaker by most recent acquisition date.
        - 'latest_date': Most recent acquisition date with cloud cover <= max.
        """
        if not scenes:
            return None

        if criterion == "latest_date":
            best = sorted(scenes, key=lambda s: s.get("datetime", ""), reverse=True)[0]
            best["selection_criterion"] = "Most recent acquisition date"
            return best

        # Default: lowest cloud cover
        best = sorted(scenes, key=lambda s: (s.get("cloud_cover", 100.0), s.get("datetime", "")))[0]
        best["selection_criterion"] = f"Lowest cloud coverage ({best.get('cloud_cover')}%)"
        return best


copernicus_catalog = CopernicusCatalogService()
