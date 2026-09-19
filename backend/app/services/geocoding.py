import logging
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request
import json

from backend.app.services.aoi import compute_bbox_for_center, is_aoi_inside_india

logger = logging.getLogger("pixelsight.geocoding")

# In-memory cache for geocoded queries
_GEOCODE_CACHE: Dict[str, List[Dict[str, Any]]] = {}

# Built-in fallback database for key Indian locations & test places
_FALLBACK_PLACES: Dict[str, Dict[str, Any]] = {
    "chennai": {
        "place_id": "fallback_chennai",
        "display_name": "Chennai, Tamil Nadu, India",
        "lat": 13.0827,
        "lon": 80.2707,
        "type": "city",
        "boundingbox": [12.90, 13.25, 80.10, 80.35],
        "inside_india": True,
    },
    "madurai": {
        "place_id": "fallback_madurai",
        "display_name": "Madurai, Tamil Nadu, India",
        "lat": 9.9252,
        "lon": 78.1198,
        "type": "city",
        "boundingbox": [9.80, 10.05, 78.00, 78.25],
        "inside_india": True,
    },
    "coimbatore": {
        "place_id": "fallback_coimbatore",
        "display_name": "Coimbatore, Tamil Nadu, India",
        "lat": 11.0168,
        "lon": 76.9558,
        "type": "city",
        "boundingbox": [10.90, 11.15, 76.85, 77.08],
        "inside_india": True,
    },
    "bengaluru": {
        "place_id": "fallback_bengaluru",
        "display_name": "Bengaluru, Karnataka, India",
        "lat": 12.9716,
        "lon": 77.5946,
        "type": "city",
        "boundingbox": [12.80, 13.15, 77.45, 77.75],
        "inside_india": True,
    },
    "hyderabad": {
        "place_id": "fallback_hyderabad",
        "display_name": "Hyderabad, Telangana, India",
        "lat": 17.3850,
        "lon": 78.4867,
        "type": "city",
        "boundingbox": [17.20, 17.60, 78.25, 78.65],
        "inside_india": True,
    },
    "mumbai": {
        "place_id": "fallback_mumbai",
        "display_name": "Mumbai, Maharashtra, India",
        "lat": 19.0760,
        "lon": 72.8777,
        "type": "city",
        "boundingbox": [18.88, 19.28, 72.75, 73.00],
        "inside_india": True,
    },
    "delhi": {
        "place_id": "fallback_delhi",
        "display_name": "New Delhi, Delhi, India",
        "lat": 28.6139,
        "lon": 77.2090,
        "type": "city",
        "boundingbox": [28.40, 28.88, 77.00, 77.40],
        "inside_india": True,
    },
    "kolkata": {
        "place_id": "fallback_kolkata",
        "display_name": "Kolkata, West Bengal, India",
        "lat": 22.5726,
        "lon": 88.3639,
        "type": "city",
        "boundingbox": [22.40, 22.75, 88.20, 88.50],
        "inside_india": True,
    },
    "kerala": {
        "place_id": "fallback_kerala",
        "display_name": "Kerala, India",
        "lat": 10.8505,
        "lon": 76.2711,
        "type": "state",
        "boundingbox": [8.18, 12.80, 74.85, 77.45],
        "inside_india": True,
    },
    "tamil nadu": {
        "place_id": "fallback_tamil_nadu",
        "display_name": "Tamil Nadu, India",
        "lat": 11.1271,
        "lon": 78.6569,
        "type": "state",
        "boundingbox": [8.08, 13.58, 76.24, 80.35],
        "inside_india": True,
    },
    "anna nagar": {
        "place_id": "fallback_anna_nagar",
        "display_name": "Anna Nagar, Chennai, Tamil Nadu, India",
        "lat": 13.0850,
        "lon": 80.2101,
        "type": "suburb",
        "boundingbox": [13.07, 13.10, 80.19, 80.23],
        "inside_india": True,
    },
    "marina beach": {
        "place_id": "fallback_marina_beach",
        "display_name": "Marina Beach, Chennai, Tamil Nadu, India",
        "lat": 13.0499,
        "lon": 80.2824,
        "type": "beach",
        "boundingbox": [13.03, 13.07, 80.27, 80.29],
        "inside_india": True,
    },
    "new york": {
        "place_id": "fallback_ny",
        "display_name": "New York, United States",
        "lat": 40.7128,
        "lon": -74.0060,
        "type": "city",
        "boundingbox": [40.47, 40.91, -74.25, -73.70],
        "inside_india": False,
    },
    "rome": {
        "place_id": "fallback_rome",
        "display_name": "Rome, Lazio, Italy",
        "lat": 41.9028,
        "lon": 12.4964,
        "type": "city",
        "boundingbox": [41.77, 42.02, 12.35, 12.65],
        "inside_india": False,
    },
}


def _query_nominatim(query_str: str, limit: int, countrycodes: Optional[str] = None) -> List[Dict[str, Any]]:
    encoded = urllib.parse.quote(query_str)
    url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit={limit}&addressdetails=1"
    if countrycodes:
        url += f"&countrycodes={countrycodes}"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "PixelSight-Geocoding-Agent/1.0 (satellite-intelligence; research@pixelsight.ai)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=3.5) as response:
        if response.status == 200:
            return json.loads(response.read().decode("utf-8"))
    return []


def geocode_location(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Geocodes a place name with priority and validation for India locations.
    Returns results with `inside_india` verification flags.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    norm_key = clean_query.lower()
    if norm_key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[norm_key]

    results: List[Dict[str, Any]] = []

    # 1. Attempt live query to Nominatim scoped to India first (&countrycodes=in)
    raw_data = []
    try:
        raw_data = _query_nominatim(clean_query, limit, countrycodes="in")
        # If nothing found in India, query globally so we can accurately detect out-of-India searches
        if not raw_data:
            raw_data = _query_nominatim(clean_query, limit, countrycodes=None)
    except Exception as exc:
        logger.warning(f"Live Nominatim geocoding lookup failed for '{clean_query}': {exc}")

    if raw_data:
        for item in raw_data:
            lat = float(item["lat"])
            lon = float(item["lon"])
            bbox = [float(x) for x in item.get("boundingbox", [lat - 0.05, lat + 0.05, lon - 0.05, lon + 0.05])]
            default_aoi = compute_bbox_for_center(lat, lon, width_km=1.28, height_km=1.28)
            is_valid_india, reason = is_aoi_inside_india(default_aoi)

            results.append({
                "place_id": str(item.get("place_id", f"{lat}_{lon}")),
                "display_name": item.get("display_name", clean_query),
                "lat": lat,
                "lon": lon,
                "type": item.get("type", "location"),
                "boundingbox": bbox,
                "default_aoi_1_28km": default_aoi,
                "inside_india": is_valid_india,
                "scope_message": "Within supported India region." if is_valid_india else "PixelSight currently supports processing only within India.",
            })

    # 2. Check fallback database if live results are empty or failed
    if not results:
        for key, place in _FALLBACK_PLACES.items():
            if key in norm_key or norm_key in key:
                lat = place["lat"]
                lon = place["lon"]
                default_aoi = compute_bbox_for_center(lat, lon, width_km=1.28, height_km=1.28)
                is_valid = place.get("inside_india", True)
                results.append({
                    **place,
                    "default_aoi_1_28km": default_aoi,
                    "inside_india": is_valid,
                    "scope_message": "Within supported India region." if is_valid else "PixelSight currently supports processing only within India.",
                })

    if results:
        _GEOCODE_CACHE[norm_key] = results

    return results
