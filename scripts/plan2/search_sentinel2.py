import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]

STAC_URL = "https://stac.dataspace.copernicus.eu/v1/search"
COLLECTION = "sentinel-2-l2a"


# Approximate AOIs.
# These are deliberately small areas for scene discovery,
# not final tile boundaries.

AOIS = {
    "region1_madurai": {
        "bbox": [78.00, 9.85, 78.20, 10.05],
    },
    "region2_coimbatore": {
        "bbox": [76.85, 10.90, 77.10, 11.10],
    },
    "region3_thanjavur": {
        "bbox": [79.00, 10.65, 79.30, 10.90],
    },
}


def search_region(name, bbox):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    payload = {
        "collections": [COLLECTION],
        "bbox": bbox,
        "datetime": "2023-01-01T00:00:00Z/2026-12-31T23:59:59Z",
        "limit": 10,
        "query": {
            "eo:cloud_cover": {
                "lte": 10
            }
        },
        "sortby": [
            {
                "field": "properties.eo:cloud_cover",
                "direction": "asc"
            }
        ]
    }

    response = requests.post(
        STAC_URL,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    features = data.get("features", [])

    if not features:
        print("No scenes found.")
        return

    for i, feature in enumerate(features, start=1):

        properties = feature.get(
            "properties",
            {}
        )

        print()
        print(f"[{i}]")
        print("ID:", feature.get("id"))
        print(
            "Date:",
            properties.get("datetime")
        )
        print(
            "Cloud:",
            properties.get("eo:cloud_cover")
        )

        print(
            "Platform:",
            properties.get("platform")
        )

        print(
            "Tile:",
            properties.get("s2:mgrs_tile")
        )


def main():

    print("=" * 70)
    print("PixelSight Plan 2 — Sentinel-2 L2A Scene Search")
    print("=" * 70)

    print()
    print("Collection:", COLLECTION)
    print("Cloud threshold: <= 10%")
    print("Date range: 2023–2026")

    for name, info in AOIS.items():

        search_region(
            name,
            info["bbox"]
        )


if __name__ == "__main__":
    main()