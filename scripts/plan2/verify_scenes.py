import requests


STAC_URL = "https://stac.dataspace.copernicus.eu/v1"

SCENES = {
    "train": "S2B_MSIL2A_20260303T050649_N0512_R019_T43PHM_20260303T101734",
    "validation": "S2A_MSIL2A_20230413T050651_N0510_R019_T43PGN_20240829T072229",
    "test": "S2C_MSIL2A_20251215T050231_N0511_R119_T44PKS_20251215T074508",
}


def main():

    for split, scene_id in SCENES.items():

        print("=" * 70)
        print(split.upper())
        print("=" * 70)

        url = (
            f"{STAC_URL}/collections/"
            f"sentinel-2-l2a/items/{scene_id}"
        )

        response = requests.get(
            url,
            timeout=60
        )

        print("HTTP:", response.status_code)

        response.raise_for_status()

        item = response.json()

        properties = item.get("properties", {})

        print("ID:", item.get("id"))
        print("Datetime:", properties.get("datetime"))
        print("Cloud:", properties.get("eo:cloud_cover"))
        print("Platform:", properties.get("platform"))
        print("MGRS tile:", properties.get("s2:mgrs_tile"))

        print()
        print("Assets:")

        for name, asset in item.get("assets", {}).items():

            print(
                f"  {name}: "
                f"{asset.get('title', '')}"
            )

            print(
                f"       {asset.get('href', '')}"
            )

        print()


if __name__ == "__main__":
    main()