import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


BUCKET = "eodata"
ENDPOINT = "https://eodata.dataspace.copernicus.eu"

SCENES = {
    "train": (
        "Sentinel-2/MSI/L2A/2026/03/03/"
        "S2B_MSIL2A_20260303T050649_N0512_R019_T43PHM_20260303T101734.SAFE/"
    ),
    "validation": (
        "Sentinel-2/MSI/L2A_N0500/2023/04/13/"
        "S2A_MSIL2A_20230413T050651_N0510_R019_T43PGN_20240829T072229.SAFE/"
    ),
    "test": (
        "Sentinel-2/MSI/L2A/2025/12/15/"
        "S2C_MSIL2A_20251215T050231_N0511_R119_T44PKS_20251215T074508.SAFE/"
    ),
}

REQUIRED_FILES = (
    "_B02_10m.jp2",
    "_B03_10m.jp2",
    "_B04_10m.jp2",
    "_B08_10m.jp2",
    "_SCL_20m.jp2",
)

OUTPUT_ROOT = Path("dataset") / "plan2" / "sentinel2"


def make_client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=os.environ["CDSE_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["CDSE_S3_SECRET_KEY"],
        region_name="us-east-1",
    )


def find_required_files(s3, prefix):
    paginator = s3.get_paginator("list_objects_v2")

    matches = []

    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if key.endswith(REQUIRED_FILES):
                matches.append(key)

    return sorted(matches)


def download_scene(s3, split, prefix):
    print(f"\n=== {split.upper()} ===")
    print(f"Searching: {prefix}")

    keys = find_required_files(s3, prefix)

    if not keys:
        raise RuntimeError(f"No required files found for {split}")

    for key in keys:
        filename = Path(key).name
        output_dir = OUTPUT_ROOT / split
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / filename

        if output_path.exists():
            print(f"[SKIP] {filename}")
            continue

        print(f"[DOWNLOAD] {filename}")

        try:
            s3.download_file(BUCKET, key, str(output_path))
        except ClientError as exc:
            raise RuntimeError(
                f"Failed to download {key}"
            ) from exc

        print(f"[OK] {output_path}")


def main():
    if not os.environ.get("CDSE_S3_ACCESS_KEY"):
        raise RuntimeError("CDSE_S3_ACCESS_KEY is not set")

    if not os.environ.get("CDSE_S3_SECRET_KEY"):
        raise RuntimeError("CDSE_S3_SECRET_KEY is not set")

    s3 = make_client()

    print("PixelSight Sentinel-2 downloader")
    print(f"Output: {OUTPUT_ROOT.resolve()}")

    for split, prefix in SCENES.items():
        download_scene(s3, split, prefix)

    print("\nAll requested files downloaded successfully.")


if __name__ == "__main__":
    main()