import os

import pytest


ACCESS_KEY = os.getenv("CDSE_S3_ACCESS_KEY")
SECRET_KEY = os.getenv("CDSE_S3_SECRET_KEY")

if not ACCESS_KEY or not SECRET_KEY:
    pytest.skip(
        "CDSE S3 credentials not configured; skipping CDSE integration test.",
        allow_module_level=True,
    )

import boto3


s3 = boto3.client(
    "s3",
    endpoint_url="https://eodata.dataspace.copernicus.eu",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name="us-east-1",
)


def test_cdse_s3_connection():
    print("Testing CDSE S3 connection...")

    response = s3.list_objects_v2(
        Bucket="eodata",
        MaxKeys=5,
    )

    print("SUCCESS!")
    print(f"Objects returned: {len(response.get('Contents', []))}")

    for obj in response.get("Contents", []):
        print(obj["Key"])