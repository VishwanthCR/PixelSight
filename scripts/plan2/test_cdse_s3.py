import os
import boto3

ACCESS_KEY = os.environ["CDSE_S3_ACCESS_KEY"]
SECRET_KEY = os.environ["CDSE_S3_SECRET_KEY"]

s3 = boto3.client(
    "s3",
    endpoint_url="https://eodata.dataspace.copernicus.eu",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name="us-east-1",
)

print("Testing CDSE S3 connection...")

response = s3.list_objects_v2(
    Bucket="eodata",
    MaxKeys=5,
)

print("SUCCESS!")
print(f"Objects returned: {len(response.get('Contents', []))}")

for obj in response.get("Contents", []):
    print(obj["Key"])