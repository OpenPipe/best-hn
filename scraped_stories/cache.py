import os
from panza.cache import S3Cache

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
bucket = os.getenv("AWS_S3_BUCKET", "my-default-bucket")
prefix = "hacker-news-scraped-stories"
region = os.getenv("AWS_REGION", "us-west-2")
endpoint = os.getenv("AWS_S3_ENDPOINT_URL")

s3_cache = S3Cache(
    f"{bucket}/{prefix}",
    auto_create_bucket=True,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region,
    endpoint_url=endpoint,
)
