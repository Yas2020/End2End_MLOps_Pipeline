import os
import json
import boto3
from pathlib import Path
from io import BytesIO
from utils.logging.logger import get_logger

# === Logging setup ===
logger = get_logger("model_training")


S3_BUCKET = "mlflow-artifacts"

# === MinIO Client ===
s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

def download_data(path="data/raw"):
    local_dir = Path(path)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    with open("version_meta.json", 'r') as f:
        meta = json.load(f)

    version_tag = meta["version_tag"]
    
    # Fraud data
    key = f"data/raw/fraud_data_{version_tag}.csv"
    buffer = BytesIO()
    s3.download_fileobj(S3_BUCKET, key, buffer)
    buffer.seek(0)
    with open(key, 'wb') as f:
        f.write(buffer.getvalue())
    
    # Metadata
    key = f"data/raw/metadata_{version_tag}.json"
    buffer = BytesIO()
    s3.download_fileobj(S3_BUCKET, key, buffer)
    buffer.seek(0)
    with open(key, 'wb') as f:
        json.load(buffer)

    logger.info(f"✅ Data saved locally in {local_dir}")

if __name__ == "__main__":
    download_data()


