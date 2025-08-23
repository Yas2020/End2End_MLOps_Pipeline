import os
import joblib
import json
import pandas as pd
import boto3
from io import BytesIO
from typing import Any


# === MinIO Client ===
s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

def load_upload_minio(s3_uri: str, filetype: str, mode: str = "load", obj: Any = None) -> Any:
    """
    Load or upload a file to/from MinIO/S3.

    Args:
        s3_uri (str): S3 URI, e.g., "s3://bucket/path/file.json".
        filetype (str): "json" or "joblib".
        mode (str): "load" to download, "upload" to upload.
        obj (Any): Object to upload (required if mode="upload").

    Returns:
        Any: Loaded object if mode="load", otherwise None.
    """
    bucket = s3_uri.split("/")[2]
    key = "/".join(s3_uri.split("/")[3:])
    # BytesIO() creates a memory buffer — like a pretend file that lives in RAM without writing to Disk- the cleanest way to serialize Python objects
    buffer = BytesIO()

    if mode == "load":
        s3.download_fileobj(bucket, key, buffer)
        buffer.seek(0)
        if filetype == "json":
            return json.load(buffer)
        elif filetype == "joblib":
            return joblib.load(buffer)
        elif filetype == "csv":
            return pd.read_csv(buffer)
        elif filetype == "bytes":
            return buffer.getvalue()
        else:
            raise ValueError(f"Unsupported filetype: {filetype}")

    elif mode == "upload":
        if obj is None:
            raise ValueError("Object to upload must be provided when mode='upload'.")
        if filetype == "json":
            buffer.write(json.dumps(obj).encode("utf-8"))
        elif filetype == "joblib":
            joblib.dump(obj, buffer)
        elif filetype == "pandas":
            obj.to_csv(buffer)
        elif filetype == "bytes":
            buffer.getvalue()
        else:
            raise ValueError(f"Unsupported filetype: {filetype}")
        buffer.seek(0)
        s3.upload_fileobj(buffer, bucket, key)
    else:
        raise ValueError("mode must be either 'load' or 'upload'.")
