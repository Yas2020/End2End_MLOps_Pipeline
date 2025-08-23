import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from utils.minio.load_upload_minio import load_upload_minio
from utils.logging.logger import get_logger

# === Logging setup ===
logger = get_logger("model_training")

# === Get S3 Client ===
S3_BUCKET = "mlflow-artifacts"

# Set version
def generate_version_tag():
    return datetime.now().strftime("v%Y%m%d_%H%M%S")

# Simulate fraud data
def generate_fraud_data(n_samples=1000, fraud_ratio=0.03, seed=42):
    np.random.seed(seed)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legit transactions
    legit = pd.DataFrame({
        "transaction_id": np.arange(n_legit),
        "amount": np.random.exponential(scale=50, size=n_legit),
        "transaction_time": np.random.normal(loc=10000, scale=2000, size=n_legit),
        "transaction_type": np.random.choice(["online", "in-person", "phone"], n_legit),
        "location_region": np.random.choice(["US-West", "US-East", "EU", "Asia"], n_legit),
        "is_fraud": 0
    })

    # Fraud transactions (outliers)
    fraud = pd.DataFrame({
        "transaction_id": np.arange(n_legit, n_samples),
        "amount": np.random.exponential(scale=300, size=n_fraud),  # Higher amounts
        "transaction_time": np.random.normal(loc=20000, scale=5000, size=n_fraud),
        "transaction_type": np.random.choice(["online", "phone"], n_fraud),
        "location_region": np.random.choice(["US-West", "EU"], n_fraud),
        "is_fraud": 1
    })

    data = pd.concat([legit, fraud], ignore_index=True)
    data = data.sample(frac=1).reset_index(drop=True)  # Shuffle
    return data

# Save CSV + metadata
def upload_to_s3(data, version_tag):
    s3_uri = f"s3://{S3_BUCKET}/data/raw/fraud_data_{version_tag}.csv"
    # raw_data_path = save_raw_data_to_minio(s3_uri, data)
    load_upload_minio(s3_uri, filetype="pandas", mode="upload", obj=data)
    logger.info(f"Simulated data saved at {s3_uri}.")
    
    # Create a root metadata for full pipeline etl -> preprocessing -> training cycle - used by downstream processors
    meta = {
        "version_tag": version_tag,
        "raw_data_remote": str(s3_uri),
        "etl_timestamp": datetime.now().isoformat(),
    }

    # Save version_meta.json locally
    with open("version_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"✅ Saved version_meta.json locally.")
    
    # Save data metadata 
    s3_uri = f"s3://{S3_BUCKET}/data/raw/metadata_{version_tag}.json"
    metadata = {
        "version": version_tag,
        "rows": len(data),
        "fraud_ratio": data["is_fraud"].mean(),
        "generated_on": datetime.now().isoformat()
    }

    load_upload_minio(s3_uri, filetype="json", mode="upload", obj=metadata)
    logger.info(f"Meta Version saved to: {s3_uri}")


# Main ETL Task
def main():
    aws_s3_endpoint = os.getenv("AWS_ENDPOINT_URL")
    logger.info(f"AWS ENDPOINT is: {aws_s3_endpoint}")
    version = generate_version_tag()
    data = generate_fraud_data(n_samples=1000, fraud_ratio=0.03)
    logger.info("Simulated data generated.")
    upload_to_s3(data, version)
    

if __name__ == "__main__":
    main()
