import os
import json
import joblib
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils.logging.logger import get_logger
# from utils.minio.load_upload_minio import load_upload_minio

# === Logging setup ===
logger = get_logger("preprocessing_pipeline")

with open("version_meta.json", 'r') as f:
    meta = json.load(f)
logger.info("version_meta.json loaded from local path!")

version_tag = meta["version_tag"]
raw_data_remote = meta["raw_data_remote"]

# === Download Data ===
df = pd.read_csv(f"data/raw/fraud_data_{version_tag}.csv")

# === 2. Define features ===
numeric_features = ["amount", "transaction_time"]
categorical_features = ["transaction_type", "location_region"]

# === 3. Build and fit transformer ===
numeric_pipeline = Pipeline([("scaler", StandardScaler())])
categorical_pipeline = Pipeline([("encoder", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])
preprocessor.fit(df[numeric_features + categorical_features])
logger.info("preprocessor trained!")


os.makedirs("artifacts/preprocess", exist_ok=True)
pipeline_path = f"artifacts/preprocess/fraud_pipeline_{version_tag}.pkl"
joblib.dump(preprocessor, pipeline_path)
logger.info(f"preprocessor saved to local path {pipeline_path}!")


# === 5. Save Metadata ===
preprocess_metadata = {
    "version_tag": version_tag,
    "numeric_features": list(numeric_features),
    "categorical_features": list(categorical_features),
    "raw_data_remote": raw_data_remote,
    "preprocessing_date": datetime.now().isoformat()
}

# Save metadata JSON
metadata_path = f"artifacts/preprocess/preprocess_metadata_{version_tag}.json"
with open(metadata_path, 'w') as f:
    json.dump(preprocess_metadata, f, indent=2)


meta["raw_data_local"]= f"data/raw/fraud_data_{version_tag}.csv"
meta["raw_data_remote"]= raw_data_remote
meta["preprocessor_local"] = pipeline_path
meta["preprocess_metadata_local"] = metadata_path
meta["preprocess_timestamp"] = preprocess_metadata["preprocessing_date"]

# load_upload_minio(version_meta_s3_url, filetype="json", mode="upload", obj=meta)
with open("version_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

logger.info(f"✅ Saved pipeline at: {pipeline_path}")
logger.info(f"📄 Saved metadata: {metadata_path}")
logger.info(f"📦 Updated version_meta.json")
