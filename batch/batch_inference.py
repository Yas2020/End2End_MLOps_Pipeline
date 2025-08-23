import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime

# Load version metadata
with open("version_meta.json") as f:
    meta = json.load(f)

version = meta["version_tag"]
model_pipeline = joblib.load(meta["model_file"])
outlier_detector = joblib.load(meta["outlier_artifact"])

# Load batch input CSV
input_csv = f"data/batch/input_batch_{version}.csv"  # You decide where batch inputs go
df = pd.read_csv(input_csv)

# Prepare raw feature columns
raw_features = ["amount", "transaction_time", "transaction_type", "location_region"]
X_raw = df[raw_features]

# Preprocess
X_transformed = model_pipeline.named_steps["preprocessor"].transform(X_raw)

assert model_pipeline.named_steps["preprocessor"].n_features_in_ == X_raw.shape[1], \
    "Model pipeline expects raw features. Do not pass transformed input to it."


# Detect outliers
outlier_flags = outlier_detector.predict(X_transformed) == -1

# Predict
df["is_outlier"] = outlier_flags
df["is_fraud"] = None
df["probability"] = None

valid_idx = ~df["is_outlier"]
df.loc[valid_idx, "is_fraud"] = model_pipeline.predict(X_raw[valid_idx])
df.loc[valid_idx, "probability"] = model_pipeline.predict_proba(X_raw[valid_idx])[:, 1]

# Save output CSV
output_dir = Path("data/batch/output")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"batch_predictions_{version}.csv"
df.to_csv(output_file, index=False)

# Log metrics
summary = {
    "version_tag": version,
    "total_rows": len(df),
    "outliers": int(outlier_flags.sum()),
    "predicted_fraud": int(df["is_fraud"].sum(skipna=True)),
    "batch_processed_at": datetime.now().isoformat()
}

summary_file = output_dir / f"batch_metrics_{version}.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=4)

print(f"Batch inference done. Output: {output_file}")
print(f"Summary: {summary_file}")
