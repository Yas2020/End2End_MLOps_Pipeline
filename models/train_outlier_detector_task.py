import os
import json
import joblib
import mlflow
import pandas as pd
import numpy as np
import mlflow.sklearn
from datetime import datetime
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.ensemble import IsolationForest
from utils.logging.logger import get_logger

# === Logging setup ===
logger = get_logger("model_training")

# === Constants ===
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "OutlierDetectionModel"
MODEL_NAME = "OutlierDetectionModel"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# === Load local version metadata (origin of pipeline run) ===
with open("version_meta.json", "r") as f:
    meta = json.load(f)
version_tag = meta["version_tag"]
preprocessor_local = meta["preprocessor_local"]
preprocess_metadata_local = meta["preprocess_metadata_local"]
raw_data_local = meta["raw_data_local"]

with open(preprocess_metadata_local, "r") as f:
    preprocess_meta = json.load(f)
numeric_features = preprocess_meta["numeric_features"]
categorical_features = preprocess_meta["categorical_features"]

def train_register_outlier_detector():

    df = pd.read_csv(raw_data_local)
    
    # Only use normal transactions
    df = df[df["is_fraud"] == 0]
    raw_features = numeric_features + categorical_features
    X = df[raw_features]

    # ---- Preprocess / Transform ----
    preprocessor = joblib.load(preprocessor_local)
    X_transformed = preprocessor.transform(X)

    # === Fit Outlier Detector ===
    outlier_detector = IsolationForest(random_state=42)
    outlier_detector.fit(X_transformed)

    # Set anomaly threshold - 99th percentile of normals
    scores = outlier_detector.score_samples(X_transformed)
    anomaly_threshold = np.quantile(scores, 0.99) 
    logger.info(f"Anomaly threshold is set to be {anomaly_threshold} according to the model.")

    outlier_metadata = {
        "version_tag": version_tag,
        "model": "Sklearn-IsolationForest",
        "preprocessor_local": preprocessor_local,
        "raw_data_local": raw_data_local,
        "anomaly_threshold": anomaly_threshold,
        "training_timestamp": datetime.now().isoformat()
    }
    outlier_local = "artifacts/models/outlier"
    os.makedirs(outlier_local, exist_ok=True)
    with open(f"{outlier_local}/outlier_metadata_{version_tag}.json", "w") as f:
        json.dump(outlier_metadata, f, indent=2)

    outlier_detector_local = f"{outlier_local}/outlier_detector_{version_tag}.pkl"
    joblib.dump(outlier_detector, outlier_detector_local)

    run_name = f"train_{version_tag}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run id: {run_id}")

        # ---- Tags & Params ----
        mlflow.set_tag("version_tag", version_tag)
        mlflow.set_tag("raw_data_remote", meta["raw_data_remote"])
        mlflow.set_tag("preprocess_metadata", preprocess_meta)

        # params: features (as comma-separated strings), input format
        params = {
            "version_tag": version_tag,
            "anomaly_threshold": anomaly_threshold, 
            "numeric_features": ",".join(preprocess_meta["numeric_features"]),
            "categorical_features": ",".join(preprocess_meta["categorical_features"]),
            "input_format": "pandas.DataFrame",
            "training_script": "train/train_outlier_detector.py",
        }
        mlflow.log_param(params)
       
        mlflow.log_dict({"anomaly_threshold": anomaly_threshold}, "config.json")
        
        signature = infer_signature(X_transformed, outlier_detector.predict(X_transformed))
        mlflow.sklearn.log_model(sk_model=outlier_detector, 
                                name="OutlierDetector",
                                registered_model_name=MODEL_NAME,
                                signature=signature,
                                input_example=X_transformed[:5])
        
        logger.info("Outlier detector and its artifact logged & registered.")

        # Transition the newly-registered version to Production
        client = MlflowClient()
        model_versions = client.search_model_versions(f"name= '{MODEL_NAME}' and run_id= '{run_id}'")
        latest_model = max(model_versions, key=lambda mv: int(mv.version))
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest_model.version,
            stage="Production",
            archive_existing_versions=True
        )
    
    meta.update({
        "outlier_detector_local": outlier_detector_local,
        "outlier_uri": f"runs:/{run_id}/model"
    })
    with open("version_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Model and its artifacts registered and transitioned to 'Production' (version {latest_model.version})")
    logger.info("✅ Training & registration complete.")
    
if __name__ == "__main__":
    train_register_outlier_detector()