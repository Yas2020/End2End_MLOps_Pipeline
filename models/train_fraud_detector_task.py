#!/usr/bin/env python3
import os
import json
import math
import subprocess
import joblib
import mlflow
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, f1_score,
    precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient


# from preprocess import preprocess_data
from utils.logging.logger import get_logger

# === Logging setup ===
logger = get_logger("model_training")

# === Constants ===
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "FraudDetectionModel")
MODEL_NAME = os.getenv("MODEL_NAME", "FraudDetectionModel")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def safe_isnan(x):
    return isinstance(x, float) and math.isnan(x)

def train_model(X_transformed, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_transformed, y)
    return model

def evaluate_model(model, X_transformed, y):
    # Use probabilities for ROC/AUC where possible, fallback to predictions    
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_transformed)[:, 1]
    else:
        # If no predict_proba, use decision_function or predictions
        if hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_transformed)
        else:
            y_scores = model.predict(X_transformed)
    fpr, tpr, _ = roc_curve(y, y_scores)
    auc_val = auc(fpr, tpr)
    
    y_pred = model.predict(X_transformed)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_score": float(f1_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "auc": float(auc_val)
    }
    return metrics

def save_model_pipeline(preprocessor, model, version_tag):
    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    model_dir = Path("artifacts/models/fraud_detector")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"fraud_dtector_{version_tag}.pkl"
    joblib.dump(final_pipeline, model_file)
    return final_pipeline, str(model_file)

def main():
    # === Load local version metadata (origin of pipeline run) ===
    with open("version_meta.json", "r") as f:
        meta = json.load(f)

    version_tag = meta["version_tag"]
    preprocessor_local = meta["preprocessor_local"]
    preprocess_metadata_local = meta["preprocess_metadata_local"]
    raw_data_local = meta["raw_data_local"]

    with open(preprocess_metadata_local, "r") as f:
        preprocess_meta = json.load(f)

    # Start an MLflow run and log metadata, artifacts, params & metrics during the run
    run_name = f"train_{version_tag}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run id: {run_id}")

        # === Tags & Params ===
        mlflow.set_tag("version_tag", version_tag)
        mlflow.log_param("numeric_features", ",".join(preprocess_meta["numeric_features"]))
        mlflow.log_param("categorical_features", ",".join(preprocess_meta["categorical_features"]))
        mlflow.log_param("input_format", "pandas.DataFrame")
        mlflow.log_param("training_script", "train/train_task.py")
        mlflow.log_param("model_type", "logreg")
        
        # Find and log the git commit hash for the dataset version by which model was trained
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        mlflow.log_param("dataset_git_commit", git_commit)

        # === Load data ===
        logger.info(f"Loading preprocessor from {preprocessor_local} to process raw data")

        preprocessor = joblib.load(preprocessor_local)        
        numeric_features = preprocess_meta["numeric_features"]
        categorical_features = preprocess_meta["categorical_features"]
        df = pd.read_csv(raw_data_local)
        raw_features = numeric_features + categorical_features
        X = df[raw_features]
        y = df["is_fraud"]

        # ---- Preprocess / Transform ----
        X_transformed = preprocessor.transform(X)
        logger.info("Preprocessing/transform applied to raw data.")

        # === Train model ===
        model = train_model(X_transformed, y)
        metrics = evaluate_model(model, X_transformed, y)
        logger.info(f"Evaluation metrics: {metrics}")

        # === Log metrics (skip None and NaN) ===
        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                logger.debug(f"Skipping metric {k} because value is None/NaN")
                continue
            mlflow.log_metric(k, float(v))

        # === Log raw data which was the source of version used for training the model ===
        dataset = mlflow.data.from_pandas(
            df.head(20),
            source=meta["raw_data_remote"],
            name=f"fraud_{version_tag}"
        )
        mlflow.log_input(dataset)

        # === Log final pipeline ===
        final_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        fraud_detector_local = "artifacts/models/fraud"
        os.makedirs(fraud_detector_local, exist_ok=True)
        joblib.dump(final_pipeline, f"{fraud_detector_local}/fraud_detector_{version_tag}.pkl")

        signature = infer_signature(X, final_pipeline.predict(X))
        mlflow.sklearn.log_model(sk_model=final_pipeline, 
                                name="FraudDetector",
                                registered_model_name=MODEL_NAME,
                                signature=signature,
                                input_example=X[:5])

        # Log the pipeline as an MLflow model (sklearn flavor)
        logger.info("Model pipeline logged to MLflow (artifact_path='model')")

        # === Log training metadata as dict artifact ===
        train_meta = {
            "version_tag": version_tag,
            "metrics": metrics,
            "raw_data_local": raw_data_local,
            "raw_features": list(raw_features),
            "preprocess_metadata_local": preprocess_metadata_local,
            "preprocessor_local": preprocessor_local,
            "training_script": "train/train_task.py",
            "training_timestamp": datetime.now().isoformat()
        }
        
        meta_path = f"{fraud_detector_local}/metadata_model_{version_tag}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # === Update version_meta (in-MLflow) and also log as artifact ===
        # version_meta = dict(meta)  # copy original
        meta.update({
            "raw_features": list(raw_features),
            "train_metrics": metrics,
            "train_timestamp": train_meta["training_timestamp"],
            "train_metadata_file": "train/train_metadata.json",
            "mlflow_run_id": run_id,
            "mlflow_model_uri": f"runs:/{run_id}/model"
        })

        with open("version_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
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

        logger.info(f"Model registered and transitioned to 'Production' (version {latest_model.version})")
        logger.info("✅ Training & registration complete.")


if __name__ == "__main__":
    main()