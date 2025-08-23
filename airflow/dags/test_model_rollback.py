import os
import mlflow
from sklearn.pipeline import Pipeline
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from utils.delayed_model.delayed_model import DelayedLogisticRegression

default_args = {
    'start_date': datetime(2025, 8, 1),
    'retries': 1
}

# === Constants ===
MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "FraudDetectionModel"
MODEL_NAME = "FraudDetectionModel"

def register_high_latency_model_from_latest_version():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MODEL_NAME)

    # ✅ 1. Load original pipeline from MLflow (preprocessing + original model)
    logged_model_uri = f"models:/{MODEL_NAME}/Production"
    original_pipeline = mlflow.sklearn.load_model(logged_model_uri)

    # Replace LogisticRegression with Delayed Version
    steps = original_pipeline.steps
    new_steps = []
    for name, step in steps:
        if isinstance(step, LogisticRegression):
            delayed_model = DelayedLogisticRegression()
            delayed_model.__dict__.update(step.__dict__) # Copy trained weights
            step = delayed_model    
        new_steps.append((name, step))
    # Create new pipeline with delayed model
    delayed_pipeline = Pipeline(steps=new_steps)
    
    with mlflow.start_run():
        # Log the delayed model (sklearn model under 'model' subdirectory)
        mlflow.sklearn.log_model(
            sk_model=delayed_pipeline, 
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )
        print("✅ Delayed model logged and registered!")

    # Transition to Production stage
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[-1]
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"🚀 Model version {latest_version.version} transitioned to 'Production'")
    print("✅ Model registration and promotion complete.")
    print("✅ High latency model registered successfully and is in production stage.")


with DAG("register_high_latency_model", 
        #  schedule_interval=None, 
         default_args=default_args, 
         catchup=False) as dag:

    rollback = PythonOperator(
        task_id="rollback_model",
        python_callable=register_high_latency_model_from_latest_version
    )


