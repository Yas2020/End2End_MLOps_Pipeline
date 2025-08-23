from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
import requests
import os

default_args = {
    'start_date': datetime(2025, 8, 1),
    'retries': 1
}

# === Constants ===
MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "FraudDetectionModel"
MODEL_NAME = "FraudDetectionModel"

def rollback_model():
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        versions = client.get_latest_versions("FraudDetectionModel")

        # Get current production version
        prod_version = next((v for v in versions if v.current_stage == "Production"), None)
        prev_versions = client.search_model_versions("name='FraudDetectionModel'")
        prev_versions_sorted = sorted(prev_versions, key=lambda v: int(v.version))

        if len(prev_versions_sorted) < 2:
            raise Exception("No previous version to roll back to.")

        # Get the previous version before current production
        prev_version = prev_versions_sorted[-2]

        # Move current production to Staging
        client.transition_model_version_stage(
            name="FraudDetectionModel",
            version=prod_version.version,
            stage="Staging",
            archive_existing_versions=False
        )

        # Move previous version to Production
        client.transition_model_version_stage(
            name="FraudDetectionModel",
            version=prev_version.version,
            stage="Production",
            archive_existing_versions=False
        )

        rollback_response = requests.post(
            f"http://inference-api:8000/rollback_model?version={int(prev_version.version)}"
        )

        print(f"Rollback response: {rollback_response.text}")
    
    except Exception as e:
        print(f"Rollback failed: {e}")
        raise e


with DAG("model_rollback", 
        #  schedule_interval=None, 
         default_args=default_args, 
         catchup=False) as dag:

    rollback = PythonOperator(
        task_id="rollback_model",
        python_callable=rollback_model
    )
