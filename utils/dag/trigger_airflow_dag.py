import os
import requests
from datetime import datetime, timezone

AIRFLOW_URL = os.environ["AIRFLOW_URL"]
AIRFLOW_USERNAME = os.environ["AIRFLOW_USERNAME"]
AIRFLOW_PASSWORD = os.environ["AIRFLOW_PASSWORD"]

def trigger_airflow_dag(dag_id: str, conf: dict = None):
    """
    Authenticate with Airflow 3.x (FAB) and trigger a DAG run.

    Args:
        dag_id: DAG to trigger.
        username: Airflow username.
        password: Airflow password.
        airflow_url: Base URL of Airflow.
        conf: Optional dict to pass to DAG run.

    Returns:
        Dict response from DAG run creation.
    """
    conf = conf or {}

    # Get JWT token
    auth_resp = requests.post(
        f"{AIRFLOW_URL}/auth/token",
        headers = {"Content-Type": "application/json"},
        json={"username": AIRFLOW_USERNAME, "password": AIRFLOW_PASSWORD}
    )
    auth_resp.raise_for_status()
    token = auth_resp.json()["access_token"]

    # Trigger DAG run
    logical_date = datetime.now(timezone.utc).isoformat()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "logical_date": logical_date,
        "conf": conf
    }

    dag_resp = requests.post(
        f"{AIRFLOW_URL}/api/v2/dags/{dag_id}/dagRuns",
        headers=headers,
        json=payload
    )
    dag_resp.raise_for_status()
    return dag_resp


if __name__ == "__main__":
    resp = trigger_airflow_dag(
        dag_id="model_rollback",
        conf={}  # optional
    )
    print("DAG triggered:", resp)
