from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import redis
import json
import numpy as np

from classification_model.train_model import train_model

def stream_data_with_drift():
    r = redis.Redis(host='redis', port=6379)
    for i in range(10):
        data = np.random.normal(loc=0.0, scale=1.0, size=100)
        if i > 5:
            data += 2.0  # Inject drift
        message = json.dumps(data.tolist())
        r.lpush('data_queue', message)
        print(f"Pushed batch {i} to queue")


def detect_drift():
    r = redis.Redis(host='redis', port=6379)
    drift_found = False
    batch_count = 0

    while True:
        message = r.rpop('data_queue')  # Get latest message
        if not message:
            break  # Queue empty

        batch_count += 1
        data = np.array(json.loads(message))
        drift_score = abs(np.mean(data))
        print(f"Batch {batch_count} - Drift score: {drift_score:.2f}")

        if drift_score > 2.0:
            print("Drift detected in batch", batch_count)
            drift_found = True

    if not drift_found:
        print("No drift detected.")


with DAG(
    dag_id="ml_pipeline_dag",
    start_date=datetime(2024, 1, 1),
    # schedule_interval=None,
    catchup=False
) as dag:

    stream_task = PythonOperator(
    task_id="stream_data_task",
    python_callable=stream_data_with_drift
    )

    drift_task = PythonOperator(
        task_id="detect_drift_task",
        python_callable=detect_drift
    )

    stream_task >> drift_task  # Define task order

    train_task = PythonOperator(
        task_id="train_classification_model_task",
        python_callable=train_model
    )

    drift_task >> train_task  # Run after drift detection