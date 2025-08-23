from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'start_date': datetime(2025, 8, 1),
    'retries': 1
}

with DAG(
    dag_id="daily_batch_inference",
    # schedule_interval="@daily",
    default_args=default_args,
    catchup=False
) as dag:

    generate_batch_input = BashOperator(
        task_id="generate_batch_input",
        bash_command="cd /opt/airflow/ml_pipeline && python batch/generate_batch_input.py"
    )
    
    run_batch_inference = BashOperator(
        task_id="run_batch_inference",
        bash_command="cd /opt/airflow/ml_pipeline && python batch/batch_inference.py"
    )

    # Task order
    generate_batch_input >> run_batch_inference
