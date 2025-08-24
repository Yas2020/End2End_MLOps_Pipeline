from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'yas',
    'start_date': datetime.now().isoformat(),
    'retries': 0,
}

with DAG(
    dag_id='ml_pipeline_dvc',
    default_args=default_args,
    # schedule_interval=None,  # Trigger manually or later cron
    catchup=False,
    tags=['mlops', 'dvc']
) as dag:
    
    # Use --force flag only if you want to download new raw data that is not versioned before by DVC 
    ingest = BashOperator(
        task_id='ingest_dvc',
        bash_command='''
            cd /opt/airflow/ml_pipeline 
            dvc repro ingest
            dvc push
        '''
    )

    preprocess = BashOperator(
        task_id='preprocess_dvc',
        bash_command='''
            cd /opt/airflow/ml_pipeline
            dvc pull --force
            dvc repro preprocess
            dvc push
        '''
    ) 
    
    train_outlier = BashOperator(
        task_id='train_outlier_detector_dvc',
        bash_command='''
            cd /opt/airflow/ml_pipeline
            dvc pull --force
            dvc repro train_outlier
            dvc push
        '''
    )

    train_model = BashOperator(
        task_id='train_fraud_detector_dvc',
        bash_command='''
            git config --global --add safe.directory /opt/airflow/ml_pipeline
            cd /opt/airflow/ml_pipeline
            dvc pull --force
            dvc repro train_model
            dvc push
        ''',
    )

    ingest >> preprocess >> train_outlier >> train_model










  # setup_dvc_remote = BashOperator(
    #     task_id="setup_dvc_remote",
    #     bash_command="""
    #     cd /opt/airflow/ml_pipeline
    #     aws --endpoint-url http://minio:9000 s3 ls s3://mlflow-artifacts
    #     dvc remote remove minio || true
    #     dvc remote add -d minio s3://mlflow-artifacts
    #     dvc remote modify minio endpointurl http://minio:9000
    #     dvc remote modify minio access_key_id minioadmin
    #     dvc remote modify minio secret_access_key minioadmin
    #     dvc remote modify minio use_ssl false
    #     export AWS_ACCESS_KEY_ID=minioadmin
    #     export AWS_SECRET_ACCESS_KEY=minioadmin
    #     export AWS_ENDPOINT_URL=http://minio:9000
    #     dvc push --verbose
    #     echo "=== DVC CONFIG AFTER SETUP ==="
    #     cat .dvc/config
    #     """
    # )