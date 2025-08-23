FROM python:3.11-slim

RUN pip install --upgrade pip && \
    pip install mlflow[extras] psycopg2-binary boto3 scikit-learn

WORKDIR /mlflow
EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://mlflow:mlflowpass@mlflow_postgres:5432/mlflow_db", \
     "--default-artifact-root", "s3://mlflow-artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
