FROM apache/airflow:slim-3.0.5-python3.11

USER root

# Install OS dependencies
RUN apt-get update && \
    apt-get install -y gcc g++ libpq-dev build-essential git && \
    apt-get clean

# Copy requirements
COPY requirements.txt .

# Switch back to airflow user
USER airflow

# Upgrade pip and install Python packages with official Airflow constraints
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.5/constraints-3.11.txt"

# Set PYTHONPATH for your ML pipeline
ENV PYTHONPATH="/opt/airflow/ml_pipeline:${PYTHONPATH}"
