# .devcontainer/Dockerfile
FROM python:3.10-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl gcc g++ libssl-dev libffi-dev \
    # redis-server sudo procps \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Create user for DevContainer and give sudo with no password required
RUN useradd -ms /bin/bash Yas && \
    echo "Yas ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /project
ENV PYTHONPATH="/home/Yas/.local/bin:${PYTHONPATH}"
ENV AWS_ACCESS_KEY=${AWS_ACCESS_KEY}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_ENDPOINT_URL=${AWS_ENDPOINT_URL}

# To avoid retyping export AIRFLOW_HOME=..., add this
# ENV AIRFLOW_HOME=/project/airflow

# Build the monitoring system in Docker Compose
# CMD ["uvicorn", "metrics_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
