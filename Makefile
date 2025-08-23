# Makefile for ML pipeline services

AIRFLOW = -f ./airflow/docker-compose.airflow.yaml 
MLFLOW = -f ./mlflow/docker-compose.mlflow.yaml
MONITOR = -f ./monitoring/docker-compose.monitoring.yaml
INFERENCE = -f ./inference/docker-compose.inference.yaml

# Run everything
all: up

up:
	docker compose $(AIRFLOW) $(MLFLOW) $(MONITOR) $(INFERENCE) up -d

down:
	docker compose $(AIRFLOW) $(MLFLOW) $(MONITOR) $(INFERENCE) down

rebuild:
	docker-compose $(MLFLOW) $(MONITOR) $(API) down
	docker-compose $(MLFLOW) $(MONITOR) $(API) build --no-cache
	docker-compose $(MLFLOW) $(MONITOR) $(API) up -d

.PHONY: airflow
airflow:
	docker compose $(AIRFLOW) up -d --build

.PHONY: mlflow
mlflow:
	docker compose $(MLFLOW) up -d

monitor:
	docker compose $(MONITOR) up -d

.PHONY: inference
inference:
	docker compose $(INFERENCE) up -d

clean:
	docker compose $(AIRFLOW) $(MLFLOW) $(MONITOR) $(INFERENCE) down -v --remove-orphans
	docker network prune -f
	docker volume prune -f
