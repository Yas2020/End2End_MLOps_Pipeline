# Makefile for ML pipeline services

AIRFLOW = -f ./airflow/docker-compose.airflow.yaml 
MLFLOW = -f ./mlflow/docker-compose.mlflow.yaml
MONITOR = -f ./monitoring/docker-compose.monitoring.yaml
INFERENCE = -f ./inference/docker-compose.inference.yaml

# Run everything
all: up

up:
	docker compose $(AIRFLOW) up -d
	docker compose $(MLFLOW) up -d
	docker compose $(MONITOR) up -d
	docker compose $(INFERENCE) up -d

down:
	docker compose $(AIRFLOW) down
	docker compose $(MLFLOW) down
	docker compose $(MONITOR) down
	docker compose $(INFERENCE) down

clean:
	make down
	docker network prune -f
	docker volume prune -f
	docker system prune -f
