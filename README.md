# Full MLOps Pipeline in Production 
This project demonstrates a full life cycle of a machine learning operations pipeline from data, model development up to deployment and monitoring using open source tools with some best practices.

### Project Overview
This project implements a full MLOps system for fraud detection with:
- Data preprocessing & robust ML pipeline versioned by DVC, orchestrated by Airflow 3.x
- FastAPI-based inference server for online and batch predictions with 
  - Outlier detection (Isolation Forest or Z-score/IQR)
  - Input validation and schema enforcement (using Pydantic)
- Observability & monitoring: Prometheus + Grafana dashboards, alerts rules as code
- Logging and tracing for latency and pipeline analysis (OpenTelemetry + Jaeger)
- Automated model rollback via MLflow model registry + Airflow DAGs +  Alerts

### ML Pipeline
ML pipeline is reproduced by DVC and run as an Airflow DAG for maximum flexibility. This DAG has the following stages:
1. **Ingest**
   - Pulls raw dataset from remote and version it
2. **Preprocessing**
   - Trains a preprocessing pipeline (Standardization, encoding, missing value imputation)
   - Saves and versions `preprocessor.pkl` artifact for inference
3. **Outlier Detection**
    - Uses Isolation Forest (contamination=0.01) or Z-score/IQR to flag extreme inputs to prevent unreliable predictions at inference time
4. **Model Training**
    - Train, version model (preprocess pipeline+fraud detection model) with metadata (metrics, tags, input data)
    - Log & register model in MLflow (`mlflow.sklearn.log_model`) with signature

DVC ensures you can reproduce any experiment with the exact same dataset, pipeline and middle artifacts or configurations. It also chain the ML stages such that every stage runs only if the outputs of the previous stage has changed otherwise skipped. At every stage, latest version is pulled and after changed made by the stage, it will be pushed to remote DVC repo so it can be easily reproduced by other teammates without having to download the original data. They can start from any commit hash, pull the version and reproduce the pipeline for that experiment. DVC keeps data and pipelines tracked and synced together.

```sh
git checkout <commit-or-tag>  # pick the corresponding commit with the version
dvc pull  # downloads the exact deps/outs from remote
dvc repro preprocess. # returns the stage if the code has changed
```
To ingest new raw data (whether starting for the first time or want to try new raw data for new experiment), run
```sh
dvc repro ingest
``` 

### Inference Pipeline
FastAPI Microservice
   - Single inference: POST `/predict`
   - Batch inference: POST `/predict/batch`
   - Outlier detection integrated: inputs flagged before prediction
   - Input validation via Pydantic schemas
   - Confidence threshold checks to prevent low-confidence predictions
   - logged and traced every request life cycle


Return JSON:
```sh
././test_infer_ep.sh
---
Request #10: {"amount": 5000.0, "transaction_time": 2.0, "transaction_type": "online", "location_region": "Asia"}
{"is_fraud":1,"probability":1.0,"is_outlier":false,"anomaly_score":0.6764662176126935,"review_required":true,"version":"v20250821_170631","message":"Prediction Successful."}
---
Request #11: {"amount": 900, "transaction_time": 2.0, "transaction_type": "online", "location_region": "EU"}
{"is_fraud":0,"probability":0.36741115261724666,"is_outlier":false,"anomaly_score":0.6674423395914119,"review_required":false,"version":"v20250821_170631","message":"Prediction Successful."}
```

### Monitoring & Observability
- Each request logged (inference.log)
- Metrics exposed for Prometheus: request count, latency, outlier count, fraud count
- Prometheus Metrics
- Auto-instrument FastAPI with `prometheus-fastapi-instrumentator`
- Custom metrics via `prometheus-client`
- Grafana as Code to provision dashboards, panels from JSON/YAML files
- Prometheus alerts using alertmanager, Grafana alerts for dashboards with code

Example metrics panels:
Model: fraud prediction rate, outlier count, accuracy, latency
System: CPU, memory, disk usage, network

Alerts defined in YAML:
- High latency > 0.5s
- Outlier ratio > 10%

Tracing (OpenTelemetry + Jaeger)
- Automatic FastAPI instrumentation
- Manual spans for critical logic
- Trace IDs included in logs for correlation
- Visualize per-request flow & latency in Jaeger UI

### Automated Model Rollback
Trigger Criteria
- Latency spike (inference_latency_seconds > 0.5s)
- Accuracy drop (requires ground truth)
- Outlier rate surge (>10%)
- Business KPI drop

#### Rollback Flow
- Prometheus/Grafana alert fires → POST to FastAPI `/alert`
- FastAPI triggers Airflow DAG (`/api/v1/dags/model_rollback/dagRuns`)
- Airflow DAG finds last stable MLflow model
- Depromotes current model from Production → Staging
- Reloads previous model via FastAPI `/model_rollback endpoint`

#### Testing Example
- DelayedLogisticRegression subclass simulates slow prediction. Saved in same module path as logging for MLflow unpickling
- Trigger DAG `register_high_latency_model` to deploy the slow model into production (version increases by 1).
- Run `test_infer_ep.sh` to send traffic to inference endpoint `/predict` to see inference latency increases (to 5s), 
![](./images/latency_increase.png)

which triggers Prometheus alert. 
![](./images/alert-fires.png)
It fires to call FastAPI `/alert`

![](./images/alert_received.png)
- The FastAPI alert triggers an Airflow DAG to roll the model back to the previous version in production.  
- FastAPI calls Airflow API ro run model_rollback DAG: `http://airflow-apiserver:8080/api/v2/dags/model_rollback/dagRuns"`. This DAG de-promotes the slow model to stage and puts back the previous model into production in MLflow. Also calls FastAPI endpoint `/rollback_model` to load the previous model.


## Running Locally

- Run all the services: airflow, mlflow, inference and monitoring
- Go to `http://localhost:9001` to create a bucket named "mlflow-artifacts".
pip install -r inference/requirements.txt



<!-- # Start Prometheus + Grafana + Alertmanager via Docker Compose
docker-compose -f monitoring/docker-compose-monitoring.yaml up
Docker & Deployment
Dockerfile for inference service, includes /utils in PYTHONPATH
docker-compose orchestrates:
FastAPI inference service
Prometheus + Alertmanager
Grafana + provisioned dashboards
Node-exporter for system metrics
Batch inference triggered via Airflow DAGs
 -->


<!-- Example API Requests
Single inference
POST http://localhost:8000/predict
Content-Type: application/json

{
  "transaction_amount": 123.45,
  "user_id": 98765,
  ...
}
 -->
