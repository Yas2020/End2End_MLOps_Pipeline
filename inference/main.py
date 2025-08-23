import json
import uuid
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from app.predict import predict_single, predict_batch
from app.metrics import *
from utils.logging.logger import get_logger
from utils.tracing.tracing import setup_tracer 
from utils.dag.trigger_airflow_dag import trigger_airflow_dag
from app.model_loader import load_latest_production_model
from app.schema import TransactionInput, PredictionOutput
from typing import List

app = FastAPI(title="Fraud Detection Inference API")

# Instrumentation for Prometheus-exposes metric automatically-ready to scrape
instrumentator = Instrumentator().instrument(app).expose(app)
# get automatic traces for HTTP requests plus manual detailed spans inside your code.
FastAPIInstrumentor.instrument_app(app)

logger = get_logger("inference_server")
tracer = setup_tracer(service_name="inference_server")


# === Load Artifacts ===
with open("version_meta.json") as f:
    meta = json.load(f)
version_tag = meta["version_tag"]

# Load latest production model from MLflow Registry
model_pipeline = None  # Global cache for model
# outlier_detector = joblib.load(meta["outlier_artifact"])
outlier_detector = None

# Benefit: avoid reloading model from MLflow every time.
# Drawback: to rollback, FastAPI needs to reload model manually or via reload endpoint.
@app.on_event("startup")
def startup_event():
    global model_pipeline
    global outlier_detector
    global anomaly_threshold
    # Initialize metrics to 0
    fraud_counter.labels(version=version_tag, endpoint="/predict").inc(0)
    outlier_counter.labels(version=version_tag, endpoint="/predict").inc(0)
    request_counter.labels(version=version_tag, endpoint="/predict").inc(0)
    #  Load latest model version registered for production
    try:
        outlier_detector, anomaly_threshold = load_latest_production_model(model_name="OutlierDetectionModel")
        model_pipeline = load_latest_production_model(model_name="FraudDetectionModel")
    except Exception as e:
        logger.error(e)
        logger.info("Run 'ml_pipeline_dvc' Airflow DAG first to have models trained & registered into Production stage!")

@app.post("/predict", response_model=PredictionOutput)
def predict(transaction: TransactionInput):
    global model_pipeline
    
    request_id = str(uuid.uuid4())
    logger.info(f"Request star", extra={"request_id": request_id, "version": version_tag})

    with tracer.start_as_current_span("predict_single"):  
        with inference_latency.labels(version=version_tag, endpoint="/predict").time():
            input_dict = transaction.model_dump()
            try:   
                res = predict_single(
                    input_dict, 
                    model_pipeline, 
                    outlier_detector,
                    anomaly_threshold, 
                    version_tag
                )
                logger.info({
                    "request_id": request_id,
                    "is_fraud": res["is_fraud"],
                    "is_outlier": res["is_outlier"],
                    "probability": res["probability"],
                    "version": str(version_tag)
                })
                return res
            except Exception as e:
                logger.error(
                    "Inference error", 
                    exc_info=True,
                    extra={"request_id": request_id, "version": version_tag}
                )
                raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/predict/batch", response_model=List[PredictionOutput])
def predict(transactions: List[TransactionInput]):
    input_dicts = [t.model_dump() for t in transactions]
    try:
        return predict_batch(
            input_dicts, 
            model_pipeline, 
            outlier_detector, 
            version_tag
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/alert")
async def handle_alert(alerts: dict):
    alert_labels = alerts["alerts"][0]['labels']
    alert_name = alert_labels["alertname"]
    logger.warning("ALERT RECEIVED", extra={"alert_name": alert_name})
    if alert_name == "HighInferenceLatency":
        # Trigger rollback DAG
        try:
            response = trigger_airflow_dag(dag_id="model_rollback", conf={})
        except Exception as e:
            logger.error("Airflow API call failed.", e)
        if response.status_code == 200:
            logger.info(response.json())
            return {"status": "Rollback triggered"}
        else:
            return {"status": "Failed to trigger rollback", "details": response.text}
    return {"status": "Alert received"}


@app.post("/rollback_model")
def rollback_model(version: int):
    global model_pipeline
    try:
        logger.info(f"Rolling back to version {version}")
        model_pipeline = load_latest_production_model(model_name="FraudDetectionModel", version=version)
        return {"status": "Model rolled back", "version": version}
    except Exception as e:
        logger.error(f"Model rollback failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Rollback error: {str(e)}")