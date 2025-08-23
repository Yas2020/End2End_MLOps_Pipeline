from fastapi import FastAPI
from fastapi import Response, Request
from pydantic import BaseModel
import redis
import os
import json
import logging
import sys


from pythonjsonlogger import jsonlogger

# logs will be JSON — great for scraping, parsing, monitoring
logHandler = logging.StreamHandler(sys.stdout)
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)

# Create logger
logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Set JSON formatter
formatter = jsonlogger.JsonFormatter(
    '%(asctime)s %(levelname)s %(name)s %(message)s'
)
logHandler.setFormatter(formatter)

# Clear old handlers and add new one
logger.handlers = []  # remove default handlers
logger.addHandler(logHandler)



app = FastAPI()

# Redis config via ENV
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "1"))

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

class MetricsInput(BaseModel):
    model_name: str
    accuracy: float
    loss: float
    training_duration: float
    metrics_timestamp: int
    metrics_datetime: str


@app.post("/model-metrics")
def store_model_metrics(metrics: MetricsInput):
    key = f"metrics:{metrics.model_name}"
    value = json.dumps({"accuracy": metrics.accuracy, 
                        "training_duration": metrics.training_duration, 
                        "loss": metrics.loss,
                        "metrics_timestamp": metrics.metrics_timestamp,
                        "metrics_datetime": metrics.metrics_datetime})
    r.set(key, value)
    return {"message": "Metrics stored."}

@app.get("/model-metrics")
def prometheus_metrics():
    '''
    Prometheus expects plain text in a specific format. 
    It can’t parse JSON.
    '''
    keys = r.keys("metrics:*")
    lines = []

    for key in keys:
        model_name = key.decode().split(":")[1]
        data = json.loads(r.get(key))
        acc = data.get("accuracy", 0)
        dur = data.get("training_duration", 0)
        loss = data.get("loss", 0)
        ts = data.get("metrics_timestamp", 0)

        lines.append(f'model_accuracy{{model_name="{model_name}"}} {acc}')
        lines.append(f'training_duration_seconds{{model_name="{model_name}"}} {dur}')
        lines.append(f'loss{{model_name="{model_name}"}} {loss}')
        lines.append(f'timestamp{{model_name="{model_name}"}} {ts}')

    body = "\n".join(lines) + "\n"
    return Response(content=body, media_type="text/plain")


@app.post("/alert")
async def receive_alert(request: Request):
    payload = await request.json()
    logging.info(f"Received Alert: {payload}")
    return {"status": "Alert received"}