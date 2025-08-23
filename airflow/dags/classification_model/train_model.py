from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

import requests
import time
from datetime import datetime, timezone
import sys
import logging
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


def train_model():
    start = time.time()

    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)


    # Compute log loss
    loss = log_loss(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    duration = time.time() - start

############# Trigger Alerts #################
    # Uncomment to Trigger Alerts
    # Simulate poor accuracy
    y_pred = [0] * len(y_test)  # Predict all zeros
    acc = accuracy_score(y_test, y_pred)

    # Simulate slow training
    time.sleep(3)  # Delay > 2 seconds
    duration = time.time() - start

##################################################


    logger.info(f"Model accuracy: {acc:.2f}")
    logger.info(f"Training time: {duration:.2f} seconds")
    logger.info(f"Log loss: {loss:.4f}")

    metrics_data = {
        "model_name": "LogisticRegression_v1",
        "accuracy": acc,
        "training_duration": duration,
        "loss": loss,
        "metrics_timestamp": int(time.time()),
        "metrics_datetime": datetime.now(timezone.utc).isoformat()
    }

    try:
        response = requests.post("http://metrics-server:8000/model-metrics", json=metrics_data)
        response.raise_for_status()
        logger.info("Metrics successfully sent to metrics server.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send metrics: {e}")

    logger.info(f"Metrics for model {metrics_data["model_name"]} successfully posted.")