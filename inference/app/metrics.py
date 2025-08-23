from prometheus_client import Histogram
from prometheus_client import Counter


fraud_counter = Counter("fraud_predictions_total", "Total fraud predictions", ["version", "endpoint"])
outlier_counter = Counter("outlier_detected_total", "Total outliers detected", ["version", "endpoint"])
request_counter = Counter("inference_requests_total", "Total number of inference requests", ["version", "endpoint"])

inference_latency = Histogram(
    "inference_latency_seconds",
    "Latency of inference requests",
    ["version", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0, 60.0, float("inf"))
)