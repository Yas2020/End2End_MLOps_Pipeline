
import pandas as pd
import numpy as np
from .metrics import *
from utils.logging.logger import get_logger

logger = get_logger("inference")

raw_features = ["amount", "transaction_time", "transaction_type", "location_region"]

def predict_single(input_dict, model_pipeline, outlier_detector, anomaly_threshold, version_tag):
    request_counter.labels(version=version_tag, endpoint="/predict").inc()

    X_raw = pd.DataFrame([input_dict], columns=raw_features)

    # Preprocess input
    preprocessor = model_pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_raw)

    # Score returns larger values for inliers; lower for outliers
    anomaly_score = float(-outlier_detector.score_samples(X_transformed)[0])
    is_outlier = anomaly_score <= anomaly_threshold
    if is_outlier:
        outlier_counter.labels(version=version_tag, endpoint="/predict").inc()

    # Predict
    probabilities = model_pipeline.predict_proba(X_raw)[0]
    is_fraud = np.argmax(probabilities)

    if is_fraud == 1:
        fraud_counter.labels(version=version_tag, endpoint="/predict").inc()

    return {
        "is_fraud": int(is_fraud),
        "probability": float(probabilities[1]),
        "is_outlier": is_outlier,
        "anomaly_score": anomaly_score,
        "review_required": is_outlier or is_fraud,
        "version": str(version_tag),
        "message": "Prediction Successful."
    }


def predict_batch(input_dicts, model_pipeline, outlier_detector, version_tag):
    
    # Convert list of inputs to DataFrame
    X_raw = pd.DataFrame(input_dicts, columns=raw_features)
    # Preprocess
    X_transformed = model_pipeline.named_steps["preprocessor"].transform(X_raw)
    # Outlier detection
    is_outlier_flags = outlier_detector.predict(X_transformed) == -1
    
    # Predict only for non-outliers
    predictions = []
    for i, outlier_flag in enumerate(is_outlier_flags):
        if outlier_flag:
            predictions.append({
                "is_fraud": None,
                "probability": None,
                "is_outlier": True,
                "version": version_tag,
                "message": "Outlier detected – no prediction"
            })
        else:
            is_fraud = model_pipeline.predict(X_raw.iloc[[i]])[0]
            prob = model_pipeline.predict_proba(X_transformed[i:i+1])[0][1]
            predictions.append({
                "is_fraud": int(is_fraud),
                "probability": float(prob),
                "is_outlier": False,
                "version": version_tag
            })

    return {"results": predictions}