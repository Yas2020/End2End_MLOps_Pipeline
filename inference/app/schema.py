from pydantic import BaseModel
from enum import Enum
from typing import Optional


class TransactionInput(BaseModel):
    amount: float
    transaction_time: float
    transaction_type: str
    location_region: str

class PredictionOutput(BaseModel):
    is_fraud: int
    probability: float
    is_outlier: bool
    anomaly_score: float
    review_required: bool
    version: str
    message: str

class Environment(str, Enum):
    MODEL_NAME="FraudDetectionModel"
    MODEL_STAGE="Production"
    MLFLOW_TRACKING_URI="http://mlflow:5000"
    EXPERIMENT_NAME="EXPERIMENT_NAME"
