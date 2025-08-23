import mlflow
import json
from utils.logging.logger import get_logger
from .schema import Environment

logger = get_logger()

def load_latest_production_model(model_name, version=None):
    mlflow.set_tracking_uri(Environment.MLFLOW_TRACKING_URI.value)
    try:
        client = mlflow.tracking.MlflowClient()
        if version:
            model = mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
            logger.info(f"Rolled back to model version: {version}")
        else:
            prod_model = client.get_latest_versions(model_name, stages=[Environment.MODEL_STAGE.value])[0]
            version = prod_model.version
            logger.info("Loading Production model from MLflow", extra={"model_version": version})
            model = mlflow.sklearn.load_model(f"models:/{model_name}/{Environment.MODEL_STAGE.value}")
            logger.info(f"Production {model_name} loaded!")
            
        if model_name == "OutlierDetectionModel":
            run_id = prod_model.run_id
            logger.info(f"all the artifacts: {client.list_artifacts(run_id)}")
            config_path = client.download_artifacts(run_id, path="config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            anomaly_threshold = config["anomaly_threshold"]
            logger.info(f"Model artifact: anomaly threshold = {anomaly_threshold} .")
            return model, anomaly_threshold
        
        return model

    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise e
