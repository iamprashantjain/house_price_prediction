import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

# MLflow auth (already working)
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri("https://dagshub.com/iamprashantjain/house_price_prediction.mlflow")

with mlflow.start_run():
    mlflow.log_param("parameter name", "value")
    mlflow.log_metric("metric name", 1)