import os
import sys
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.logger import logging
from src.exception import customexception
from dotenv import load_dotenv;load_dotenv()
import os

# Config Class
@dataclass
class ModelEvaluationConfig:
    processed_data_path: str
    model_path: str
    target_column: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str

    metrics_path: str
    predictions_path: str
    evaluation_threshold: float
    model_info_path: str

def load_config(config_path="params.yaml") -> ModelEvaluationConfig:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return ModelEvaluationConfig(
    processed_data_path=config["data"]["processed_data_path"],
    model_path=config["model"]["save_path"],
    target_column=config["features"]["target_column"],
    mlflow_tracking_uri=config["mlflow"]["tracking_uri"],
    mlflow_experiment_name=config["mlflow"]["experiment_name"],
    metrics_path=config["model_evaluation"]["metrics_save_path"],
    predictions_path=config["model_evaluation"]["predictions_save_path"],
    evaluation_threshold=config["model_evaluation"]["evaluation_threshold"],
    model_info_path=config["model_evaluation"]["model_info_path"]
)


# Evaluation Class
class ModelEvaluator:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

    # Load Model
    def load_model(self):
        try:
            model = joblib.load(self.config.model_path)
            logging.info(f"Model loaded from {self.config.model_path}")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    # Load Data
    def load_data(self):
        try:
            X_test = np.load(os.path.join(self.config.processed_data_path, "X_test.npy"))
            y_test = np.load(os.path.join(self.config.processed_data_path, "y_test.npy"))

            logging.info(f"Loaded test data: {X_test.shape}")
            return X_test, y_test
        except Exception as e:
            logging.error(f"Error loading test data: {e}")
            raise

    # Metrics
    def evaluate(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        accuracy = (1 - mae / y_true.mean()) * 100

        metrics = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "MAPE": float(mape),
            "Accuracy_Percentage": float(accuracy)
        }

        return metrics

    # Save Outputs
    def save_metrics(self, metrics):
        os.makedirs("reports", exist_ok=True)
        with open(self.config.metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info("Metrics saved")

    def save_predictions(self, y_true, y_pred):
        df = pd.DataFrame({
            "actual": y_true,
            "predicted": y_pred,
            "error": y_true - y_pred
        })
        df.to_csv(self.config.predictions_path, index=False)
        logging.info("Predictions saved")

    def save_model_info(self, run_id):
        info = {
            "run_id": run_id,
            "model_path": "model"
        }
        with open(self.config.model_info_path, "w") as f:
            json.dump(info, f, indent=4)
        logging.info("Model info saved")

    # Feature Importance
    def get_feature_importance(self, model, n_features):
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        elif hasattr(model, "coef_"):
            return model.coef_
        return None

    # Main Pipeline
    def run(self):
        with mlflow.start_run() as run:

            # Load
            model = self.load_model()
            X_test, y_test = self.load_data()

            # Predict
            y_pred = model.predict(X_test)

            # Evaluate
            metrics = self.evaluate(y_test, y_pred)

            # Save locally
            self.save_metrics(metrics)
            self.save_predictions(y_test, y_pred)

            # MLflow logging
            mlflow.log_metrics(metrics)

            # Log model params
            if hasattr(model, "get_params"):
                mlflow.log_params(model.get_params())

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Feature importance
            importance = self.get_feature_importance(model, X_test.shape[1])
            if importance is not None:
                fi_df = pd.DataFrame({
                    "feature_index": range(len(importance)),
                    "importance": importance
                })
                fi_path = "reports/feature_importance.csv"
                fi_df.to_csv(fi_path, index=False)
                mlflow.log_artifact(fi_path)

            # Save run info (IMPORTANT for registry)
            self.save_model_info(run.info.run_id)

            # Log artifacts
            mlflow.log_artifact(self.config.metrics_path)
            mlflow.log_artifact(self.config.predictions_path)
            mlflow.log_artifact(self.config.model_info_path)

            logging.info("Model Evaluation Completed")

            return metrics


# Run
if __name__ == "__main__":
    config = load_config("params.yaml")
    evaluator = ModelEvaluator(config)
    evaluator.run()