import os
import json
import yaml
import mlflow
import logging
from dataclasses import dataclass
from mlflow.tracking import MlflowClient
from src.logger import logging
from src.exception import customexception
from dotenv import load_dotenv;load_dotenv()
import os


# ================= CONFIG =================
@dataclass
class ModelRegistryConfig:
    mlflow_tracking_uri: str
    model_name: str
    stage: str
    model_info_path: str
    evaluation_threshold: float
    registry_info_path: str


def load_config(config_path="params.yaml") -> ModelRegistryConfig:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return ModelRegistryConfig(
        mlflow_tracking_uri=config["mlflow"]["tracking_uri"],
        model_name=config["model_registry"]["model_name"],
        stage=config["model_registry"]["stage"],
        model_info_path=config["model_evaluation"]["model_info_path"],
        evaluation_threshold=config["model_evaluation"]["evaluation_threshold"],
        registry_info_path=config["model_registry"]["registry_info_path"]
    )

# ================= REGISTRY CLASS =================
class ModelRegistry:

    def __init__(self, config: ModelRegistryConfig):
        self.config = config

        # Dagshub authentication
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        self.client = MlflowClient()

    # -------- Load model info --------
    def load_model_info(self):
        try:
            with open(self.config.model_info_path, "r") as f:
                info = json.load(f)
            logging.info(f"Loaded run_id: {info['run_id']}")
            return info
        except Exception as e:
            logging.error(f"Error loading model info: {e}")
            raise

    # -------- Get metrics --------
    def get_metrics(self, run_id):
        try:
            run = self.client.get_run(run_id)
            return run.data.metrics
        except Exception as e:
            logging.error(f"Error fetching metrics for run {run_id}: {e}")
            raise

    # -------- Register model --------
    def register_model(self, run_id, model_path):
        try:
            model_uri = f"runs:/{run_id}/{model_path}"
            result = mlflow.register_model(model_uri=model_uri, name=self.config.model_name)
            version = result.version
            logging.info(f"Registered {self.config.model_name} v{version}")
            return version
        except Exception as e:
            logging.error(f"Registration failed: {e}")
            raise

    # -------- Promote model --------
    def promote_model(self, version, metrics):
        try:
            r2 = metrics.get("R2", 0)
            if r2 >= self.config.evaluation_threshold:
                self.client.transition_model_version_stage(
                    name=self.config.model_name,
                    version=version,
                    stage=self.config.stage,
                    archive_existing_versions=True
                )
                logging.info(f"Promoted to {self.config.stage}")
            else:
                logging.info(f"Not promoted (R2={r2})")
        except Exception as e:
            logging.error(f"Promotion failed: {e}")
            raise

    # -------- Save registry info --------
    def save_registry_info(self, version, run_id):
        info = {
            "model_name": self.config.model_name,
            "version": version,
            "stage": self.config.stage,
            "run_id": run_id
        }
        os.makedirs(os.path.dirname(self.config.registry_info_path), exist_ok=True)
        with open(self.config.registry_info_path, "w") as f:
            json.dump(info, f, indent=4)
        logging.info(f"Registry info saved at {self.config.registry_info_path}")

    # -------- Run --------
    def run(self):
        info = self.load_model_info()
        run_id = info["run_id"]
        model_path = info.get("model_path", "model")

        metrics = self.get_metrics(run_id)
        version = self.register_model(run_id, model_path)
        self.promote_model(version, metrics)
        self.save_registry_info(version, run_id)

        logging.info("Model Registry Completed")


# ================= MAIN =================
if __name__ == "__main__":
    config = load_config("params.yaml")
    registry = ModelRegistry(config)
    registry.run()