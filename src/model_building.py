import yaml
import numpy as np
import os
import sys
import joblib
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

from src.logger import logging
from src.exception import customexception


# CONFIG CLASS
@dataclass
class ModelBuildingConfig:
    model_type: str = None
    model_params: dict = None
    model_save_path: str = None

    @classmethod
    def from_yaml(cls, yaml_path: str = "params.yaml"):
        try:
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)

            model_config = config.get('model', {})

            return cls(
                model_type=model_config.get('type', 'RandomForestRegressor'),
                model_params=model_config.get('params', {}),
                model_save_path=model_config.get('save_path', "models/model.pkl")
            )

        except Exception as e:
            logging.error(f"Error loading params.yaml: {str(e)}")
            raise customexception(e, sys)


# MODEL BUILDING CLASS
class ModelBuilding:

    def __init__(self, yaml_config_path: str = "params.yaml"):
        self.config = ModelBuildingConfig.from_yaml(yaml_config_path)

    # LOAD DATA
    def load_processed_data(self, processed_data_path: str):
        try:
            X_train = np.load(os.path.join(processed_data_path, 'X_train.npy'))
            X_test = np.load(os.path.join(processed_data_path, 'X_test.npy'))
            y_train = np.load(os.path.join(processed_data_path, 'y_train.npy'))
            y_test = np.load(os.path.join(processed_data_path, 'y_test.npy'))

            # Feature names (optional)
            feature_names_path = os.path.join(processed_data_path, 'feature_names.npy')
            if os.path.exists(feature_names_path):
                feature_names = np.load(feature_names_path, allow_pickle=True)
            else:
                feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

            logging.info(f"Data Loaded: X_train {X_train.shape}, X_test {X_test.shape}")

            return X_train, X_test, y_train, y_test, feature_names

        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise customexception(e, sys)

    # GET MODEL
    def get_model(self):
        model_type = self.config.model_type
        params = self.config.model_params

        try:
            if model_type == "RandomForestRegressor":
                return RandomForestRegressor(**params)

            elif model_type == "GradientBoostingRegressor":
                return GradientBoostingRegressor(**params)

            elif model_type == "LinearRegression":
                return LinearRegression(**params)

            elif model_type == "Ridge":
                return Ridge(**params)

            elif model_type == "Lasso":
                return Lasso(**params)

            elif model_type == "DecisionTreeRegressor":
                return DecisionTreeRegressor(**params)

            elif model_type == "XGBRegressor":
                return xgb.XGBRegressor(**params)

            elif model_type == "KNeighborsRegressor":
                return KNeighborsRegressor(**params)

            elif model_type == "SVR":
                return SVR(**params)

            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        except Exception as e:
            logging.error(f"Model creation failed: {str(e)}")
            raise customexception(e, sys)

    # TRAIN MODEL
    def train_model(self, X_train, y_train):
        try:
            model = self.get_model()

            logging.info(f"Training started for {self.config.model_type}")
            model.fit(X_train, y_train)
            logging.info("Training completed")

            return model

        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise customexception(e, sys)

    # SAVE MODEL
    def save_model(self, model):
        try:
            save_path = self.config.model_save_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            joblib.dump(model, save_path)

            logging.info(f"Model saved at {save_path}")
            return save_path

        except Exception as e:
            logging.error(f"Model saving failed: {str(e)}")
            raise customexception(e, sys)

    # SAVE METADATA
    def save_metadata(self, metadata: dict):
        try:
            metadata_path = "models/metadata.yaml"
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

            with open(metadata_path, "w") as f:
                yaml.dump(metadata, f)

            logging.info("Metadata saved")

        except Exception as e:
            logging.error(f"Metadata saving failed: {str(e)}")
            raise customexception(e, sys)

    # MAIN PIPELINE
    def initiate_model_building(self, processed_data_path: str):

        logging.info("MODEL BUILDING STARTED")

        try:
            # Load data
            X_train, X_test, y_train, y_test, feature_names = self.load_processed_data(processed_data_path)

            # Train
            model = self.train_model(X_train, y_train)

            # Save model
            model_path = self.save_model(model)

            # Metadata
            metadata = {
                "model_type": self.config.model_type,
                "model_params": self.config.model_params,
                "train_shape": X_train.shape,
                "test_shape": X_test.shape,
                "features": feature_names.tolist() if isinstance(feature_names, np.ndarray) else feature_names,
                "model_path": model_path
            }

            self.save_metadata(metadata)

            logging.info("MODEL BUILDING COMPLETED")

            print("\n Model Training Done")
            print(f"Model: {self.config.model_type}")
            print(f"Saved at: {model_path}")

            return model

        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            raise customexception(e, sys)


if __name__ == "__main__":
    builder = ModelBuilding()
    builder.initiate_model_building("data/processed/")