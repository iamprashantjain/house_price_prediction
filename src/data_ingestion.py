import yaml
import pandas as pd
import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import customexception

@dataclass
class DataIngestionConfig:
    merged_data_path: str = None
    train_data_path: str = None
    test_data_path: str = None
    flats_raw_data_path: str = None
    houses_raw_data_path: str = None
    test_size: float = None
    random_state: int = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str = "params.yaml"):
        """Load configuration from YAML file"""
        try:
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Extract paths from YAML
            data_config = config.get('data', {})
            raw_paths = data_config.get('raw_data_paths', {})
            split_config = config.get('data_split', {})
            
            return cls(
                merged_data_path=data_config.get('merged_data_path', 
                    os.path.join("data", "raw.csv")),
                train_data_path=data_config.get('train_data_path',
                    os.path.join("data", "train.csv")),
                test_data_path=data_config.get('test_data_path',
                    os.path.join("data", "test.csv")),
                flats_raw_data_path=raw_paths.get('flats', 
                    "data/flats.csv"),
                houses_raw_data_path=raw_paths.get('houses', 
                    "data/houses.csv"),
                test_size=split_config.get('test_size', 0.2),
                random_state=split_config.get('random_state', 42)
            )
        except Exception as e:
            logging.error(f"Error loading params.yaml: {str(e)}")

class DataIngestion:
    def __init__(self, yaml_config_path: str = "params.yaml"):
        # Load configuration from YAML
        self.ingestion_config = DataIngestionConfig.from_yaml(yaml_config_path)
        self.raw_data_path_1 = self.ingestion_config.flats_raw_data_path
        self.raw_data_path_2 = self.ingestion_config.houses_raw_data_path
    
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            # Read the two raw CSV files into separate dataframes
            data1 = pd.read_csv(self.raw_data_path_1)
            data2 = pd.read_csv(self.raw_data_path_2)
            
            logging.info(f"Reading data from source: {self.raw_data_path_1} and {self.raw_data_path_2}")

            # Add 'property_type' column to distinguish between flats and houses
            data1['property_type'] = 'flat'
            data2['property_type'] = 'house'
           
            # Concatenate the two dataframes into one
            data = pd.concat([data1, data2], ignore_index=True)
            logging.info(f"Both datasets concatenated successfully with shape: {data.shape}")
            
            # Save the concatenated data to the specified path
            os.makedirs(os.path.dirname(self.ingestion_config.merged_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.merged_data_path, index=False)
            logging.info(f"Merged data saved at: {self.ingestion_config.merged_data_path}")

            # Train-test split
            logging.info("Performing train-test split")
            train_data, test_data = train_test_split(
                data, 
                test_size=self.ingestion_config.test_size,
                random_state=self.ingestion_config.random_state,
                shuffle=True
            )
            
            logging.info(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
            
            # Save train and test data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            
            logging.info(f"Train data saved at: {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved at: {self.ingestion_config.test_data_path}")

            logging.info("Data ingestion completed successfully.")
            
            # Return the path to the merged data
            return {
                'train_data_path': self.ingestion_config.train_data_path,
                'test_data_path': self.ingestion_config.test_data_path,
                'merged_data_path': self.ingestion_config.merged_data_path
            }

        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise customexception(e, sys)
        
if __name__ == "__main__":
    ingestion = DataIngestion()
    paths = ingestion.initiate_data_ingestion()
    print(f"Files created: {paths}")