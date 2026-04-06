import yaml
import pandas as pd
import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import customexception

@dataclass
class DataIngestionConfig:
    merged_data_path: str = None
    flats_raw_data_path: str = None
    houses_raw_data_path: str = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str = "params.yaml"):
        """Load configuration from YAML file"""
        try:
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
            
            data_config = config.get('data', {})
            raw_paths = data_config.get('raw_data_paths', {})
            
            return cls(
                merged_data_path=data_config.get('merged_data_path', "data/raw.csv"),
                flats_raw_data_path=raw_paths.get('flats', "data/flats.csv"),
                houses_raw_data_path=raw_paths.get('houses', "data/houses.csv")
            )
        except Exception as e:
            logging.error(f"Error loading params.yaml: {str(e)}")

class DataIngestion:
    def __init__(self, yaml_config_path: str = "params.yaml"):
        self.ingestion_config = DataIngestionConfig.from_yaml(yaml_config_path)
        self.raw_data_path_1 = self.ingestion_config.flats_raw_data_path
        self.raw_data_path_2 = self.ingestion_config.houses_raw_data_path
    
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            # Read the two raw CSV files
            data1 = pd.read_csv(self.raw_data_path_1)
            data2 = pd.read_csv(self.raw_data_path_2)
            
            logging.info(f"Reading data from: {self.raw_data_path_1} (shape: {data1.shape})")
            logging.info(f"Reading data from: {self.raw_data_path_2} (shape: {data2.shape})")

            # Add 'property_type' column to distinguish between flats and houses
            data1['property_type'] = 'flat'
            data2['property_type'] = 'house'
           
            # Concatenate the two dataframes
            data = pd.concat([data1, data2], ignore_index=True)
            logging.info(f"Datasets concatenated successfully. Total shape: {data.shape}")
            
            # Save the merged data (NO train-test split here)
            os.makedirs(os.path.dirname(self.ingestion_config.merged_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.merged_data_path, index=False)
            logging.info(f"Merged data saved at: {self.ingestion_config.merged_data_path}")

            logging.info("Data ingestion completed successfully.")
            
            return self.ingestion_config.merged_data_path

        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise customexception(e, sys)
        
if __name__ == "__main__":
    ingestion = DataIngestion()
    data_path = ingestion.initiate_data_ingestion()
    print(f"Data saved to: {data_path}")