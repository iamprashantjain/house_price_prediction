import yaml
import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import customexception

@dataclass
class DataTransformationConfig:
    train_data_path: str = None
    test_data_path: str = None
    processed_data_path: str = None
    selected_features: list = None
    target_column: str = None
    test_size: float = None
    random_state: int = None
    age_mapping: dict = None
    facing_mapping: dict = None
    default_age_years: float = None
    default_facing_code: int = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str = "params.yaml"):
        try:
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
            
            data_config = config.get('data', {})
            split_config = config.get('data_split', {})
            features_config = config.get('features', {})
            fe_config = config.get('feature_engineering', {})
            
            return cls(
                train_data_path=data_config.get('train_data_path', "data/train.csv"),
                test_data_path=data_config.get('test_data_path', "data/test.csv"),
                processed_data_path=data_config.get('processed_data_path', "data/processed/"),
                selected_features=features_config.get('selected_features', []),
                target_column=features_config.get('target_column', 'price'),
                test_size=split_config.get('test_size', 0.2),
                random_state=split_config.get('random_state', 42),
                age_mapping=fe_config.get('age_mapping', {}),
                facing_mapping=fe_config.get('facing_mapping', {}),
                default_age_years=fe_config.get('default_age_years', 5),
                default_facing_code=fe_config.get('default_facing_code', 0)
            )
        except Exception as e:
            logging.error(f"Error loading params.yaml: {str(e)}")

class DataTransformation:
    def __init__(self, yaml_config_path: str = "params.yaml"):
        self.config = DataTransformationConfig.from_yaml(yaml_config_path)
    
    def fill_missing_values(self, df):
        """Fill missing values in categorical columns"""
        # Fill facing column with mode
        if 'facing' in df.columns:
            df['facing'] = df['facing'].fillna(df['facing'].mode()[0])
        
        # Fill agePossession with mode
        if 'agePossession' in df.columns:
            df['agePossession'] = df['agePossession'].fillna(df['agePossession'].mode()[0])
        
        return df
    
    def engineer_features(self, df):
        """Create new features"""
        # Create price_per_sqft
        df['price_per_sqft'] = df['price'] / df['area_sqft']
        
        # Create total_rooms
        df['total_rooms'] = df['bedroom_num'] + df['bathroom_num']
        
        # Create room_bath_ratio
        df['room_bath_ratio'] = df['bedroom_num'] / df['bathroom_num']
        
        # Encode age categories
        df['age_years'] = df['agePossession'].map(self.config.age_mapping).fillna(self.config.default_age_years)
        
        # Encode facing direction
        df['facing_code'] = df['facing'].map(self.config.facing_mapping).fillna(self.config.default_facing_code)
        
        # Encode property type
        df['property_type_code'] = (df['property_type'] == 'house').astype(int)
        
        return df
    
    def select_features(self, df):
        """Select only the required features"""
        # Ensure all selected features exist
        available_features = [col for col in self.config.selected_features if col in df.columns]
        missing_features = set(self.config.selected_features) - set(available_features)
        
        if missing_features:
            logging.warning(f"Missing features: {missing_features}")
        
        # Include target column
        features_with_target = available_features + [self.config.target_column]
        
        return df[features_with_target]
    
    def impute_and_split(self, df):
        """Impute remaining NaN values and split data"""
        # Separate features and target
        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]
        
        # Impute NaN values with median
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            shuffle=True
        )
        
        logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save processed data as numpy arrays"""
        os.makedirs(self.config.processed_data_path, exist_ok=True)
        
        np.save(os.path.join(self.config.processed_data_path, 'X_train.npy'), X_train)
        np.save(os.path.join(self.config.processed_data_path, 'X_test.npy'), X_test)
        np.save(os.path.join(self.config.processed_data_path, 'y_train.npy'), y_train)
        np.save(os.path.join(self.config.processed_data_path, 'y_test.npy'), y_test)
        
        logging.info(f"Processed data saved to {self.config.processed_data_path}")
        
        return {
            'X_train_path': os.path.join(self.config.processed_data_path, 'X_train.npy'),
            'X_test_path': os.path.join(self.config.processed_data_path, 'X_test.npy'),
            'y_train_path': os.path.join(self.config.processed_data_path, 'y_train.npy'),
            'y_test_path': os.path.join(self.config.processed_data_path, 'y_test.npy')
        }
    
    def initiate_data_transformation(self, cleaned_data_path: str):
        logging.info("Data transformation started")
        try:
            # Load cleaned data
            df = pd.read_csv(cleaned_data_path)
            logging.info(f"Loaded cleaned data with shape: {df.shape}")
            
            # Fill missing values
            df = self.fill_missing_values(df)
            
            # Engineer features
            df = self.engineer_features(df)
            
            # Select features
            df = self.select_features(df)
            logging.info(f"Selected features shape: {df.shape}")
            
            # Impute and split
            X_train, X_test, y_train, y_test = self.impute_and_split(df)
            
            # Save processed data
            paths = self.save_processed_data(X_train, X_test, y_train, y_test)
            
            # Also save as CSV for reference
            train_df = pd.DataFrame(X_train, columns=X_train.columns)
            train_df['price'] = y_train.values
            train_df.to_csv(self.config.train_data_path, index=False)
            
            test_df = pd.DataFrame(X_test, columns=X_test.columns)
            test_df['price'] = y_test.values
            test_df.to_csv(self.config.test_data_path, index=False)
            
            logging.info("Data transformation completed successfully")
            
            return paths
            
        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise customexception(e, sys)

if __name__ == "__main__":
    transformer = DataTransformation()
    paths = transformer.initiate_data_transformation("data/cleaned.csv")
    print(f"Processed data saved to: {paths}")