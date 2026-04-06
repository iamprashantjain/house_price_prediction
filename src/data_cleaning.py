import yaml
import pandas as pd
import numpy as np
import re
import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import customexception

@dataclass
class DataCleaningConfig:
    cleaned_data_path: str = None
    drop_columns: list = None
    rating_columns: list = None
    fillna_strategy: str = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str = "params.yaml"):
        try:
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
            
            data_config = config.get('data', {})
            cleaning_config = config.get('cleaning', {})
            
            return cls(
                cleaned_data_path=data_config.get('cleaned_data_path', "data/cleaned.csv"),
                drop_columns=cleaning_config.get('drop_columns', []),
                rating_columns=cleaning_config.get('rating_columns', []),
                fillna_strategy=cleaning_config.get('fillna_strategy', 'mode')
            )
        except Exception as e:
            logging.error(f"Error loading params.yaml: {str(e)}")

class DataCleaning:
    def __init__(self, yaml_config_path: str = "params.yaml"):
        self.config = DataCleaningConfig.from_yaml(yaml_config_path)
    
    def clean_price(self, price_str):
        """Convert price string to numeric value in lakhs"""
        if pd.isna(price_str):
            return np.nan
        price_str = str(price_str).lower().strip()
        
        if 'cr' in price_str:
            num = float(re.findall(r'[\d.]+', price_str)[0])
            return num * 100
        elif 'lac' in price_str:
            num = float(re.findall(r'[\d.]+', price_str)[0])
            return num
        else:
            numbers = re.findall(r'[\d.]+', price_str)
            if numbers:
                return float(numbers[0])
        return np.nan
    
    def clean_area(self, area_str):
        """Extract area in sq ft"""
        if pd.isna(area_str):
            return np.nan
        
        area_str = str(area_str).lower()
        sqft_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:sq\.?\s*ft\.?|sqft)', area_str)
        if sqft_match:
            return float(sqft_match.group(1))
        
        sqm_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:sq\.?\s*m\.?|sqm)', area_str)
        if sqm_match:
            return float(sqm_match.group(1)) * 10.764
        
        return np.nan
    
    def clean_rate(self, rate_str):
        """Extract rate per sq ft"""
        if pd.isna(rate_str):
            return np.nan
        rate_str = str(rate_str).lower()
        match = re.search(r'[\d,]+(?:\.\d+)?', rate_str.replace(',', ''))
        if match:
            return float(match.group())
        return np.nan
    
    def extract_floor_number(self, floor_str):
        """Extract floor number"""
        if pd.isna(floor_str):
            return np.nan
        floor_str = str(floor_str).lower()
        match = re.search(r'(\d+)(?:st|nd|rd|th)?\s*floor', floor_str)
        if match:
            return int(match.group(1))
        return np.nan
    
    def extract_total_floors(self, floor_str):
        """Extract total floors"""
        if pd.isna(floor_str):
            return np.nan
        floor_str = str(floor_str).lower()
        match = re.search(r'out of (\d+)', floor_str)
        if match:
            return int(match.group(1))
        return np.nan
    
    def count_features(self, feature_list):
        """Count number of features"""
        if pd.isna(feature_list):
            return 0
        if isinstance(feature_list, list):
            return len(feature_list)
        return 0
    
    def extract_rating(self, rating_list, category):
        """Extract specific rating"""
        if pd.isna(rating_list) or not isinstance(rating_list, list):
            return np.nan
        
        for item in rating_list:
            if category.lower() in item.lower():
                match = re.search(r'(\d+(?:\.\d+)?)\s*out of 5', item)
                if match:
                    return float(match.group(1))
        return np.nan
    
    def initiate_data_cleaning(self, input_path: str):
        logging.info("Data cleaning started")
        try:
            # Load raw data
            df = pd.read_csv(input_path)
            logging.info(f"Loaded raw data with shape: {df.shape}")
            
            # Clean price column
            df['price'] = df['price'].apply(self.clean_price)
            
            # Clean area column
            df['area_sqft'] = df['areaWithType'].apply(self.clean_area)
            
            # Clean rate column
            df['rate_per_sqft'] = df['rate'].apply(self.clean_rate)
            
            # Extract numeric values
            df['floor_number'] = df['floorNum'].apply(self.extract_floor_number)
            df['total_floors'] = df['floorNum'].apply(self.extract_total_floors)
            
            # Convert room counts to numeric
            df['bedroom_num'] = df['bedRoom'].str.extract(r'(\d+)').astype(float)
            df['bathroom_num'] = df['bathroom'].str.extract(r'(\d+)').astype(float)
            df['balcony_num'] = df['balcony'].str.extract(r'(\d+)').astype(float)
            
            # Count features
            df['feature_count'] = df['features'].apply(self.count_features)
            df['furnish_count'] = df['furnishDetails'].apply(self.count_features)
            
            # Extract ratings
            df['safety_rating'] = df['rating'].apply(lambda x: self.extract_rating(x, 'safety'))
            df['lifestyle_rating'] = df['rating'].apply(lambda x: self.extract_rating(x, 'lifestyle'))
            df['green_area_rating'] = df['rating'].apply(lambda x: self.extract_rating(x, 'green'))
            df['amenities_rating'] = df['rating'].apply(lambda x: self.extract_rating(x, 'amenities'))
            
            # Create additional features
            df['calculated_rate'] = (df['price'] * 100000) / df['area_sqft']
            df['is_high_value'] = (df['price'] > df['price'].median()).astype(int)
            
            # Drop unwanted columns
            columns_to_drop = [col for col in self.config.drop_columns if col in df.columns]
            df = df.drop(columns=columns_to_drop, errors='ignore')
            
            # Drop rating columns if they exist
            rating_cols = [col for col in self.config.rating_columns if col in df.columns]
            if rating_cols:
                df = df.drop(columns=rating_cols)
                logging.info(f"Dropped rating columns: {rating_cols}")
            
            # Drop rows with null prices
            df = df.dropna(subset=['price'])
            logging.info(f"Shape after dropping null prices: {df.shape}")
            
            # Save cleaned data
            os.makedirs(os.path.dirname(self.config.cleaned_data_path), exist_ok=True)
            df.to_csv(self.config.cleaned_data_path, index=False)
            logging.info(f"Cleaned data saved to {self.config.cleaned_data_path}")
            
            return self.config.cleaned_data_path
            
        except Exception as e:
            logging.error(f"Error during data cleaning: {str(e)}")
            raise customexception(e, sys)

if __name__ == "__main__":
    cleaner = DataCleaning()
    cleaned_path = cleaner.initiate_data_cleaning("data/raw.csv")
    print(f"Cleaned data saved to: {cleaned_path}")