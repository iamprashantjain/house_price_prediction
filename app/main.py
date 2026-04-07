# app/main.py
import os
import sys
import mlflow
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv
from datetime import datetime
import joblib
import yaml
from sklearn.impute import SimpleImputer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import logging
from src.exception import customexception

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using MLflow production model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
def load_config():
    try:
        with open("params.yaml", 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.warning(f"Could not load params.yaml: {e}")
        return None

config = load_config()

# Define the input data model based on your project's features
class HouseFeatures(BaseModel):
    """House features for prediction matching your data transformation"""
    area_sqft: float = Field(..., description="Area in square feet", gt=0)
    bedroom_num: float = Field(..., description="Number of bedrooms", ge=1, le=10)
    bathroom_num: float = Field(..., description="Number of bathrooms", ge=1, le=10)
    balcony_num: float = Field(..., description="Number of balconies", ge=0, le=10)
    property_type: str = Field(..., description="Property type: flat or house")
    facing: Optional[str] = Field(None, description="Direction facing: North, South, East, West")
    agePossession: Optional[str] = Field(None, description="Age of possession")
    floor_number: Optional[float] = Field(None, description="Floor number")
    total_floors: Optional[float] = Field(None, description="Total floors in building")
    feature_count: Optional[float] = Field(0, description="Number of features")
    furnish_count: Optional[float] = Field(0, description="Number of furnishing details")
    safety_rating: Optional[float] = Field(None, description="Safety rating", ge=0, le=5)
    lifestyle_rating: Optional[float] = Field(None, description="Lifestyle rating", ge=0, le=5)
    green_area_rating: Optional[float] = Field(None, description="Green area rating", ge=0, le=5)
    amenities_rating: Optional[float] = Field(None, description="Amenities rating", ge=0, le=5)
    rate_per_sqft: Optional[float] = Field(None, description="Rate per square foot")
    
    @validator('property_type')
    def validate_property_type(cls, v):
        if v.lower() not in ['flat', 'house']:
            raise ValueError('property_type must be either "flat" or "house"')
        return v.lower()
    
    class Config:
        schema_extra = {
            "example": {
                "area_sqft": 1500,
                "bedroom_num": 3,
                "bathroom_num": 2,
                "balcony_num": 2,
                "property_type": "flat",
                "facing": "North",
                "agePossession": "5-10 years",
                "floor_number": 3,
                "total_floors": 10,
                "feature_count": 5,
                "furnish_count": 3,
                "safety_rating": 4.5,
                "lifestyle_rating": 4.0,
                "green_area_rating": 3.5,
                "amenities_rating": 4.2,
                "rate_per_sqft": 5000
            }
        }

class BatchHouseFeatures(BaseModel):
    """Batch of house features for prediction"""
    houses: List[HouseFeatures]

class PredictionResponse(BaseModel):
    """Prediction response"""
    predicted_price_lakhs: float
    predicted_price_crores: float
    model_version: str
    model_stage: str
    timestamp: str
    features_used: Dict[str, Any]

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_predictions: int
    average_price_lakhs: float
    min_price_lakhs: float
    max_price_lakhs: float
    model_version: str
    timestamp: str

# Preprocessor Class matching your data_transformation.py
class FeaturePreprocessor:
    def __init__(self, config):
        self.config = config
        self.feature_columns = None
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
        # Feature engineering mappings from params.yaml
        self.age_mapping = config.get('feature_engineering', {}).get('age_mapping', {
            '0-1 years': 0.5, '1-3 years': 2, '3-5 years': 4,
            '5-10 years': 7.5, '10+ years': 15
        }) if config else {}
        
        self.facing_mapping = config.get('feature_engineering', {}).get('facing_mapping', {
            'North': 0, 'South': 1, 'East': 2, 'West': 3
        }) if config else {}
        
        self.selected_features = config.get('features', {}).get('selected_features', []) if config else []
    
    def engineer_features(self, df):
        """Create new features matching data_transformation.py"""
        df = df.copy()
        
        # Create price_per_sqft (will be computed if not provided)
        if 'rate_per_sqft' in df.columns and df['rate_per_sqft'].notna().any():
            df['price_per_sqft'] = df['rate_per_sqft']
        else:
            # Estimate from existing data or use median
            df['price_per_sqft'] = df.get('rate_per_sqft', 5000)
        
        # Create total_rooms
        df['total_rooms'] = df['bedroom_num'] + df['bathroom_num']
        
        # Create room_bath_ratio
        df['room_bath_ratio'] = df['bedroom_num'] / (df['bathroom_num'].replace(0, 1))
        
        # Encode age categories
        df['age_years'] = df['agePossession'].map(self.age_mapping).fillna(5)
        
        # Encode facing direction
        df['facing_code'] = df['facing'].map(self.facing_mapping).fillna(0)
        
        # Encode property type
        df['property_type_code'] = (df['property_type'] == 'house').astype(int)
        
        # Fill missing values in rating columns
        rating_cols = ['safety_rating', 'lifestyle_rating', 'green_area_rating', 'amenities_rating']
        for col in rating_cols:
            if col in df.columns:
                df[col] = df[col].fillna(3.0)  # Default medium rating
        
        # Fill other numeric columns
        numeric_cols = ['floor_number', 'total_floors', 'feature_count', 'furnish_count']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        return df
    
    def select_features(self, df):
        """Select only the required features"""
        if self.selected_features:
            # Get available features
            available_features = [col for col in self.selected_features if col in df.columns]
            return df[available_features]
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete preprocessing pipeline"""
        # Engineer features
        df = self.engineer_features(df)
        
        # Select features
        df = self.select_features(df)
        
        # Impute any remaining NaN values
        if not self.is_fitted:
            self.imputer.fit(df)
            self.is_fitted = True
        
        df_imputed = pd.DataFrame(
            self.imputer.transform(df),
            columns=df.columns
        )
        
        return df_imputed

# Model Manager Class
class ModelManager:
    def __init__(self):
        self.model = None
        self.model_version = None
        self.model_stage = None
        self.is_loaded = False
        self.preprocessor = None
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setup MLflow tracking from your project"""
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            logging.warning("DAGSHUB_PAT environment variable is not set")
            return
        
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        
        # Get tracking URI from config or use default
        if config and 'mlflow' in config and 'tracking_uri' in config['mlflow']:
            mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        else:
            dagshub_url = "https://dagshub.com"
            repo_owner = "iamprashantjain"
            repo_name = "house_price_prediction"
            mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
        
        logging.info("MLflow tracking configured")
    
    def load_production_model(self):
        """Load the production model from MLflow model registry"""
        try:
            if not config:
                return self.load_local_model()
            
            model_name = config.get('mlflow', {}).get('registered_model_name', 'house_price_model')
            
            # Get production model
            client = mlflow.MlflowClient()
            
            # Try Production stage first
            production_versions = client.get_latest_versions(model_name, stages=["Production"])
            
            if production_versions:
                self.model_version = production_versions[0].version
                self.model_stage = "Production"
                model_uri = f"models:/{model_name}/{self.model_version}"
                self.model = mlflow.pyfunc.load_model(model_uri)
                self.is_loaded = True
                logging.info(f"✅ Production model loaded: {model_name} v{self.model_version}")
                return True
            
            # Fallback to Staging
            staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
            if staging_versions:
                self.model_version = staging_versions[0].version
                self.model_stage = "Staging"
                model_uri = f"models:/{model_name}/{self.model_version}"
                self.model = mlflow.pyfunc.load_model(model_uri)
                self.is_loaded = True
                logging.warning(f"⚠ Using Staging model: v{self.model_version}")
                return True
            
            # Fallback to latest version
            all_versions = client.search_model_versions(f"name='{model_name}'")
            if all_versions:
                latest = max(all_versions, key=lambda x: int(x.version))
                self.model_version = latest.version
                self.model_stage = latest.current_stage or "None"
                model_uri = f"models:/{model_name}/{self.model_version}"
                self.model = mlflow.pyfunc.load_model(model_uri)
                self.is_loaded = True
                logging.warning(f"⚠ Using latest model: v{self.model_version} (stage: {self.model_stage})")
                return True
                
        except Exception as e:
            logging.error(f"Failed to load model from MLflow: {e}")
            return self.load_local_model()
    
    def load_local_model(self):
        """Fallback to local model if MLflow fails"""
        try:
            if config and 'model' in config and 'save_path' in config['model']:
                local_model_path = config['model']['save_path']
                self.model = joblib.load(local_model_path)
                self.model_version = "local"
                self.model_stage = "Local"
                self.is_loaded = True
                logging.info(f"✅ Local model loaded from: {local_model_path}")
                return True
            else:
                # Try default path
                default_path = "models/model.pkl"
                if os.path.exists(default_path):
                    self.model = joblib.load(default_path)
                    self.model_version = "local"
                    self.model_stage = "Local"
                    self.is_loaded = True
                    logging.info(f"✅ Local model loaded from: {default_path}")
                    return True
                return False
        except Exception as e:
            logging.error(f"Failed to load local model: {e}")
            return False
    
    def initialize_preprocessor(self):
        """Initialize the feature preprocessor"""
        try:
            self.preprocessor = FeaturePreprocessor(config)
            logging.info("✅ Preprocessor initialized")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize preprocessor: {e}")
            return False
    
    def predict(self, features: Dict) -> float:
        """Make prediction"""
        if not self.is_loaded:
            raise Exception("Model not loaded")
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Preprocess features
        if self.preprocessor:
            df = self.preprocessor.preprocess(df)
        
        # Make prediction
        prediction = self.model.predict(df)
        
        return float(prediction[0])
    
    def predict_batch(self, features_list: List[Dict]) -> List[float]:
        """Make batch predictions"""
        if not self.is_loaded:
            raise Exception("Model not loaded")
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        # Preprocess features
        if self.preprocessor:
            df = self.preprocessor.preprocess(df)
        
        # Make predictions
        predictions = self.model.predict(df)
        
        return [float(pred) for pred in predictions]

# Initialize model manager
model_manager = ModelManager()

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model and initialize preprocessor on startup"""
    logging.info("Starting up API...")
    
    # Load model
    success = model_manager.load_production_model()
    if success:
        logging.info("✅ Model loaded successfully")
    else:
        logging.error("❌ Failed to load model")
    
    # Initialize preprocessor
    model_manager.initialize_preprocessor()

# Health check endpoint
@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_loaded,
        "model_version": model_manager.model_version,
        "model_stage": model_manager.model_stage,
        "timestamp": datetime.now().isoformat()
    }

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: HouseFeatures):
    """
    Predict house price based on features
    """
    try:
        # Check if model is loaded
        if not model_manager.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please try again later."
            )
        
        # Convert to dictionary
        features_dict = features.dict()
        
        # Make prediction (price in lakhs)
        predicted_price_lakhs = model_manager.predict(features_dict)
        
        # Prepare response
        response = PredictionResponse(
            predicted_price_lakhs=round(predicted_price_lakhs, 2),
            predicted_price_crores=round(predicted_price_lakhs / 100, 2),
            model_version=model_manager.model_version,
            model_stage=model_manager.model_stage,
            timestamp=datetime.now().isoformat(),
            features_used=features_dict
        )
        
        logging.info(f"Prediction made: {predicted_price_lakhs:.2f} lakhs")
        return response
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchHouseFeatures):
    """
    Predict house prices for multiple properties
    """
    try:
        # Check if model is loaded
        if not model_manager.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please try again later."
            )
        
        # Convert to dictionaries
        features_list = [house.dict() for house in batch.houses]
        
        # Make batch predictions
        predictions = model_manager.predict_batch(features_list)
        
        # Prepare responses
        prediction_responses = []
        for features, price in zip(features_list, predictions):
            prediction_responses.append(
                PredictionResponse(
                    predicted_price_lakhs=round(price, 2),
                    predicted_price_crores=round(price / 100, 2),
                    model_version=model_manager.model_version,
                    model_stage=model_manager.model_stage,
                    timestamp=datetime.now().isoformat(),
                    features_used=features
                )
            )
        
        # Calculate statistics
        prices = [p for p in predictions]
        
        response = BatchPredictionResponse(
            predictions=prediction_responses,
            total_predictions=len(predictions),
            average_price_lakhs=round(sum(prices) / len(prices), 2) if prices else 0,
            min_price_lakhs=round(min(prices), 2) if prices else 0,
            max_price_lakhs=round(max(prices), 2) if prices else 0,
            model_version=model_manager.model_version,
            timestamp=datetime.now().isoformat()
        )
        
        logging.info(f"Batch prediction: {len(predictions)} predictions made")
        return response
        
    except Exception as e:
        logging.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

# Model info endpoint
@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get information about the current model"""
    return {
        "model_loaded": model_manager.is_loaded,
        "model_version": model_manager.model_version,
        "model_stage": model_manager.model_stage,
        "model_type": type(model_manager.model).__name__ if model_manager.model else None,
        "preprocessor_initialized": model_manager.preprocessor is not None,
        "selected_features": model_manager.preprocessor.selected_features if model_manager.preprocessor else None
    }

# Reload model endpoint (for admin use)
@app.post("/model/reload", tags=["Admin"])
async def reload_model():
    """Reload the production model (admin endpoint)"""
    try:
        model_manager.is_loaded = False
        success = model_manager.load_production_model()
        if success:
            model_manager.initialize_preprocessor()
            return {
                "status": "success",
                "message": "Model reloaded successfully",
                "model_version": model_manager.model_version,
                "model_stage": model_manager.model_stage
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reload model"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )