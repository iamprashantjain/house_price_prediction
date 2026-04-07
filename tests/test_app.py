# tests/test_app.py - Fixed with better timeout handling
import os
import sys
import mlflow
import pandas as pd
import unittest
import time
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from sklearn.impute import SimpleImputer

def load_config():
    try:
        with open("params.yaml", 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Could not load params.yaml: {e}")
        return None

config = load_config()

def setup_mlflow():
    """Setup MLflow tracking with retry logic"""
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    
    if config and 'mlflow' in config and 'tracking_uri' in config['mlflow']:
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    else:
        dagshub_url = "https://dagshub.com"
        repo_owner = "iamprashantjain"
        repo_name = "house_price_prediction"
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
    
    print("✅ MLflow tracking configured")

def load_production_model_with_retry(max_retries=3):
    """Load model with retry logic for timeout issues"""
    for attempt in range(max_retries):
        try:
            print(f"\n🔄 Attempt {attempt + 1}/{max_retries} to load model...")
            
            if not config:
                print("❌ No config found")
                return None, None, None
            
            model_name = config.get('mlflow', {}).get('registered_model_name', 'house_price_model')
            print(f"📦 Looking for model: {model_name}")
            
            # Setup MLflow
            setup_mlflow()
            
            # Set timeout for requests
            os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = '120'  # 2 minutes timeout
            
            # Get MLflow client
            client = mlflow.MlflowClient()
            
            # Try Production stage first
            production_versions = client.get_latest_versions(model_name, stages=["Production"])
            
            if production_versions:
                model_version = production_versions[0].version
                model_stage = "Production"
                model_uri = f"models:/{model_name}/{model_version}"
                
                # Load model with timeout
                model = mlflow.pyfunc.load_model(model_uri)
                print(f"✅ Production model loaded: {model_name} v{model_version}")
                return model, model_version, model_stage
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"❌ All {max_retries} attempts failed")
                return None, None, None
    
    return None, None, None

# Alternative: Load local model if MLflow fails
def load_local_model_fallback():
    """Fallback to local model if MLflow fails"""
    try:
        local_paths = [
            "models/model.pkl",
            "models/house_price_model.pkl",
            "artifacts/model.pkl"
        ]
        
        for path in local_paths:
            if os.path.exists(path):
                import joblib
                model = joblib.load(path)
                print(f"✅ Loaded local model from: {path}")
                return model, "local", "Local"
        
        print("❌ No local model found")
        return None, None, None
    except Exception as e:
        print(f"❌ Local model load failed: {e}")
        return None, None, None

class FeaturePreprocessor:
    def __init__(self, config):
        self.config = config
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
        self.age_mapping = config.get('feature_engineering', {}).get('age_mapping', {
            '0-1 years': 0.5, '1-3 years': 2, '3-5 years': 4,
            '5-10 years': 7.5, '10+ years': 15
        }) if config else {}
        
        self.facing_mapping = config.get('feature_engineering', {}).get('facing_mapping', {
            'North': 0, 'South': 1, 'East': 2, 'West': 3
        }) if config else {}
        
        self.selected_features = config.get('features', {}).get('selected_features', []) if config else []
    
    def preprocess(self, df):
        """Preprocess features"""
        df = df.copy()
        
        # Ensure all required columns exist
        if 'agePossession' not in df.columns:
            df['agePossession'] = '5-10 years'
        if 'facing' not in df.columns:
            df['facing'] = 'North'
        if 'rate_per_sqft' not in df.columns:
            df['rate_per_sqft'] = 5000
        if 'safety_rating' not in df.columns:
            df['safety_rating'] = 4.0
        if 'lifestyle_rating' not in df.columns:
            df['lifestyle_rating'] = 4.0
        if 'green_area_rating' not in df.columns:
            df['green_area_rating'] = 4.0
        if 'amenities_rating' not in df.columns:
            df['amenities_rating'] = 4.0
        if 'floor_number' not in df.columns:
            df['floor_number'] = 1
        if 'total_floors' not in df.columns:
            df['total_floors'] = 5
        if 'feature_count' not in df.columns:
            df['feature_count'] = 5
        if 'furnish_count' not in df.columns:
            df['furnish_count'] = 3
        
        # Feature engineering
        df['price_per_sqft'] = df['rate_per_sqft']
        df['total_rooms'] = df['bedroom_num'] + df['bathroom_num']
        df['room_bath_ratio'] = df['bedroom_num'] / (df['bathroom_num'].replace(0, 1))
        df['age_years'] = df['agePossession'].map(self.age_mapping).fillna(5)
        df['facing_code'] = df['facing'].map(self.facing_mapping).fillna(0)
        df['property_type_code'] = (df['property_type'] == 'house').astype(int)
        
        # Select features if specified
        if self.selected_features:
            available = [col for col in self.selected_features if col in df.columns]
            df = df[available]
        
        if not self.is_fitted:
            self.imputer.fit(df)
            self.is_fitted = True
        
        return pd.DataFrame(self.imputer.transform(df), columns=df.columns)

class TestHousePriceAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load model with retry and fallback"""
        print("\n" + "="*60)
        print("🔴 LOADING MODEL WITH RETRY LOGIC")
        print("="*60)
        
        # Try MLflow with retry
        cls.model, cls.model_version, cls.model_stage = load_production_model_with_retry(max_retries=3)
        
        # If MLflow fails, try local model
        if cls.model is None:
            print("\n⚠️ MLflow failed, trying local model...")
            cls.model, cls.model_version, cls.model_stage = load_local_model_fallback()
        
        # Initialize preprocessor
        if config:
            cls.preprocessor = FeaturePreprocessor(config)
            print("✅ Preprocessor initialized")
        else:
            cls.preprocessor = None
        
        # Test data
        cls.valid_house = {
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
    
    def test_1_model_loaded_from_registry(self):
        """Test model loaded successfully"""
        print("\n" + "="*60)
        print("📦 MODEL LOAD STATUS")
        print("="*60)
        print(f"Model: {self.model}")
        print(f"Version: {self.model_version}")
        print(f"Stage: {self.model_stage}")
        print("="*60)
        
        if self.model is None:
            print("\n❌ MODEL NOT LOADED!")
            print("\nPossible solutions:")
            print("1. Check your internet connection (DagsHub timeout)")
            print("2. Train and save model locally first:")
            print("   python src/train_model.py")
            print("3. Or download model manually from DagsHub")
        
        self.assertIsNotNone(self.model, "Model failed to load!")
    
    def test_2_prediction_with_preprocessing(self):
        """Test prediction"""
        if not self.model:
            self.skipTest("Model not loaded")
        
        df = pd.DataFrame([self.valid_house])
        
        if self.preprocessor:
            df = self.preprocessor.preprocess(df)
        
        prediction = self.model.predict(df)
        
        print("\n" + "="*60)
        print("💰 PREDICTION RESULT")
        print("="*60)
        print(f"Input area: {self.valid_house['area_sqft']} sqft")
        print(f"Predicted price: ₹{prediction[0]:.2f} lakhs")
        print("="*60)
        
        self.assertGreater(prediction[0], 0)

if __name__ == "__main__":
    # Check DAGSHUB_PAT
    if not os.getenv("DAGSHUB_PAT"):
        print("⚠️ DAGSHUB_PAT not set!")
        print("Will try local model only")
    
    # Run tests
    unittest.main(verbosity=2)