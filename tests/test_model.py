import unittest
import mlflow
import os
import sys
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv

load_dotenv()


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # Load configuration
        with open("params.yaml", 'r') as f:
            cls.config = yaml.safe_load(f)

        dagshub_url = "https://dagshub.com"
        repo_owner = "iamprashantjain"  # Replace with your DagsHub username
        repo_name = "house_price_prediction"  # Replace with your repo name

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the model from MLflow model registry
        cls.new_model_name = cls.config['mlflow']['registered_model_name']
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        
        if cls.new_model_version:
            cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
            cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)
            print(f"✓ Model loaded from registry: {cls.new_model_name} v{cls.new_model_version}")
        else:
            # Fallback to local model if registry not available
            print("⚠ No model found in registry, loading local model...")
            local_model_path = cls.config['model']['save_path']
            cls.new_model = joblib.load(local_model_path)
            cls.new_model_version = "local"
            print(f"✓ Local model loaded from: {local_model_path}")

        # Load test data (instead of vectorizer for house price prediction)
        cls.load_test_data()

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        try:
            client = mlflow.MlflowClient()
            latest_version = client.get_latest_versions(model_name, stages=[stage])
            return latest_version[0].version if latest_version else None
        except:
            return None

    @classmethod
    def load_test_data(cls):
        """Load holdout test data for house price prediction"""
        processed_data_path = cls.config['data']['processed_data_path']
        
        # Load test data
        cls.X_test = np.load(os.path.join(processed_data_path, 'X_test.npy'))
        cls.y_test = np.load(os.path.join(processed_data_path, 'y_test.npy'))
        
        # Load feature names if available
        feature_names_path = os.path.join(processed_data_path, 'feature_names.npy')
        if os.path.exists(feature_names_path):
            cls.feature_names = np.load(feature_names_path, allow_pickle=True)
        else:
            cls.feature_names = [f'feature_{i}' for i in range(cls.X_test.shape[1])]
        
        # Convert to DataFrame for easier handling (like in reference code)
        cls.test_data = pd.DataFrame(cls.X_test, columns=cls.feature_names)
        cls.test_data['price'] = cls.y_test
        
        print(f"✓ Test data loaded: {cls.X_test.shape[0]} samples, {cls.X_test.shape[1]} features")

    def test_model_loaded_properly(self):
        """Test 1: Verify model loaded successfully"""
        self.assertIsNotNone(self.new_model)
        print("✓ Test 1 passed: Model loaded properly")

    def test_model_signature(self):
        """Test 2: Verify model input and output signatures"""
        # Create dummy input based on expected feature shape
        dummy_input = np.random.randn(5, self.X_test.shape[1]).astype(np.float32)
        
        # Convert to DataFrame (matching the expected input format)
        dummy_df = pd.DataFrame(dummy_input, columns=self.feature_names)
        
        # Predict using the model to verify input and output shapes
        prediction = self.new_model.predict(dummy_df)
        
        # Verify the input shape matches number of features
        self.assertEqual(dummy_df.shape[1], self.X_test.shape[1], 
                        f"Expected {self.X_test.shape[1]} features, got {dummy_df.shape[1]}")
        
        # Verify the output shape (should be same as number of input samples)
        self.assertEqual(len(prediction), dummy_df.shape[0],
                        f"Expected {dummy_df.shape[0]} predictions, got {len(prediction)}")
        
        # Verify output is 1D array (regression output)
        self.assertEqual(len(prediction.shape), 1,
                        "Expected 1D output array for regression")
        
        print(f"✓ Test 2 passed: Model signature verified (input: {dummy_df.shape}, output: {prediction.shape})")

    def test_model_performance(self):
        """Test 3: Verify model performance meets thresholds"""
        # Extract features and labels from holdout test data
        X_holdout = self.test_data.iloc[:, 0:-1]  # All columns except last (price)
        y_holdout = self.test_data.iloc[:, -1]     # Last column is price
        
        # Predict using the model
        y_pred_new = self.new_model.predict(X_holdout)
        
        # Calculate regression metrics for the model
        mae = mean_absolute_error(y_holdout, y_pred_new)
        rmse = np.sqrt(mean_squared_error(y_holdout, y_pred_new))
        r2 = r2_score(y_holdout, y_pred_new)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_holdout - y_pred_new) / y_holdout)) * 100
        
        # Calculate accuracy-like metric (1 - normalized MAE)
        accuracy_like = max(0, (1 - mae / y_holdout.mean()) * 100)
        
        # Define expected thresholds for the performance metrics
        expected_r2 = 0.70      # R² should be at least 0.70
        expected_mae = 50.0      # MAE should be at most 50 lakhs
        expected_accuracy = 60.0 # Accuracy-like metric should be at least 60%
        
        print(f"\n  📊 Model Performance Metrics:")
        print(f"     • MAE:  {mae:.2f} lakhs")
        print(f"     • RMSE: {rmse:.2f} lakhs")
        print(f"     • R²:   {r2:.4f}")
        print(f"     • MAPE: {mape:.2f}%")
        print(f"     • Accuracy-like: {accuracy_like:.2f}%")
        
        # Assert that the model meets the performance thresholds
        self.assertGreaterEqual(r2, expected_r2, 
                               f'R² should be at least {expected_r2}, got {r2:.4f}')
        self.assertLessEqual(mae, expected_mae,
                           f'MAE should be at most {expected_mae}, got {mae:.2f}')
        self.assertGreaterEqual(accuracy_like, expected_accuracy,
                               f'Accuracy should be at least {expected_accuracy}%, got {accuracy_like:.2f}%')
        
        print(f"\n✓ Test 3 passed: Model meets performance thresholds")

    def test_model_consistency(self):
        """Additional Test: Model produces same output for same input"""
        # Take first 10 samples
        X_sample = self.test_data.iloc[:10, 0:-1]
        
        # Predict twice
        pred1 = self.new_model.predict(X_sample)
        pred2 = self.new_model.predict(X_sample)
        
        # Check if predictions are identical
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=6,
                                            err_msg="Model predictions are not consistent")
        
        print("✓ Additional test passed: Model produces consistent predictions")

    def test_model_prediction_ranges(self):
        """Additional Test: Predictions are within reasonable ranges"""
        X_holdout = self.test_data.iloc[:, 0:-1]
        y_pred = self.new_model.predict(X_holdout)
        y_actual = self.test_data.iloc[:, -1]
        
        # Test 1: House prices should be positive
        self.assertTrue(np.all(y_pred >= 0), 
                       f"Found negative predictions: {y_pred[y_pred < 0]}")
        
        # Test 2: No unreasonable outliers (within 3 standard deviations of actual prices)
        actual_mean = y_actual.mean()
        actual_std = y_actual.std()
        reasonable_upper_bound = actual_mean + 3 * actual_std
        
        outliers = y_pred[y_pred > reasonable_upper_bound]
        outlier_percentage = len(outliers) / len(y_pred) * 100
        
        print(f"\n  📊 Price Range Analysis:")
        print(f"     • Actual price range: {y_actual.min():.2f} - {y_actual.max():.2f} lakhs")
        print(f"     • Predicted price range: {y_pred.min():.2f} - {y_pred.max():.2f} lakhs")
        print(f"     • Actual mean ± 3σ: {actual_mean:.2f} ± {3*actual_std:.2f} lakhs")
        print(f"     • Predictions above upper bound: {outlier_percentage:.1f}%")
        print(f"     • Actual 95th percentile: {np.percentile(y_actual, 95):.2f} lakhs")
        print(f"     • Predicted 95th percentile: {np.percentile(y_pred, 95):.2f} lakhs")
        
        # Allow up to 5% of predictions to be above reasonable upper bound
        self.assertLessEqual(outlier_percentage, 5,
                            f"Too many predictions ({outlier_percentage:.1f}%) above reasonable upper bound")
        
        # Test 3: Check if predictions correlate reasonably with actuals (not extremely different)
        ratio = y_pred / (y_actual + 1e-8)
        extreme_ratio_count = np.sum((ratio > 3) | (ratio < 0.33))
        extreme_ratio_percentage = extreme_ratio_count / len(y_pred) * 100
        
        print(f"     • Predictions with ratio >3x or <0.33x actual: {extreme_ratio_percentage:.1f}%")
        
        self.assertLessEqual(extreme_ratio_percentage, 10,
                            f"Too many predictions ({extreme_ratio_percentage:.1f}%) are extremely different from actual prices")
        
        # Test 4: 95th percentile of predictions should be within 30% of actual's 95th percentile
        percentile_95_pred = np.percentile(y_pred, 95)
        percentile_95_actual = np.percentile(y_actual, 95)
        
        self.assertLessEqual(percentile_95_pred, percentile_95_actual * 1.3,
                            f"95th percentile price ({percentile_95_pred:.2f}) exceeds actual 95th percentile ({percentile_95_actual:.2f}) by too much")
        
        print(f"✓ Additional test passed: Predictions are within reasonable ranges")


if __name__ == "__main__":
    # Run the tests with verbosity
    unittest.main(verbosity=2)