# promote_model.py
import os
import mlflow
import yaml
import logging
import pandas as pd
from dotenv import load_dotenv; load_dotenv()
from src.logger import logging
from src.exception import customexception


def load_config():
    """Load configuration from params.yaml"""
    try:
        with open("params.yaml", 'r') as f:
            config = yaml.safe_load(f)
        logging.info("✓ Configuration loaded successfully")
        return config
    except Exception as e:
        logging.error(f"✗ Failed to load config: {e}")
        raise


def get_model_performance_metrics(model_name, version):
    """Get performance metrics for a specific model version"""
    try:
        client = mlflow.MlflowClient()
        
        # Get the run ID for this model version
        model_version_details = client.get_model_version(model_name, version)
        run_id = model_version_details.run_id
        
        # Get the metrics from the run
        run = client.get_run(run_id)
        metrics = run.data.metrics
        
        return metrics
    except Exception as e:
        logging.warning(f"Could not fetch metrics for version {version}: {e}")
        return None


def should_promote_to_production(metrics, thresholds=None):
    """
    Check if model meets production criteria
    """
    if metrics is None:
        logging.warning("No metrics available, skipping performance check")
        return True
    
    if thresholds is None:
        thresholds = {
            'r2_score': 0.70,           # R² should be at least 0.70
            'mae': 50.0,                 # MAE should be at most 50 lakhs
            'accuracy_like': 60.0        # Accuracy-like metric at least 60%
        }
    
    # Check if model meets all thresholds
    meets_criteria = True
    criteria_details = []
    
    if 'r2_score' in metrics:
        r2 = metrics['r2_score']
        if r2 >= thresholds['r2_score']:
            criteria_details.append(f"✓ R²: {r2:.4f} >= {thresholds['r2_score']}")
        else:
            criteria_details.append(f"✗ R²: {r2:.4f} < {thresholds['r2_score']}")
            meets_criteria = False
    
    if 'mae' in metrics:
        mae = metrics['mae']
        if mae <= thresholds['mae']:
            criteria_details.append(f"✓ MAE: {mae:.2f} <= {thresholds['mae']}")
        else:
            criteria_details.append(f"✗ MAE: {mae:.2f} > {thresholds['mae']}")
            meets_criteria = False
    
    if 'accuracy_like' in metrics:
        accuracy = metrics['accuracy_like']
        if accuracy >= thresholds['accuracy_like']:
            criteria_details.append(f"✓ Accuracy: {accuracy:.2f}% >= {thresholds['accuracy_like']}%")
        else:
            criteria_details.append(f"✗ Accuracy: {accuracy:.2f}% < {thresholds['accuracy_like']}%")
            meets_criteria = False
    
    # Log criteria check results
    logging.info("Performance criteria check:")
    for detail in criteria_details:
        logging.info(f"  {detail}")
    
    return meets_criteria


def get_current_production_model(client, model_name):
    """Get current production model version"""
    try:
        # Fixed: Use double quotes for filter string
        prod_versions = client.search_model_versions(f'name="{model_name}" and stage="Production"')
        if prod_versions:
            return prod_versions[0].version
        return None
    except Exception as e:
        logging.warning(f"Could not get production model: {e}")
        return None


def archive_production_model(client, model_name, production_version):
    """Archive the current production model"""
    if production_version:
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=production_version,
                stage="Archived"
            )
            logging.info(f"✓ Model version {production_version} archived (was Production)")
            return True
        except Exception as e:
            logging.error(f"✗ Failed to archive production model: {e}")
            return False
    return True


def promote_model_to_production(model_name, version, description=None):
    """Promote a specific model version to production"""
    try:
        client = mlflow.MlflowClient()
        
        # Transition the model to production
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
        # Add description if provided
        if description:
            client.update_model_version(
                name=model_name,
                version=version,
                description=description
            )
        
        logging.info(f"✓ Model version {version} promoted to Production")
        return True
    except Exception as e:
        logging.error(f"✗ Failed to promote model: {e}")
        return False


def format_metric(value):
    """Helper function to format metrics safely"""
    if value is None or value == 'N/A':
        return 'N/A'
    try:
        if isinstance(value, (int, float)):
            return f"{value:.4f}"
        else:
            return str(value)
    except:
        return str(value)


def promote_model():
    """
    Main function to promote model from Staging to Production
    """
    try:
        # Load configuration
        config = load_config()
        
        # Get model name from config
        model_name = config['mlflow']['registered_model_name']
        
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")
        
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        
        # Set up MLflow tracking URI
        dagshub_url = "https://dagshub.com"
        repo_owner = "iamprashantjain"  # Replace with your DagsHub username
        repo_name = "house_price_prediction"  # Replace with your repo name
        
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
        logging.info(f"✓ MLflow tracking URI set: {dagshub_url}/{repo_owner}/{repo_name}.mlflow")
        
        client = mlflow.MlflowClient()
        
        # Get the latest version in Staging - Fixed: Use double quotes
        logging.info(f"Fetching latest Staging version for model: {model_name}")
        staging_versions = client.search_model_versions(f'name="{model_name}" and stage="Staging"')
        
        if not staging_versions:
            logging.warning(f"No model found in Staging stage for {model_name}")
            logging.info("Available stages:")
            all_versions = client.search_model_versions(f'name="{model_name}"')
            for version in all_versions:
                logging.info(f"  Version {version.version}: Stage '{version.current_stage}'")
            return False
        
        latest_version_staging = staging_versions[0].version
        logging.info(f"Latest Staging version: {latest_version_staging}")
        
        # Get performance metrics for the staging model
        logging.info("Fetching performance metrics for staging model...")
        staging_metrics = get_model_performance_metrics(model_name, latest_version_staging)
        
        # Check if model meets production criteria
        logging.info("Evaluating model against production criteria...")
        meets_criteria = should_promote_to_production(staging_metrics)
        
        if not meets_criteria:
            logging.warning("Model does not meet production criteria. Promotion cancelled.")
            logging.info("You can still force promotion by setting force=True")
            return False
        
        # Get current production model - Fixed: Use double quotes
        current_production = get_current_production_model(client, model_name)
        if current_production:
            logging.info(f"Current Production version: {current_production}")
        
        # Archive current production model
        logging.info("Archiving current production model...")
        if not archive_production_model(client, model_name, current_production):
            logging.warning("Failed to archive production model, but continuing with promotion...")
        
        # Promote the new model to production - Fixed formatting issue
        description = f"Promoted to Production on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Safely format metrics
        if staging_metrics:
            r2_value = staging_metrics.get('r2_score', 'N/A')
            mae_value = staging_metrics.get('mae', 'N/A')
            
            r2_str = format_metric(r2_value)
            mae_str = format_metric(mae_value)
            
            description += f" | Metrics - R²: {r2_str}, MAE: {mae_str}"
        
        success = promote_model_to_production(model_name, latest_version_staging, description)
        
        if success:
            logging.info("=" * 60)
            logging.info(f"✅ SUCCESS: Model {model_name} version {latest_version_staging} promoted to Production")
            logging.info("=" * 60)
            
            # Log summary
            logging.info("\n📊 Promotion Summary:")
            logging.info(f"  • Model: {model_name}")
            logging.info(f"  • Version: {latest_version_staging}")
            logging.info(f"  • Previous Production: {current_production if current_production else 'None'}")
            logging.info(f"  • New Production: {latest_version_staging}")
            
            if staging_metrics:
                logging.info(f"  • Performance Metrics:")
                for key, value in staging_metrics.items():
                    if isinstance(value, (int, float)):
                        logging.info(f"    - {key}: {value:.4f}")
                    else:
                        logging.info(f"    - {key}: {value}")
            
            return True
        else:
            logging.error("Failed to promote model to production")
            return False
            
    except Exception as e:
        logging.error(f"Error in promote_model: {e}")
        raise


def force_promote_model():
    """
    Force promote model without performance checks
    """
    try:
        config = load_config()
        model_name = config['mlflow']['registered_model_name']
        
        # Setup MLflow (same as above)
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")
        
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        
        dagshub_url = "https://dagshub.com"
        repo_owner = "iamprashantjain"
        repo_name = "house_price_prediction"
        
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
        
        client = mlflow.MlflowClient()
        
        # Get latest staging version - Fixed: Use double quotes
        staging_versions = client.search_model_versions(f'name="{model_name}" and stage="Staging"')
        if not staging_versions:
            logging.error("No staging version found")
            return False
        
        latest_version_staging = staging_versions[0].version
        
        # Force promotion without checks
        logging.warning(f"Force promoting version {latest_version_staging} to Production")
        
        # Archive current production - Fixed: Use double quotes
        prod_versions = client.search_model_versions(f'name="{model_name}" and stage="Production"')
        for version in prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )
            logging.info(f"Archived version {version.version}")
        
        # Promote to production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version_staging,
            stage="Production"
        )
        
        logging.info(f"✅ Force promoted version {latest_version_staging} to Production")
        return True
        
    except Exception as e:
        logging.error(f"Error in force_promote_model: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Promote model to production')
    parser.add_argument('--force', action='store_true', 
                       help='Force promotion without performance checks')
    parser.add_argument('--version', type=str, 
                       help='Specific version to promote (default: latest staging)')
    
    args = parser.parse_args()
    
    if args.force:
        logging.info("Force promotion mode enabled")
        success = force_promote_model()
    else:
        success = promote_model()
    
    # Exit with appropriate code
    exit(0 if success else 1)