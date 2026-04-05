import os
from pathlib import Path  # Importing Path to create OS-compatible paths
import logging

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of files and directories to create
list_of_files = [
    ".github/workflows/.gitkeep",          # GitHub Actions workflows directory
    "experiment/experiments.ipynb",        # Jupyter Notebook for experimentation
    "src/__init__.py",                      # Init file for the src package
    
    # Components for various stages of the project
    "src/components/__init__.py",          # Init file for components package
    "src/components/data_ingestion.py",    # Data ingestion script
    "src/components/data_transformation.py",# Data transformation script
    "src/components/model_trainer.py",     # Model training script
    "src/components/model_evaluation.py",  # Model evaluation script
        
    # Utilities for common functions
    "src/utils/__init__.py",
    "src/utils/utils.py",                   # Utilities script
    "src/logger/__init__.py",
    "src/logger/logging.py",                # Logger for application logging
    "src/exception/__init__.py",
    "src/exception/exception.py",           # Custom exception handling

    # Configuration and setup files
    "setup.py",                             # Setup script for installation
]

# Create directories and files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create directory if it doesn't exist
    if filedir:  # Check if there is a directory path
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")
    
    # Create the file if it doesn't exist or is empty
    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            logging.info(f"Creating file: {filepath}")
            pass    #create an empty file


logging.info("Directories and files created successfully.")