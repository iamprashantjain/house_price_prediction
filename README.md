- create a project folder 
- git init
- dvc init
- requirements
- create venv and install requirements: pip install -r requirements.txt
- create github repo and add remote: git remote add origin <.git url>
- run template.py to create folder and files
- create setup.py and pip install -e .
- setup s3 buket (sigin --> s3 --> )
- dvc remote add -d myremote s3://my-bucket/dvc-store

- perform experiment to find out best strategy to cleaning, transformation, model building & hyperparameters (track with mlflow) to convert into dvc pipeline

- convert best strategies into dvc pipeline
- create components like data_ingestion, data_transformation, model_trainer, model_evalutaion, logger etc
- create CICD pipeline on github