- create a project folder 
- git init
- dvc init
- requirements
- create venv and install requirements: pip install -r requirements.txt
- create github repo and add remote: git remote add origin <.git url>
- run template.py to create folder and files
- create setup.py and pip install -e .
- setup s3 buket
- dvc remote add -d myremote s3://my-mlops-project-demo/house_price_prediction
- setup dagshub and host mlflow
- perform experiments assesment, cleaning, EDA, transformation ---> EDA round2 ---> transformation, model building & hyperparameters (track with mlflow) in 4 phases: basic, features, algorithm, hyperparameter tuning
- convert best overall into dvc pipeline: create components like data_ingestion, data_transformation, model_trainer, model_evalutaion, logger etc
- upload best model on dagshub hosted mlflow model_registry in staging
- perform model test & if success then push model to production
- create fastapi (fetch latest model in production from mlflow model_registry and make predictions)


- test fast api in CI
- dockerize this app (dockerfile)
- create and run CI/CD pipeline on github actions and save dockerized app on ECR
- deploy dokerized app from ECR to ECS

- model retraining CT pipeline using airflow (manual, scheduled, event driven like new data/model_performance_drift etc) using evidently ai