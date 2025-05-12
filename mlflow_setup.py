"""
MLflow Setup Script for Sentiment Analysis Project

This script sets up MLflow for the sentiment analysis project:
1. Creates the MLflow tracking directory
2. Initializes the model registry
3. Provides commands to start the MLflow UI
"""

import os
import mlflow
from mlflow.tracking import MlflowClient

# Create necessary directories
os.makedirs('mlruns', exist_ok=True)

# Set MLflow tracking URI to local directory
mlflow.set_tracking_uri("mlruns")

# Create MLflow client
client = MlflowClient()

# Create experiment if it doesn't exist
experiment_name = "Restaurant_Review_Sentiment_Analysis"
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
    else:
        print(f"Experiment '{experiment_name}' already exists with ID: {experiment.experiment_id}")
except Exception as e:
    print(f"Error setting up experiment: {e}")

# Check for registered models
try:
    models = client.search_registered_models()
    if models:
        print("\nRegistered Models:")
        for model in models:
            print(f"- {model.name}")
            
            # Get latest versions
            versions = client.get_latest_versions(model.name)
            for version in versions:
                print(f"  Version {version.version} (Stage: {version.current_stage})")
    else:
        print("\nNo registered models found. Run sentiment_analysis.py to create models.")
except Exception as e:
    print(f"Error checking registered models: {e}")

print("\nMLflow Setup Complete!")
print("\nTo start the MLflow UI, run:")
print("mlflow ui --backend-store-uri mlruns")
print("\nThen open http://localhost:5000 in your browser")
