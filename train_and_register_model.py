"""
Train and Register Sentiment Analysis Model with MLflow

This script trains a sentiment analysis model and registers it with MLflow.
"""

import pandas as pd
import numpy as np
import re
import os
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# MLflow imports
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('mlruns', exist_ok=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri("mlruns")

# Create MLflow client
client = MlflowClient()

# Set experiment
experiment_name = "Restaurant_Review_Sentiment_Analysis"
mlflow.set_experiment(experiment_name)

# Define preprocessing function
def preprocess_text(text, stemming=True, lemmatization=False, keep_negation=True, keep_punctuation=True):
    """Preprocess text for sentiment analysis"""
    # Convert to lowercase
    text = text.lower()
    
    # Handle contractions
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    
    # Keep certain punctuation that might indicate sentiment
    if keep_punctuation:
        # Replace multiple occurrences with single + count
        text = re.sub(r'([!?])\1+', r'\1 REPEAT', text)
    else:
        # Remove special characters, numbers, and punctuation
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Get negation words
    negation_words = {'no', 'not', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'hardly', 'rarely', 'scarcely', 'seldom'}
    
    # Remove stopwords but keep negation words if requested
    stop_words = set(stopwords.words('english'))
    if keep_negation:
        stop_words = stop_words - negation_words
    
    # Filter tokens
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    if stemming:
        stemmer = PorterStemmer()
        filtered_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    # Join tokens back into text
    processed_text = ' '.join(filtered_tokens)
    
    return processed_text

# Create a custom MLflow model class that includes preprocessing
class SentimentAnalysisModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model, vectorizer, preprocessing_config):
        self.model = model
        self.vectorizer = vectorizer
        self.preprocessing_config = preprocessing_config
        
    def predict(self, context, model_input):
        """
        Predict sentiment from raw text input
        
        Args:
            context: MLflow context
            model_input: Pandas DataFrame with a 'text' column
            
        Returns:
            Predictions (0 for negative, 1 for positive)
        """
        # Extract text from input
        if isinstance(model_input, pd.DataFrame):
            text_series = model_input['text']
        else:
            text_series = pd.Series(model_input)
        
        # Apply preprocessing if configured
        if self.preprocessing_config.get('preprocessing', False):
            processed_texts = text_series.apply(
                lambda x: preprocess_text(
                    x,
                    stemming=self.preprocessing_config.get('stemming', False),
                    lemmatization=self.preprocessing_config.get('lemmatization', False),
                    keep_negation=self.preprocessing_config.get('keep_negation', True),
                    keep_punctuation=self.preprocessing_config.get('keep_punctuation', False)
                )
            )
        else:
            processed_texts = text_series
        
        # Vectorize
        X_vec = self.vectorizer.transform(processed_texts)
        
        # Predict
        predictions = self.model.predict(X_vec)
        
        return predictions

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('Restaurant_reviews.tsv', delimiter='\t')
    print(f"Dataset shape: {df.shape}")
    print(f"Number of positive reviews: {df['Liked'].sum()}")
    print(f"Number of negative reviews: {len(df) - df['Liked'].sum()}")
    
    # Define preprocessing config
    preprocessing_config = {
        'preprocessing': True,
        'stemming': True,
        'lemmatization': False,
        'keep_negation': True,
        'keep_punctuation': True
    }
    
    # Start MLflow run
    with mlflow.start_run(run_name="svm_tfidf_model"):
        # Log preprocessing config
        for param, value in preprocessing_config.items():
            mlflow.log_param(f"preprocess_{param}", value)
        
        # Split data
        X = df['Review']
        y = df['Liked']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Preprocess text
        print("Preprocessing text...")
        X_train_processed = X_train.apply(lambda x: preprocess_text(
            x, 
            stemming=preprocessing_config['stemming'],
            lemmatization=preprocessing_config['lemmatization'],
            keep_negation=preprocessing_config['keep_negation'],
            keep_punctuation=preprocessing_config['keep_punctuation']
        ))
        
        X_test_processed = X_test.apply(lambda x: preprocess_text(
            x, 
            stemming=preprocessing_config['stemming'],
            lemmatization=preprocessing_config['lemmatization'],
            keep_negation=preprocessing_config['keep_negation'],
            keep_punctuation=preprocessing_config['keep_punctuation']
        ))
        
        # Vectorize
        print("Vectorizing text...")
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train_processed)
        X_test_vec = vectorizer.transform(X_test_processed)
        
        # Train model
        print("Training SVM model...")
        model = SVC(C=10, gamma=1, kernel='rbf', probability=True, random_state=42)
        model.fit(X_train_vec, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        
        # Save model and vectorizer as pickle files (for backward compatibility)
        with open('models/best_sentiment_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open('models/vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        
        with open('models/preprocessing_config.json', 'w') as f:
            json.dump(preprocessing_config, f)
        
        print("Model and vectorizer saved as pickle files.")
        
        # Create a custom model that includes preprocessing
        sentiment_model = SentimentAnalysisModel(
            model=model,
            vectorizer=vectorizer,
            preprocessing_config=preprocessing_config
        )
        
        # Create example input for model signature
        example_input = pd.DataFrame({"text": ["This is an example review"]})
        
        # Log the custom model
        print("Logging model to MLflow...")
        mlflow.pyfunc.log_model(
            artifact_path="sentiment_model",
            python_model=sentiment_model,
            input_example=example_input,
            registered_model_name="SentimentAnalysisModel"
        )
        
        # Get the run ID for reference
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run ID: {run_id}")
    
    # Transition the latest version to production
    latest_versions = client.get_latest_versions("SentimentAnalysisModel")
    if latest_versions:
        latest_version = latest_versions[0].version
        client.transition_model_version_stage(
            name="SentimentAnalysisModel",
            version=latest_version,
            stage="Production"
        )
        print(f"Model version {latest_version} transitioned to Production stage")
    
    print("Model training and registration complete!")

if __name__ == "__main__":
    main()
