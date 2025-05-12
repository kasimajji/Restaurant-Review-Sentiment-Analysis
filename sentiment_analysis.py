import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline

# ML Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Text preprocessing
import re
import nltk
import os
import pickle
import json
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# MLflow imports
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('eda', exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs('roc_curves', exist_ok=True)
os.makedirs('mlruns', exist_ok=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri("mlruns")

# Create MLflow client for model registry operations
client = MlflowClient()

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
def load_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path, delimiter='\t')
    print(f"Dataset shape: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")
    print(f"Number of positive reviews: {df['Liked'].sum()}")
    print(f"Number of negative reviews: {len(df) - df['Liked'].sum()}")
    return df

# Exploratory Data Analysis
def perform_eda(df):
    print("\nPerforming Exploratory Data Analysis...")
    
    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Distribution of review lengths
    df['review_length'] = df['Review'].apply(len)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='review_length', hue='Liked', bins=50, kde=True)
    plt.title('Distribution of Review Lengths by Sentiment')
    plt.xlabel('Review Length (characters)')
    plt.ylabel('Count')
    plt.savefig('eda/review_length_distribution.png')
    plt.close()
    
    # Most common words in positive and negative reviews
    def get_top_words(reviews, n=20):
        all_words = ' '.join(reviews).lower()
        all_words = re.sub(r'[^a-zA-Z\s]', '', all_words)
        words = all_words.split()
        words = [word for word in words if word not in stopwords.words('english')]
        word_freq = pd.Series(words).value_counts().head(n)
        return word_freq
    
    pos_reviews = df[df['Liked'] == 1]['Review']
    neg_reviews = df[df['Liked'] == 0]['Review']
    
    pos_words = get_top_words(pos_reviews)
    neg_words = get_top_words(neg_reviews)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    pos_words.plot.barh(ax=ax1, color='green')
    ax1.set_title('Top Words in Positive Reviews')
    neg_words.plot.barh(ax=ax2, color='red')
    ax2.set_title('Top Words in Negative Reviews')
    plt.tight_layout()
    plt.savefig('eda/top_words.png')
    plt.close()
    
    return df

# Text Preprocessing
def preprocess_text(text, stemming=False, lemmatization=True, keep_emoticons=True, 
                  keep_negation=True, keep_punctuation=False, min_word_length=2):
    """
    Preprocess text for sentiment analysis with configurable options.
    
    Args:
        text (str): Input text to preprocess
        stemming (bool): Whether to apply stemming
        lemmatization (bool): Whether to apply lemmatization
        keep_emoticons (bool): Whether to preserve emoticons/emojis
        keep_negation (bool): Whether to keep negation words like 'not', 'no', etc.
        keep_punctuation (bool): Whether to keep punctuation that might indicate sentiment
        min_word_length (int): Minimum word length to keep
        
    Returns:
        str: Preprocessed text
    """
    # Save emoticons/emojis if requested
    emoticons = []
    if keep_emoticons:
        # Simple emoticon patterns
        emoticon_pattern = r'[:;=][-o^]?[)(/\\|dpDP]'
        emoticons = re.findall(emoticon_pattern, text)
        
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
    filtered_tokens = []
    for token in tokens:
        # Keep if not a stopword and meets minimum length
        if token not in stop_words and len(token) >= min_word_length:
            filtered_tokens.append(token)
        # Keep if it's a negation word and we want to keep those
        elif keep_negation and token in negation_words:
            filtered_tokens.append(token)
    
    # Stemming or Lemmatization
    if stemming:
        stemmer = PorterStemmer()
        filtered_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        filtered_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    # Add back emoticons if requested
    if keep_emoticons and emoticons:
        filtered_tokens.extend(['EMOTICON'] * len(emoticons))
    
    # Join tokens back into text
    processed_text = ' '.join(filtered_tokens)
    
    return processed_text

# Feature Engineering
def create_features(df, preprocessing=True, vectorizer_type='tfidf', 
                   stemming=False, lemmatization=True, keep_emoticons=True,
                   keep_negation=True, keep_punctuation=False, min_word_length=2):
    print("\nCreating features...")
    
    # Preprocess text if required
    if preprocessing:
        print("Applying text preprocessing...")
        print(f"Options: stemming={stemming}, lemmatization={lemmatization}, keep_negation={keep_negation}, keep_punctuation={keep_punctuation}")
        
        df['processed_review'] = df['Review'].apply(
            lambda x: preprocess_text(
                x, 
                stemming=stemming,
                lemmatization=lemmatization,
                keep_emoticons=keep_emoticons,
                keep_negation=keep_negation,
                keep_punctuation=keep_punctuation,
                min_word_length=min_word_length
            )
        )
        text_column = 'processed_review'
    else:
        text_column = 'Review'
    
    # Split data
    X = df[text_column]
    y = df['Liked']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Vectorize text
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer(max_features=5000)
    else:  # tfidf
        vectorizer = TfidfVectorizer(max_features=5000)
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Training set shape: {X_train_vec.shape}")
    print(f"Testing set shape: {X_test_vec.shape}")
    
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer

# Model Training and Evaluation
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    print("\nTraining and evaluating models...")
    
    models = {
        'Multinomial Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification Report
        report = classification_report(y_test, y_pred)
        print(f"Classification Report for {name}:\n{report}")
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        # ROC Curve (for models that support predict_proba)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            results[name]['roc_auc'] = roc_auc
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend(loc='lower right')
            plt.savefig(f'roc_curves/roc_curve_{name.replace(" ", "_").lower()}.png')
            plt.close()
    
    # Compare models
    model_comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results],
        'Precision': [results[model]['precision'] for model in results],
        'Recall': [results[model]['recall'] for model in results],
        'F1 Score': [results[model]['f1'] for model in results],
        'ROC AUC': [results[model].get('roc_auc', np.nan) for model in results]
    })
    
    model_comparison = model_comparison.sort_values('F1 Score', ascending=False)
    print("\nModel Comparison:")
    print(model_comparison)
    
    # Visualize model comparison
    plt.figure(figsize=(12, 8))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    model_comparison_plot = model_comparison[['Model'] + metrics].melt(id_vars='Model', value_vars=metrics)
    sns.barplot(x='Model', y='value', hue='variable', data=model_comparison_plot)
    plt.xticks(rotation=45, ha='right')
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    plt.savefig('eda/model_comparison.png')
    plt.close()
    
    # Find the best model based on F1 score
    best_model_name = model_comparison.iloc[0]['Model']
    best_model = results[best_model_name]['model']
    
    return results, best_model_name, best_model

# Hyperparameter Tuning for the best model
def tune_best_model(X_train, y_train, best_model_name, best_model):
    print(f"\nPerforming hyperparameter tuning for {best_model_name}...")
    
    param_grid = {}
    
    if best_model_name == 'Multinomial Naive Bayes':
        param_grid = {
            'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]
        }
    elif best_model_name == 'Logistic Regression':
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga']
        }
    elif best_model_name == 'Support Vector Machine':
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.1, 1]
        }
    elif best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    elif best_model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    elif best_model_name == 'K-Nearest Neighbors':
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    elif best_model_name == 'Decision Tree':
        param_grid = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
    
    if param_grid:
        grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    with open('models/best_sentiment_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    with open('models/preprocessing_config.json', 'w') as f:
        json.dump(preprocessing_config, f)
    
    # Start an MLflow run for logging
    with mlflow.start_run(run_name=f"{model_name}_production"):
        # Log parameters
        for param, value in preprocessing_config.items():
            mlflow.log_param(f"preprocess_{param}", value)
        
        # Log model type
        mlflow.log_param("model_type", model_name)
        
        # Log model hyperparameters if available
        if hasattr(model, 'get_params'):
            for param, value in model.get_params().items():
                mlflow.log_param(param, value)
        
        # Create a custom model that includes preprocessing
        sentiment_model = SentimentAnalysisModel(
            model=model,
            vectorizer=vectorizer,
            preprocessing_config=preprocessing_config
        )
        
        # Create example input for model signature
        example_input = pd.DataFrame({"text": ["This is an example review"]})
        
        # Log the custom model
        mlflow.pyfunc.log_model(
            artifact_path="sentiment_model",
            python_model=sentiment_model,
            input_example=example_input,
            registered_model_name="SentimentAnalysisModel"
        )
        
        # Get the run ID for reference
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run ID: {run_id}")
        
    # Transition the latest version to staging
    latest_version = get_latest_model_version("SentimentAnalysisModel")
    if latest_version:
        client.transition_model_version_stage(
            name="SentimentAnalysisModel",
            version=latest_version,
            stage="Production"
        )
        print(f"Model version {latest_version} transitioned to Production stage")
    
    print("Model and vectorizer saved successfully with MLflow.")

# Helper function to get the latest model version
def get_latest_model_version(model_name):
    """
    Get the latest version of a registered model.
    
    Args:
        model_name (str): Name of the registered model
        
    Returns:
        int: Latest version number or None if model doesn't exist
    """
    try:
        model_versions = client.get_latest_versions(model_name)
        if model_versions:
            return model_versions[0].version
        return None
    except Exception as e:
        print(f"Error getting latest model version: {e}")
        return None

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
                    lemmatization=self.preprocessing_config.get('lemmatization', True),
                    keep_emoticons=self.preprocessing_config.get('keep_emoticons', True),
                    keep_negation=self.preprocessing_config.get('keep_negation', True),
                    keep_punctuation=self.preprocessing_config.get('keep_punctuation', False),
                    min_word_length=self.preprocessing_config.get('min_word_length', 2)
                )
            )
        else:
            processed_texts = text_series
        

# Function to predict sentiment on new data
def predict_sentiment(text, model, vectorizer, preprocessing=True, stemming=False, 
                     lemmatization=True, keep_emoticons=True, keep_negation=True, 
                     keep_punctuation=False, min_word_length=2):
    """
    Predict sentiment for a given text using the trained model.
    
    Args:
        text (str): The text to predict sentiment for
        model: The trained sentiment model
        vectorizer: The vectorizer used to transform text
        preprocessing (bool): Whether to apply preprocessing
        stemming, lemmatization, etc.: Preprocessing options
        
    Returns:
        tuple: (prediction, probability)
    """
    if preprocessing:
        processed_text = preprocess_text(
            text,
            stemming=stemming,
            lemmatization=lemmatization,
            keep_emoticons=keep_emoticons,
            keep_negation=keep_negation,
            keep_punctuation=keep_punctuation,
            min_word_length=min_word_length
        )
    else:
        processed_text = text
    
    # Vectorize the text
    text_vec = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(text_vec)[0]
    
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(text_vec)[0][prediction]
        return prediction, probability
    else:
        return prediction, 0.5

# ... (rest of the code remains the same)

# Main function
def main():
    # ... (rest of the code remains the same)

    # Test on sample reviews
    print("\nPredicting sentiment on sample reviews:")
    sample_reviews = [
        "The food was amazing and the service was excellent!",
        "Terrible experience, would not recommend to anyone.",
        "Average food, nothing special but not bad either."
    ]
    
    for review in sample_reviews:
        prediction, probability = predict_sentiment(
            review, 
            tuned_model, 
            vectorizer, 
            preprocessing=False
        )
        sentiment = "Positive" if prediction == 1 else "Negative"
        prob_text = f" (Probability: {probability:.4f})" if probability is not None else ""
        
        print(f"Review: '{review}'")
        print(f"Predicted sentiment: {sentiment}{prob_text}\n")

if __name__ == "__main__":
    main()
