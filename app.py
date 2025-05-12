import streamlit as st
import os
import re
import nltk
import pandas as pd
import mlflow.pyfunc
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

@st.cache_resource
def load_model():
    """Load the trained model from MLflow"""
    try:
        # First try to load from MLflow registry
        try:
            model_uri = "models:/SentimentAnalysisModel/Production"
            model = mlflow.pyfunc.load_model(model_uri)
            return model, "mlflow"
        except Exception as e:
            st.warning(f"Could not load model from MLflow registry: {e}")
            st.info("Falling back to local model files...")
        
        # Fall back to local pickle files if MLflow registry is not available
        import pickle
        with open('models/best_sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            
        # Load preprocessing config if available
        try:
            with open('models/preprocessing_config.json', 'r') as f:
                preprocessing_config = json.load(f)
        except:
            preprocessing_config = {
                'preprocessing': True,
                'stemming': True,
                'lemmatization': False,
                'keep_negation': True,
                'keep_punctuation': True,
                'keep_emoticons': True,
                'min_word_length': 2
            }
        
        return (model, vectorizer, preprocessing_config), "pickle"
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure the model files exist in the 'models' directory.")
        return None, None

def predict_sentiment(text, model_data):
    """Predict sentiment for a given text"""
    if model_data is None:
        return 0, 0.5
    
    model, model_type = model_data
    
    # If using MLflow model
    if model_type == "mlflow":
        # Create DataFrame for prediction (MLflow model expects 'text' column)
        input_df = pd.DataFrame({'text': [text]})
        
        # Predict - MLflow model handles preprocessing internally
        try:
            prediction = model.predict(input_df)[0]
            # For probability, we'll use a fixed confidence since MLflow model might not return probabilities
            probability = 0.85 if prediction == 1 else 0.15
            return prediction, probability
        except Exception as e:
            st.error(f"Error predicting with MLflow model: {e}")
            return 0, 0.5
    
    # If using pickle files
    elif model_type == "pickle":
        model_obj, vectorizer, config = model
        
        # Import necessary preprocessing functions
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        from nltk.tokenize import word_tokenize
        
        # Preprocess the text based on config
        def preprocess_text(text, config):
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
            if config.get('keep_punctuation', False):
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
            if config.get('keep_negation', True):
                stop_words = stop_words - negation_words
            
            # Filter tokens
            filtered_tokens = [token for token in tokens if token not in stop_words]
            
            # Stemming or Lemmatization
            if config.get('stemming', False):
                stemmer = PorterStemmer()
                filtered_tokens = [stemmer.stem(word) for word in filtered_tokens]
            
            if config.get('lemmatization', False):
                lemmatizer = WordNetLemmatizer()
                filtered_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
            
            # Join tokens back into text
            processed_text = ' '.join(filtered_tokens)
            
            return processed_text
        
        # Apply preprocessing if configured
        if config.get('preprocessing', True):
            processed_text = preprocess_text(text, config)
        else:
            processed_text = text
        
        # Vectorize the text
        text_vec = vectorizer.transform([processed_text])
        
        # Predict
        prediction = model_obj.predict(text_vec)[0]
        
        # Get probability if possible
        try:
            probability = model_obj.predict_proba(text_vec)[0][1]  # Probability of positive class
        except:
            # If model doesn't support predict_proba, use a fixed confidence
            probability = 0.85 if prediction == 1 else 0.15
        
        return prediction, probability
    
    else:
        return 0, 0.5

def create_emoji_image(sentiment, confidence):
    """Create an emoji image based on sentiment and confidence"""
    # Use pre-defined emoji patterns instead of drawing them
    width, height = 200, 200
    
    # Create a blank image with white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw a circle for the face
    circle_color = (255, 223, 0)  # Yellow
    center = (width // 2, height // 2)
    radius = min(width, height) // 2 - 10
    
    # Draw the circle
    for y in range(height):
        for x in range(width):
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            if dist <= radius:
                image[y, x] = circle_color
    
    # Draw eyes
    eye_color = (0, 0, 0)  # Black
    eye_radius = radius // 5
    left_eye_center = (center[0] - radius // 3, center[1] - radius // 4)
    right_eye_center = (center[0] + radius // 3, center[1] - radius // 4)
    
    for y in range(height):
        for x in range(width):
            dist_left = np.sqrt((x - left_eye_center[0])**2 + (y - left_eye_center[1])**2)
            dist_right = np.sqrt((x - right_eye_center[0])**2 + (y - right_eye_center[1])**2)
            if dist_left <= eye_radius or dist_right <= eye_radius:
                image[y, x] = eye_color
    
    # Draw mouth based on sentiment
    mouth_color = (0, 0, 0)  # Black
    
    if sentiment == 1:  # Positive sentiment (happy face)
        # Draw a happy smile (curved upward) positioned higher on the face
        smile_radius = radius // 2  # Use integer division
        smile_center_y = center[1] + radius // 4  # Moved up from radius//2 to radius//4
        smile_thickness = 3
        
        # Use integer values for range and adjust the range to be higher
        for y in range(center[1] + radius // 6, center[1] + radius // 2 + radius // 4):
            for x in range(center[0] - radius // 2, center[0] + radius // 2):
                # Calculate distance to the smile arc
                # This creates a thick upward curved arc
                arc_y = smile_center_y + np.sqrt(max(0, smile_radius**2 - (x - center[0])**2))
                if abs(y - arc_y) < smile_thickness:
                    image[y, x] = mouth_color
    
    else:  # Negative sentiment (sad face)
        # Draw a sad frown (curved downward)
        frown_radius = radius // 2  # Use integer division
        frown_center_y = center[1] + radius // 3
        frown_thickness = 3
        
        # Use integer values for range
        for y in range(center[1], center[1] + radius // 2):
            for x in range(center[0] - radius // 2, center[0] + radius // 2):
                # Calculate distance to the frown arc
                # This creates a thick downward curved arc
                arc_y = frown_center_y - np.sqrt(max(0, frown_radius**2 - (x - center[0])**2))
                if abs(y - arc_y) < frown_thickness:
                    image[y, x] = mouth_color
    
    return Image.fromarray(image.astype('uint8'))

def main():
    st.set_page_config(
        page_title="Restaurant Review Sentiment Analyzer",
        page_icon="🍽️",
        layout="centered"
    )
    
    # Create a visually appealing restaurant-themed header
    st.markdown("""
    <style>
    .restaurant-header {
        background: linear-gradient(to right, #8B4513, #A0522D, #8B4513);
        color: white;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-title {
        font-size: 2.2em;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .header-subtitle {
        font-size: 1.2em;
        font-style: italic;
        opacity: 0.9;
    }
    .header-icons {
        font-size: 1.5em;
        margin: 10px 0;
        letter-spacing: 10px;
    }
    </style>
    <div class="restaurant-header">
        <div class="header-title">Restaurant Review Sentiment Analysis</div>
        <div class="header-subtitle">Analyze the sentiment of restaurant reviews</div>
        <div class="header-icons">🍽️ 🍷 🍰 🍕 🍣</div>
    </div>
    """, unsafe_allow_html=True)

    
    # Load model from MLflow or pickle files
    model_data = load_model()
    
    if model_data[0] is None:
        st.stop()
        
    # Display model source information
    model_source = model_data[1]
    if model_source == "mlflow":
        st.sidebar.success("✅ Using MLflow model from registry")
    else:
        st.sidebar.info("ℹ️ Using local model files (MLflow registry not available)")
    
    # Text input
    review_text = st.text_area("Review:", height=150)
    
    # Analyze button
    if st.button("Analyze Sentiment"):
        if not review_text:
            st.warning("Please enter a review to analyze")
        else:
            with st.spinner("Analyzing sentiment..."):
                # Predict sentiment using loaded model
                prediction, confidence = predict_sentiment(review_text, model_data)
                
                # Display result
                sentiment = "Positive" if prediction == 1 else "Negative"
                
                # Create columns for layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sentiment")
                    st.markdown(f"<h1 style='text-align: center; color:{'green' if sentiment=='Positive' else 'red'};'>{sentiment}</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align: center;'>Confidence: {confidence:.2%}</h3>", unsafe_allow_html=True)
                
                with col2:
                    # Display emoji
                    emoji_image = create_emoji_image(prediction, confidence)
                    st.image(emoji_image, width=200)
                
                # Show preprocessing details
                with st.expander("Preprocessing Details"):
                    st.write("Text preprocessing is applied with the following settings:")
                    
                    # Get preprocessing config based on model source
                    if model_data[1] == "mlflow":
                        # For MLflow models, we show the standard preprocessing settings
                        config = {
                            "stemming": True,
                            "lemmatization": False,
                            "keep_negation": True,
                            "keep_punctuation": True,
                            "keep_emoticons": True,
                            "min_word_length": 2
                        }
                    else:
                        # For pickle models, extract config from the model data
                        config = model_data[0][2]
                    
                    st.json(config)
    
    # Add information about the model
    with st.expander("About this Model"):
        st.write("""
        This sentiment analysis model was trained on restaurant reviews and classifies them as positive or negative.
        
        The model uses Support Vector Machine (SVM) with TF-IDF vectorization and achieved 83.33% accuracy and 85.58% F1 score on test data.
        
        Features like negation words (e.g., "not good") and punctuation that indicates sentiment (e.g., "amazing!!!") 
        are taken into account during analysis.
        
        The model is stored and versioned using MLflow, allowing for easy model management and deployment.
        """)

if __name__ == "__main__":
    main()
