from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import logging
from src.preprocessing import process_tweet
# Removed unused imports
from typing import Dict
import os

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load saved models and vectorizer
try:
    BASE_PATH = os.getenv("MODEL_BASE_PATH", "results")
    model = joblib.load(os.path.join(BASE_PATH, "kmeans_model.joblib"))
    vectorizer = joblib.load(os.path.join(BASE_PATH, "vectorizer.joblib"))
    logger.info("Models and vectorizer loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise RuntimeError("Failed to load models or vectorizer. The Base path is : ", BASE_PATH)

# Define input data format
class TextData(BaseModel):
    text: str

@app.get("/")
def home() -> Dict[str, str]:
    return {"message": "Welcome to the Text Clustering API"}

@app.get("/results")
def get_results() -> Dict[str, float]:
    """Fetch stored clustering results."""
    try:
        silhouette = joblib.load(os.path.join(BASE_PATH, "silhouette.joblib"))
        purity = joblib.load(os.path.join(BASE_PATH, "purity_score.joblib"))
        return {"silhouette_score": silhouette, "purity_score": purity}
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        raise HTTPException(status_code=500, detail="Failed to load clustering results.")

@app.post("/predict")
def predict_cluster(data: TextData) -> Dict[str, str | int]:
    """Process new text and predict its cluster."""
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    
    try:
        processed_text = process_tweet(data.text)  # Preprocess input text
        cleaned_text = " ".join(processed_text)  # Convert list to string
        X_new = vectorizer.transform([cleaned_text])  # Vectorize text
        cluster = model.predict(X_new)[0]  # Predict cluster
        return {"text": data.text, "predicted_cluster": cluster}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to predict cluster.")
