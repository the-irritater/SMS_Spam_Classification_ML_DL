"""
FastAPI Production Endpoint for SMS Spam Classification
=========================================================
Accepts JSON requests with SMS text messages and returns
spam prediction, confidence probability, and inference latency.

Usage:
    uvicorn app:app --reload --port 8000
"""

import time
import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI(
    title="SMS Spam Classification API",
    description="Real-time SMS Spam detection using Deep Learning (LSTM) and Keras Tokenizer",
    version="1.0.0",
)

MODEL_PATH = "My_model.h5"
TOKENIZER_PATH = "My_model.pkl"

model = None
tokenizer = None
MAX_LEN = 100
PADDING_TYPE = "post"
TRUNC_TYPE = "post"
THRESHOLD = 0.5


@app.on_event("startup")
def load_artifacts():
    """Load model and tokenizer into memory during API startup."""
    global model, tokenizer
    try:
        if os.path.exists(TOKENIZER_PATH):
            with open(TOKENIZER_PATH, "rb") as f:
                tokenizer = pickle.load(f)
            print("Tokenizer loaded successfully.")
        else:
            raise FileNotFoundError(f"Tokenizer file not found at {TOKENIZER_PATH}")

        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Keras LSTM model loaded successfully.")
        else:
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    except Exception as e:
        print(f"Error loading model artifacts: {e}")


class PredictRequest(BaseModel):
    message: str = Field(
        ...,
        example="WINNER! As a valued network customer you have been selected to receive a $900 prize claim code.",
    )


class PredictResponse(BaseModel):
    message: str
    prediction: str
    is_spam: bool
    spam_probability: float
    confidence: float
    latency_ms: float


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "SMS Spam Classifier API",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict_spam(request: PredictRequest):
    """
    Predict whether an SMS message is Spam or Ham.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model artifacts are not loaded.")

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    start_time = time.time()

    sequences = tokenizer.texts_to_sequences([request.message])
    padded = pad_sequences(
        sequences, maxlen=MAX_LEN, padding=PADDING_TYPE, truncating=TRUNC_TYPE
    )

    prob = float(model.predict(padded, verbose=0)[0][0])
    is_spam = prob > THRESHOLD
    label = "Spam" if is_spam else "Ham"
    confidence = prob if is_spam else (1.0 - prob)

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return PredictResponse(
        message=request.message,
        prediction=label,
        is_spam=is_spam,
        spam_probability=round(prob, 4),
        confidence=round(confidence, 4),
        latency_ms=latency_ms,
    )
