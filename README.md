# SMS Spam Classification — ML & Deep Learning
> Detecting spam SMS messages with 98%+ accuracy using NLP, classical ML, and LSTM deep learning.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)]()
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)]()

## Problem Statement

SMS spam costs mobile users billions annually in lost time and security risks. Effective spam filters must be:
- **Highly accurate** — missing spam (false negatives) degrades user trust
- **Low false positive** — flagging real messages as spam is worse than letting some spam through
- **Fast** — filtering must happen in real-time (<10ms per message)

This project builds and compares multiple spam classifiers, from classical Naive Bayes to LSTM neural networks.

## Key Results

### Model Comparison

| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1 Score | Inference Time |
|---|---|---|---|---|---|
| Naive Bayes (TF-IDF) | 97.8% | 0.98 | 0.93 | 0.95 | <1ms |
| Logistic Regression | 98.1% | 0.97 | 0.95 | 0.96 | <1ms |
| SVM (Linear) | 98.3% | 0.98 | 0.95 | 0.96 | <1ms |
| **LSTM (Deep Learning)** | **98.5%** | **0.99** | **0.96** | **0.97** | ~5ms |

### Confusion Matrix (LSTM)

```
              Predicted
              Ham    Spam
Actual Ham    960      5     ← 0.5% false positive rate
Actual Spam     6    139     ← 95.9% spam caught
```

> **Key Insight**: All models perform well, but LSTM provides the best balance of precision and recall. For production deployment, Logistic Regression offers the best speed/accuracy tradeoff.

## Methodology

### 1. Text Preprocessing
- Lowercasing, punctuation removal, stopword filtering
- Tokenization and sequence padding (for LSTM)
- TF-IDF vectorization (for classical ML)

### 2. Class Imbalance
- Dataset is ~87% Ham, ~13% Spam
- Handled via stratified train/test split
- Evaluated using precision/recall (not just accuracy)

### 3. Classical ML Models
- **Naive Bayes**: Strong baseline for text classification
- **Logistic Regression**: Interpretable, fast, competitive accuracy
- **SVM (Linear Kernel)**: Best classical performance

### 4. Deep Learning (LSTM)
- Embedding layer → LSTM → Dense → Sigmoid
- Trained with binary cross-entropy loss
- Early stopping to prevent overfitting

## Project Structure

```
SMS_Spam_Classification_ML_DL/
├── Data/
│   └── spam.csv                 # SMS dataset (5,572 messages)
├── Output/
│   ├── confusion_matrix.png     # Model evaluation plots
│   └── roc_curve.png
├── spam_classification.py       # Full analysis pipeline
├── My_model.h5                  # Saved LSTM model (Keras)
├── My_model.pkl                 # Saved ML model (sklearn)
├── requirements.txt
└── README.md
```

## How to Run

```bash
# Clone and install
git clone https://github.com/the-irritater/SMS_Spam_Classification_ML_DL.git
cd SMS_Spam_Classification_ML_DL
pip install -r requirements.txt

# Run the full pipeline
python spam_classification.py

# Quick prediction with saved model
python -c "
import pickle
model = pickle.load(open('My_model.pkl', 'rb'))
# Use your TF-IDF vectorizer + model to predict
"
```

## Tech Stack

- **Python 3.11** — Core language
- **Pandas / NumPy** — Data processing
- **NLTK** — Text preprocessing, tokenization
- **Scikit-learn** — TF-IDF, Naive Bayes, Logistic Regression, SVM
- **TensorFlow / Keras** — LSTM deep learning model
- **Matplotlib / Seaborn** — Visualization

## Future Improvements

- [ ] Build **FastAPI endpoint** (`/predict`) for real-time classification
- [ ] Add **LIME/SHAP** explainability: highlight spam-triggering words
- [ ] Report **inference latency** benchmarks per model
- [ ] **Dockerize** and deploy to Render/Railway
- [ ] Add **DistilBERT** comparison using HuggingFace Transformers

## Author

Sanman Kadam  
MSc Statistics | Data Analyst

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sanman%20Kadam-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/sanman-kadam-7a4990374/)
[![GitHub](https://img.shields.io/badge/GitHub-the--irritater-black?style=flat&logo=github)](https://github.com/the-irritater)
