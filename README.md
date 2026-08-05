# SMS Spam Classification: Machine Learning and Deep Learning

Detecting SMS spam messages with high precision and recall using Natural Language Processing, classical machine learning algorithms, and Long Short-Term Memory (LSTM) neural networks.

## Problem Statement

Unsolicited SMS spam causes financial loss and security risks. An effective filtering system must meet three criteria:
1. High overall classification accuracy.
2. Low false positive rate to prevent legitimate messages from being misclassified.
3. Fast execution latency for real-time message filtering.

This repository evaluates multiple machine learning baseline algorithms against an LSTM deep learning model.

## Model Evaluation & Comparison

| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1 Score | Inference Time |
|---|---|---|---|---|---|
| Naive Bayes (TF-IDF) | 97.8% | 0.98 | 0.93 | 0.95 | <1ms |
| Logistic Regression | 98.1% | 0.97 | 0.95 | 0.96 | <1ms |
| Support Vector Machine | 98.3% | 0.98 | 0.95 | 0.96 | <1ms |
| LSTM Neural Network | 98.5% | 0.99 | 0.96 | 0.97 | ~5ms |

## Confusion Matrix (LSTM Model)

```
              Predicted
              Ham    Spam
Actual Ham    960      5     (0.5% False Positive Rate)
Actual Spam     6    139     (95.9% Spam Recall)
```

Logistic Regression provides an optimal balance between execution speed and classification accuracy for production deployment.

## Methodology

1. **Text Preprocessing**: Lowercasing, punctuation removal, stopword filtering, tokenization, and sequence padding.
2. **Class Imbalance Handling**: Evaluated using precision, recall, and F1 metrics rather than raw accuracy alone.
3. **Classical ML Baselines**: Evaluated Naive Bayes, Logistic Regression, and Support Vector Machines.
4. **Deep Learning Architecture**: Constructed a Keras Sequential model featuring Embedding, Bidirectional LSTM, and Dense layers with Sigmoid activation.

## Project Structure

```
SMS_Spam_Classification_ML_DL/
├── Data/
│   └── spam.csv
├── Output/
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── app.py
├── Dockerfile
├── spam_classification.py
├── My_model.h5
├── My_model.pkl
├── requirements.txt
└── README.md
```

## How to Run

### Local Pipeline Execution
```bash
pip install -r requirements.txt
python spam_classification.py
```

### FastAPI Service Execution
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Docker Execution
```bash
docker build -t sms-spam-api .
docker run -p 8000:8000 sms-spam-api
```

## Author

Sanman Kadam  
MSc Statistics | Data Analyst
