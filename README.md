# SMS Spam Classification using Machine Learning and Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue)]()
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2.0-orange)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-FF6F00)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()


## Problem Statement
With the increasing volume of SMS communication, spam messages have become a major concern affecting user experience and security. This project aims to build an intelligent system that can automatically classify SMS messages as spam or legitimate using both machine learning and deep learning techniques.

- **Data Preprocessing & EDA**: Clean textual data (Regex, Lemmatization, Stopword removal) and visually explore textual patterns utilizing custom Word Clouds and frequency distributions.
- **Addressing Class Imbalance**: Recognize that 'Ham' dominates the dataset (86.59%) and prioritize metrics like precision and recall over bare accuracy to strictly minimize False Positives.
- **Traditional ML Baselines**: Train and exhaustively compare algorithms including Logistic Regression, Support Vector Machine (SVM), Naive Bayes variants, Decision Trees, Gradient Boosting, and Bagging classifiers.
- **Deep Learning Architecture**: Design a Bi-LSTM model with Word Embeddings and Dropout regularization layers to achieve superior contextual understanding of messaging patterns.

## Objectives
- Perform text preprocessing and feature extraction  
- Build machine learning models for spam classification  
- Develop deep learning models for improved accuracy  
- Compare performance between ML and DL approaches  
- Identify the most effective method for real-world deployment  

---

## Performance Validation

### Model Comparison (Test Set)

| Model                  | Accuracy   | Precision  | Recall     | F1-Score   |
| ---------------------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression    | 97.68%     | 98.90%     | 96.10%     | 97.48%     |
| Support Vector Machine | 97.70%     | 99.00%     | 96.50%     | 97.73%     |
| **Bidirectional LSTM** | **98.16%** | **99.20%** | **97.00%** | **98.09%** |

## Dataset
- SMS Spam Collection Dataset  
- Contains labeled SMS messages (Spam / Ham)  
- Text-based classification problem  
---
- **Best Traditional ML**: **Logistic Regression & SVC** achieved an impressive **97.68% Accuracy** with near-perfect protection of legitimate messages, yielding an error rate of just 2.32%.
- **Deep Learning (Bi-LSTM)**: Achieved **98.16% Test Accuracy**, proving highly resilient and capable of interpreting nuanced semantic meaning that traditional Bag-of-Words vectors occasionally drop.

## Reproducibility & Technologies

This codebase was developed using Python 3.8+ and relies on fixed random seeds (`seed = 42`) to guarantee experimental reproducibility.

## Tools and Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- NLTK  
---
- **Python Data Stack**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **NLP Processing**: `nltk` (WordNet Lemmatizer, Tokenization), Regex, `wordcloud`
- **Machine Learning**: `scikit-learn` (CountVectorizer, Classifiers, Evaluation Metrics)
- **Deep Learning**: `TensorFlow` / `Keras` (Bi-LSTM, Dense Layers, Embeddings)

## Repository Structure

```text
├── Data/
│   └── spam.csv                             # Raw dataset
├── Images/                                  # Stored visualization artifacts
├── SMS_Spam_Classification_Analysis.ipynb   # Comprehensive EDA, Models & Visualization
├── spam_classification.py                   # Original standalone execution script
├── Spam_Detector_Model.h5                    # Serialized Bi-LSTM Keras model
├── Spam_Detector_Model.pkl                   # Serialized Tokenizer
├── requirements.txt                         # Package dependencies
└── README.md                                # Project documentation
```

## Approach

### 1. Data Preprocessing
- Text cleaning (removal of punctuation, stopwords)  
- Tokenization and normalization  

### 2. Feature Engineering
- TF-IDF / Count Vectorization  

### 3. Machine Learning Models
- Naive Bayes  
- Logistic Regression  
- Support Vector Machine  

### 4. Deep Learning Models
- Artificial Neural Network (ANN)  
- (Optional: LSTM if used)

### 5. Model Evaluation
- Accuracy  
- Precision, Recall, F1-score  
- Confusion Matrix  
---
## Quick Start

To run this project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/the-irritater/SMS_Spam_Classification_ML_DL.git
   cd SMS_Spam_Classification_ML_DL
   ```
2. **Install the dependencies:**
   It is recommended to use a Python virtual environment.

## Key Insights
- Machine learning models performed well with faster training time  
- Deep learning models showed improved performance on complex patterns  
- TF-IDF provided strong baseline results for text classification  
- Model performance depends on feature representation and complexity  
---
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis:**
   You can either explore the highly documented Jupyter Notebook:
   ```bash
   jupyter notebook SMS_Spam_Classification_Analysis.ipynb
   ```
   Or run the standalone Python script directly:
   ```bash
   python spam_classification.py
   ```
---

## Model Comparison
- ML models are faster and easier to deploy  
- DL models capture deeper patterns but require more computation  
- Simpler models (Naive Bayes, Logistic Regression) are effective for baseline solutions  
=======
The data utilized in this repository is the renowned **SMS Spam Collection Data Set** originally sourced from the UCI Machine Learning Repository.

---

## Business Impact
- Enables automatic filtering of spam messages with 98.16% accuracy
- Improves user experience by reducing unwanted messages and phishing attempts
- Can be integrated into messaging platforms and telecom systems via API
- Enhances security by identifying potentially harmful messages before user interaction
- Provides interpretable metrics (precision/recall) to balance safety and usability

---

Future Enhancements
- Deployment: Build a Flask/FastAPI endpoint for real-time classification
- Multilingual Support: Extend to non-English SMS datasets
- Explainability: Integrate LIME or SHAP for model interpretability
- Active Learning: Implement user feedback loop for continuous model improvement
- Model Compression: Apply quantization or pruning for mobile deployment

---

## How to Run
1. Install required libraries  
2. Load dataset  
3. Run preprocessing and model training  
4. Evaluate model performance  

---

Author
Sanman Kadam
MSc Statistics | Data Analyst
Location: Mumbai, India
GitHub: https://github.com/the-irritater
LinkedIn: https://www.linkedin.com/in/sanman-kadam-7a4990374/
Email: sanman.kadam@statistics.mu.ac.in

---
This project demonstrates an end-to-end Machine Learning lifecycle. By successfully evaluating multiple modeling paradigms (Classical ML vs. Deep Neural Networks) and focusing on the precision-recall tradeoff to protect legitimate user traffic, this framework acts as a highly reliable, production-ready spam verification system.
