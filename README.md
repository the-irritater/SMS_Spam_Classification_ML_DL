# SMS Spam Classification using Machine Learning and Deep Learning

## Problem Statement
With the increasing volume of SMS communication, spam messages have become a major concern affecting user experience and security. This project aims to build an intelligent system that can automatically classify SMS messages as spam or legitimate using both machine learning and deep learning techniques.
=======
# SMS Spam Classification Pipeline: NLP & Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue)]()
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2.0-orange)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-FF6F00)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

## Problem Statement

Spam messages pose significant cybersecurity risks and degrade user experience through phishing attempts, scams, and unwanted solicitations. The objective of this project is to construct a reliable, automated Natural Language Processing (NLP) system capable of accurately intercepting spam SMS messages while strictly avoiding the misclassification of legitimate user communications (False Positives).

## Project Overview

To solve this text classification challenge, this project takes a comprehensive engineering approach. It establishes strong baselines using **8 different Traditional Machine Learning models** (analyzing TF-IDF features) and subsequently implements an advanced **Bidirectional LSTM Neural Network** using TensorFlow/Keras to capture complex sequential text dependencies.

## Key Technical Objectives
>>>>>>> e60ef773 (README.md updated doing some changes.)
=======
# SMS Spam Classification using Machine Learning and Deep Learning

## Problem Statement
With the increasing volume of SMS communication, spam messages have become a major concern affecting user experience and security. This project aims to build an intelligent system that can automatically classify SMS messages as spam or legitimate using both machine learning and deep learning techniques.
>>>>>>> 0a4478bc842c7510289e633c290626756c3b3bc7

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

=======

## Performance Validation

### Model Comparison (Test Set)
>>>>>>> e60ef773 (README.md updated doing some changes.)

| Model                  | Accuracy   | Precision  | Recall     | F1-Score   |
| ---------------------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression    | 97.68%     | 98.90%     | 96.10%     | 97.48%     |
| Support Vector Machine | 97.70%     | 99.00%     | 96.50%     | 97.73%     |
| **Bidirectional LSTM** | **98.16%** | **99.20%** | **97.00%** | **98.09%** |

<<<<<<< HEAD
## Dataset
- SMS Spam Collection Dataset  
- Contains labeled SMS messages (Spam / Ham)  
- Text-based classification problem  
=======
- **Best Traditional ML**: **Logistic Regression & SVC** achieved an impressive **97.68% Accuracy** with near-perfect protection of legitimate messages, yielding an error rate of just 2.32%.
- **Deep Learning (Bi-LSTM)**: Achieved **98.16% Test Accuracy**, proving highly resilient and capable of interpreting nuanced semantic meaning that traditional Bag-of-Words vectors occasionally drop.

## Reproducibility & Technologies
>>>>>>> e60ef773 (README.md updated doing some changes.)

This codebase was developed using Python 3.8+ and relies on fixed random seeds (`seed = 42`) to guarantee experimental reproducibility.

<<<<<<< HEAD
## Tools and Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- NLTK  
=======
- **Python Data Stack**: `pandas`, `numpy`, `matplotlib`, `seaborn`
- **NLP Processing**: `nltk` (WordNet Lemmatizer, Tokenization), Regex, `wordcloud`
- **Machine Learning**: `scikit-learn` (CountVectorizer, Classifiers, Evaluation Metrics)
- **Deep Learning**: `TensorFlow` / `Keras` (Bi-LSTM, Dense Layers, Embeddings)

## Repository Structure
>>>>>>> e60ef773 (README.md updated doing some changes.)

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
=======
## Quick Start

To run this project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/the-irritater/SMS_Spam_Classification_ML_DL.git
   cd SMS_Spam_Classification_ML_DL
   ```
>>>>>>> e60ef773 (README.md updated doing some changes.)

2. **Install the dependencies:**
   It is recommended to use a Python virtual environment.

<<<<<<< HEAD
## Key Insights
- Machine learning models performed well with faster training time  
- Deep learning models showed improved performance on complex patterns  
- TF-IDF provided strong baseline results for text classification  
- Model performance depends on feature representation and complexity  
=======
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
>>>>>>> e60ef773 (README.md updated doing some changes.)

## Dataset Citation

## Model Comparison
- ML models are faster and easier to deploy  
- DL models capture deeper patterns but require more computation  
- Simpler models (Naive Bayes, Logistic Regression) are effective for baseline solutions  
=======
The data utilized in this repository is the renowned **SMS Spam Collection Data Set** originally sourced from the UCI Machine Learning Repository.

- **Source**: [UCI Machine Learning Repository: SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
>>>>>>> e60ef773 (README.md updated doing some changes.)

## Conclusion & Business Value

## Business Impact
- Enables automatic filtering of spam messages  
- Improves user experience by reducing unwanted messages  
- Can be integrated into messaging platforms and telecom systems  
- Enhances security by identifying potentially harmful messages  

---

## How to Run
1. Install required libraries  
2. Load dataset  
3. Run preprocessing and model training  
4. Evaluate model performance  

---

## Author
Sanman Kadam  
MSc Statistics Student | Aspiring Data Analyst  
GitHub: https://github.com/the-irritater
=======
This project demonstrates an end-to-end Machine Learning lifecycle. By successfully evaluating multiple modeling paradigms (Classical ML vs. Deep Neural Networks) and focusing on the precision-recall tradeoff to protect legitimate user traffic, this framework acts as a highly reliable, production-ready spam verification system.

## Future Enhancements
=======

---

## Dataset
- SMS Spam Collection Dataset  
- Contains labeled SMS messages (Spam / Ham)  
- Text-based classification problem  

---

## Tools and Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- NLTK  
>>>>>>> 0a4478bc842c7510289e633c290626756c3b3bc7

- **Deployment**: Build a Flask/FastAPI API endpoint for real-time classification.
- **Multilingual Support**: Extend dataset to include non-English SMS messages.
- **Explainability**: Integrate LIME or SHAP for model interpretability.
- **Active Learning**: Implement feedback loop to retrain model on user-reported spam.

<<<<<<< HEAD
## Author

**Sanman Kadam**  
_MSc Statistics | Data Analyst_

- **Location**: Mumbai, India
- **GitHub**: [the-irritater](https://github.com/the-irritater)
- **LinkedIn**: [Sanman Kadam](https://www.linkedin.com/in/sanman-kadam-7a4990374/)
- **Email**: your.email@example.com

## License

This project is open-source and available under the [MIT License](LICENSE).
>>>>>>> e60ef773 (README.md updated doing some changes.)
=======
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

## Key Insights
- Machine learning models performed well with faster training time  
- Deep learning models showed improved performance on complex patterns  
- TF-IDF provided strong baseline results for text classification  
- Model performance depends on feature representation and complexity  

---

## Model Comparison
- ML models are faster and easier to deploy  
- DL models capture deeper patterns but require more computation  
- Simpler models (Naive Bayes, Logistic Regression) are effective for baseline solutions  

---

## Business Impact
- Enables automatic filtering of spam messages  
- Improves user experience by reducing unwanted messages  
- Can be integrated into messaging platforms and telecom systems  
- Enhances security by identifying potentially harmful messages  

---

## How to Run
1. Install required libraries  
2. Load dataset  
3. Run preprocessing and model training  
4. Evaluate model performance  

---

## Author
Sanman Kadam  
MSc Statistics Student | Aspiring Data Analyst  
GitHub: https://github.com/the-irritater
>>>>>>> 0a4478bc842c7510289e633c290626756c3b3bc7
