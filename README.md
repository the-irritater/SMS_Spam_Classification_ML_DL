SMS Spam Classification using ML & Deep Learning
📌 Project Overview

This project focuses on classifying SMS messages as Spam or Ham (Not Spam) using both traditional Machine Learning algorithms and a Deep Learning Bidirectional LSTM model.

The objective is to build a high-performance spam detection system with strong generalization capability.

🚀 Models Implemented
🔹 Machine Learning Models:

Gaussian Naive Bayes

Multinomial Naive Bayes

Decision Tree

Logistic Regression

K-Nearest Neighbors

Support Vector Classifier

Gradient Boosting

Bagging Classifier

🔹 Deep Learning Model:

Bidirectional LSTM (bLSTM)

📊 Final Model Performance (bLSTM)

Test Accuracy: 98.16%

Test Loss: 0.0744

Spam Precision: 91%

Spam Recall: 93%

Ham Precision: 99%

Overall Error Rate: 1.84%

The model shows minimal overfitting and excellent generalization.

🧠 Deep Learning Architecture

Embedding Layer (64 dimensions)

Bidirectional LSTM (100 units)

Dropout Regularization (0.3)

Dense Layer (ReLU Activation)

Sigmoid Output Layer

Total Trainable Parameters: 742,121

SMS-Spam-Classification-ML-DL/
│
├── spam_classification.py
├── spam.csv
├── My_model.h5
├── My_model.pkl
├── requirements.txt
└── README.md


Conclusion
The Bidirectional LSTM model achieved high classification performance and demonstrates strong capability in real-world spam detection tasks.

Contributors

Sanman Kadam
Rutuja Shinde
