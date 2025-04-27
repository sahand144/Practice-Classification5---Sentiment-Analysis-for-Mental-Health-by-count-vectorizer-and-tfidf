Mental Health Text Classification

This repository contains a Python-based machine learning project for classifying mental health conditions based on textual statements. The dataset includes statements (people's words) and their corresponding mental health status (e.g., normal, anxiety, happy, etc.). The project uses CountVectorizer and TfidfVectorizer to transform text data into numerical features and trains classification models to predict mental health conditions.

Table of Contents





Project Overview



Dataset



Features



Installation



Usage



Results



Contributing



License

Project Overview

The goal of this project is to classify mental health conditions from textual statements using natural language processing (NLP) and machine learning. The pipeline includes:





Preprocessing text data (lowercasing, removing special characters, tokenization, stopword removal).



Converting text to numerical features using CountVectorizer and TfidfVectorizer.



Training and evaluating classifiers like Naive Bayes, Logistic Regression, and Random Forest.



Comparing performance between vectorization methods.

Dataset

The dataset (mental_health_data.csv) consists of two columns:





Statement: Textual data representing people's statements (e.g., "I feel so stressed").



Status: Categorical labels indicating mental health conditions 

Features



Text preprocessing using NLTK ( stopword removal).


Feature extraction with CountVectorizer and TfidfVectorizer (unigrams and bigrams).



Support for multiple classifiers: Naive Bayes, Logistic Regression, Random Forest, Decision Tree.



Evaluation metrics: accuracy, precision, recall, F1-score.



Optional hyperparameter tuning and handling of imbalanced data.

Installation





Install the required dependencies:

pip install -r requirements.txt



Install NLTK resources:

import nltk
nltk.download('punkt')
nltk.download('stopwords')

Usage





Place your dataset (mental_health_data.csv) in the project root directory.



Run the main script to preprocess data, vectorize text, train models, and evaluate performance:

Sentiment Analysis for Mental Health.py



The script will:





Preprocess the text data.



Apply CountVectorizer and TfidfVectorizer.



Train and evaluate multiple classifiers.



Output accuracy and classification reports.



To make predictions on new statements:

import joblib
from preprocess import preprocess_text

# Load model and vectorizer
model = joblib.load('Sentiment Analysis for Mental Health.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
encoder = joblib.load('label_encoder.pkl')

# Predict
new_statement = "I feel so stressed and overwhelmed"
processed = preprocess_text(new_statement)
vectorized = vectorizer.transform([processed])
prediction = model.predict(vectorized)
status = encoder.inverse_transform(prediction)[0]
print(f"Predicted Status: {status}")

Results





Vectorizers:





CountVectorizer: Effective for capturing word frequency but sensitive to common words.



TfidfVectorizer: Generally outperforms CountVectorizer by emphasizing discriminative words.



Classifiers:





Logistic Regression with TfidfVectorizer typically achieves the highest accuracy (e.g., ~85% on balanced datasets).



Naive Bayes is fast and suitable for smaller datasets.



Random Forest captures complex patterns but is slower.



Detailed results (accuracy, precision, recall, F1-score) are printed in the console after running main.py.

Contributing

Contributions are welcome! To contribute:





Fork the repository.



Create a new branch (git checkout -b feature-branch).



Make your changes and commit (git commit -m "Add feature").



Push to the branch (git push origin feature-branch).



Open a pull request.

Please ensure your code follows PEP 8 guidelines and includes appropriate tests.
