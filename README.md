# Fake News Detection using Machine Learning

This project detects fake news using the LIAR dataset and Machine Learning.

## Features
- Text preprocessing using TF-IDF
- Logistic Regression classifier
- Real-time prediction system
- Dataset: LIAR dataset (train.tsv)

## How to Run

1. Create virtual environment:
   python -m venv venv
   venv\Scripts\activate

2. Install dependencies:
   pip install pandas numpy scikit-learn nltk joblib

3. Download NLTK resources:
   python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   exit()

4. Train model:
   python retrain_model.py

5. Run detector:
   python prediction.py
