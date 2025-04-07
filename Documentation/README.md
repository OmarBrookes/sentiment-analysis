# Sentiment Analysis Classifier – README

## Overview
This project presents a multi-label sentiment analysis tool developed using a RoBERTa-based transformer model. The application allows users to input product reviews and returns predictions across six sentiment categories: positive, negative, neutral, mixed, sarcastic, and ironic.

The system is designed to address challenges in detecting complex sentiments like sarcasm and irony, which traditional lexicon-based and ML-based approaches struggle with. The final product is a web-based application hosted on Hugging Face Spaces for public access. https://huggingface.co/spaces/OmarBrookes/sentiment-analyser 

## Folder Structure
- Dataset Preparation:
  - dataset_code.ipynb – Processes and analyses the dataset.
  - batch_400K(no_dupe).csv – Cleaned, labelled dataset used for training and testing.
  
- Model Training:
  - training_model_code.ipynb – Fine-tunes RoBERTa for multi-label classification.

- Model Testing:
  - testing_model_code.ipynb – Evaluates model performance using classification reports and visualizations.

- Final Model:
  - final_model_code.ipynb – Consolidates threshold tuning and final predictions.
  - roberta-multilabel-full/ – Saved model checkpoint.

- Deployment:
  - app.py – Streamlit frontend used for Hugging Face Spaces deployment.
  - requirements.txt – List of Python dependencies for running the web app.

## Instructions
To run locally:
1. Install required packages: `pip install -r requirements.txt`
2. Run `app.py` using: `streamlit run app.py`
3. The app will launch in your default browser.

To retrain the model:
1. Open `training_model_code.ipynb`
2. Ensure the dataset `batch_400K(no_dupe).csv` is in the same directory.
3. Train using the pre-defined pipeline with threshold tuning.

## Dataset
The dataset used in this project was sourced from an open-access repository available at:
https://deepdatalake.com/details.php?dataset_id=109

It contains approximately 400,000 entries based on real-world tweets, each labelled across multiple sentiment categories: positive, negative, neutral, mixed, sarcastic, and ironic. The dataset is licensed for academic and non-commercial use and was selected due to its large size and balanced distribution of complex sentiment classes.

Basic cleaning was performed, including duplicate removal, text normalisation, and the exclusion of unclear entries. Additional preprocessing included clarity scoring and token filtering. Label distributions and visualisations (such as word clouds and label frequency plots) are available in the notebook: `dataset_code.ipynb`.

## Results
The final RoBERTa model achieved the following approximate F1-scores:
- Positive: 90%
- Negative: 86%
- Neutral: 63%
- Mixed: 57%
- Sarcastic: 66%
- Ironic: 65%

## Deployment
The final version of the web application is deployed via Hugging Face Spaces:
https://huggingface.co/spaces/OmarBrookes/sentiment-analyser

## Author
Omar Alnaib
Oxford Brookes University