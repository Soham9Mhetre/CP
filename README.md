# Transactional Fraud Detection using Deep Learning

This project implements a deep learning model for detecting fraudulent transactions using an Autoencoder for anomaly detection.

## Project Structure
- `data/`: Dataset files (place creditcard.csv here)
- `models/`: Saved trained models and plots
- `src/`: Source code
- `notebooks/`: Jupyter notebooks for exploration and training
- `requirements.txt`: Python dependencies
- `venv/`: Virtual environment (created automatically)

## Setup
1. Ensure Python 3.10+ is installed.
2. Create virtual environment: `python -m venv venv`
3. Activate: 
   - Command Prompt: `venv\Scripts\activate`
   - PowerShell: `venv\Scripts\Activate.ps1`
4. Install dependencies: `pip install -r requirements.txt`

## Dataset
Download the Credit Card Fraud Detection dataset from Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud
Place `creditcard.csv` in the `data/` folder.

## Training
Run the training script: `python src/train.py`

This will train an autoencoder on normal transactions and evaluate on the test set.

## Model
- Uses TensorFlow/Keras
- Autoencoder architecture: Encoder (14 units) -> Decoder
- Trained on normal transactions only
- Anomalies detected based on reconstruction error threshold

## Evaluation
- Confusion Matrix
- Classification Report
- Reconstruction Error Histogram