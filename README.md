# Spectral-Temporal Graph Network for Fraud Detection

This project implements a Graph Neural Network based fraud detection system for financial transaction graphs. The system is designed to detect camouflage fraud attacks where malicious users hide among legitimate users.

The model combines three main techniques:

• Spectral Graph Filtering  
• Graph Attention Networks (GAT)  
• Temporal Behavior Modeling (LSTM)

The project is implemented using PyTorch and PyTorch Geometric and is tested on the Elliptic Bitcoin Transaction Dataset.

---


# Project Architecture

The system follows a **Filter → Encode → Temporal Reasoning → Prediction** pipeline.

Transaction Graph  
↓  
Spectral Graph Filter  
↓  
Graph Attention Network (GAT)  
↓  
Temporal LSTM Module  
↓  
Classifier  
↓  
Fraud Prediction

---

# Key Components

## Spectral Graph Filter

The spectral filter removes high-frequency adversarial noise in the graph.

Adversarial attacks often inject fake edges to hide fraud nodes. Spectral filtering suppresses these noisy graph signals before the model learns from the data.

File location:  
models/spectral_filter.py

---

## Graph Attention Network (GAT)

The GAT encoder learns adaptive importance weights for neighbors.

Instead of averaging all neighbors equally like a GCN, the attention mechanism learns which connections are important. This helps detect camouflage connections between fraud and legitimate transactions.

File location:  
models/gat_encoder.py

---

## Temporal Modeling (LSTM)

Financial fraud often occurs in transaction sequences such as laundering chains.

Example pattern:

A → B → C → D → E

The temporal module captures behavioral evolution over time using an LSTM network.

The Elliptic dataset contains **49 time steps** which are used to compute temporal embeddings and detect abnormal transaction flows.

File location:  
models/temporal_lstm.py

---

# Dataset

This project uses the **Elliptic Bitcoin Dataset**.

Dataset characteristics:

Nodes: ~203,769 transactions  
Edges: ~234,355 transaction flows  
Features: 166 features per node  
Time steps: 49 temporal snapshots  
Classes: Licit or Illicit

Nodes represent **Bitcoin transactions** and edges represent **movement of funds between transactions**.

Dataset loader:  
data/load_dataset.py

---

# Installation

Clone the repository:

git clone <repository-url>  
cd CP

Create virtual environment:

python -m venv venv

Activate environment:

Windows:

venv\Scripts\activate

Install dependencies:

pip install torch  
pip install torch-geometric  
pip install scikit-learn

---

# Project Structure

CP  
│  
├── data  
│   └── load_dataset.py  
│  
├── models  
│   ├── spectral_filter.py  
│   ├── gat_encoder.py  
│   └── temporal_lstm.py  
│  
├── training  
│   └── train.py  
│  
└── README.md

---

# Training the Model

Run training using:

python -m training.train

Training uses:

• Adam optimizer  
• Weighted Cross Entropy Loss  
• Class imbalance correction

Fraud class weight:

[1 , 9]

This ensures the model prioritizes detecting fraud transactions.

---

# Evaluation Metrics

The system is evaluated using:

Accuracy  
Precision  
Recall  
F1 Score

Example result:

Accuracy: 0.89  

Fraud Precision: 0.33  
Fraud Recall: 0.68  
Fraud F1 Score: 0.45

Higher recall indicates the system successfully detects a large portion of fraudulent transactions.

---

# How the System Detects Fraud

The model detects fraud using three mechanisms:

Spectral Defense  
Removes adversarial noise introduced by graph injection attacks.

Attention Based Edge Filtering  
Reduces the influence of suspicious connections using graph attention weights.

Temporal Behavior Modeling  
Captures abnormal transaction sequences and laundering patterns using an LSTM network.

---

# Future Improvements

The current system implements the core architecture. Future enhancements include:

Dual-View Contrastive Learning  
Evidential Deep Learning Head for uncertainty estimation  
GNNExplainer for interpretability  
Reinforcement learning based edge pruning

---
# Author

**Soham Mhetre**  
B.Tech Computer Science (AI & ML)

---


# References

Elliptic Bitcoin Dataset  
PyTorch Geometric Documentation