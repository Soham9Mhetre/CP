# Graph Neural Network Based Transaction Fraud Detection

## Overview

This project implements a **Graph Neural Network (GNN) based fraud detection system** for financial transactions using the **Elliptic Bitcoin Transaction Dataset**.

Traditional machine learning models treat transactions independently. However, financial fraud often occurs through **networks of connected transactions**. This project models transactions as a **graph structure**, enabling the detection of suspicious patterns using relationships between transactions.

The system combines **spectral graph filtering and graph neural networks** to detect fraudulent transactions.

---

## Problem Statement

Detecting financial fraud is difficult because:

* Fraudulent transactions are **rare (class imbalance)**
* Fraud often occurs in **transaction chains or clusters**
* Attackers may create **camouflage connections** to hide suspicious activity

This project addresses these challenges using **graph-based learning techniques**.

---

## Dataset

This project uses the **Elliptic Bitcoin Transaction Dataset**.

### Dataset Properties

| Property                 | Value                 |
| ------------------------ | --------------------- |
| Number of transactions   | 203,769               |
| Number of edges          | 234,355               |
| Features per transaction | 165                   |
| Classes                  | Normal (0), Fraud (1) |

Each node represents a **Bitcoin transaction**, and edges represent **money flow between transactions**.

### Dataset Structure

```text
Data(
x=[203769,165],
edge_index=[2,234355],
y=[203769],
train_mask=[203769],
test_mask=[203769]
)
```

Where:

* **x** → node feature matrix
* **edge_index** → graph connections
* **y** → transaction labels

---

## Dataset Setup

The dataset is **not included in this repository** because it exceeds GitHub’s file size limit.

### Step 1 — Download the Dataset

Download the dataset from:

https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

You need a **Kaggle account** to download the dataset.

---

### Step 2 — Extract the Dataset

After downloading, extract the files.

You should obtain the following files:

```
elliptic_txs_features.csv
elliptic_txs_classes.csv
elliptic_txs_edgelist.csv
```

---

### Step 3 — Place the Files in the Project

Create the following directory structure:

```
data/
 └── raw/
```

Place the dataset files inside:

```
data/raw/
    elliptic_txs_features.csv
    elliptic_txs_classes.csv
    elliptic_txs_edgelist.csv
```

Final project structure should look like:

```
CP/
│
models/
│   gcn.py
│   spectral_filter.py
│
training/
│   train.py
│
data/
 └── raw/
     elliptic_txs_features.csv
     elliptic_txs_classes.csv
     elliptic_txs_edgelist.csv
│
requirements.txt
README.md
.gitignore
```

---

## System Architecture

The current pipeline is:

```
Transaction Graph
        ↓
Spectral Graph Filter
        ↓
Graph Convolutional Network (GCN)
        ↓
Weighted Cross Entropy Loss
        ↓
Fraud Classification
```

---

## Model Architecture

```
Input Features (165)
        ↓
Spectral Filter
        ↓
GCN Layer (165 → 64)
        ↓
ReLU Activation
        ↓
GCN Layer (64 → 2)
        ↓
Fraud Prediction
```

---

## Handling Class Imbalance

Fraud cases are significantly fewer than normal transactions.

To address this, the model uses **weighted cross entropy loss**:

```
weights = [1, 9]
```

This penalizes fraud misclassification more heavily.

---

## Installation

Clone the repository:

```
git clone https://github.com/Soham9Mhetre/CP.git
cd CP
```

Create a virtual environment:

```
python -m venv venv
```

Activate the environment.

Windows:

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Project

After placing the dataset in `data/raw/`, run:

```
python -m training.train
```

This script will:

1. Load the transaction graph
2. Train the Graph Neural Network
3. Evaluate fraud detection performance

---

## Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

Example results:

| Metric         | Value |
| -------------- | ----- |
| Accuracy       | ~0.82 |
| Fraud Recall   | ~0.66 |
| Fraud F1 Score | ~0.32 |

The model detects approximately **66% of fraudulent transactions**.

---

## Project Structure

```
CP/
│
models/
│   gcn.py
│   spectral_filter.py
│
training/
│   train.py
│
data/
│   raw/
│
requirements.txt
README.md
.gitignore
```

---

## Future Improvements

Possible extensions include:

* Graph Attention Networks (GAT)
* Temporal transaction modeling
* Contrastive learning
* Uncertainty estimation

---

## Author

Soham Mhetre
Computer Science (AI & ML)

---

## License

This project is intended for **academic and research purposes**.
