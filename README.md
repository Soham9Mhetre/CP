# Graph Neural Network Based Transaction Fraud Detection

## Overview

This project implements a **Graph Neural Network (GNN) based fraud detection system** for financial transactions using the **Elliptic Bitcoin Transaction Dataset**.

Traditional machine learning models treat transactions independently, but financial fraud often occurs through **networks of connected transactions**. This project models transactions as a **graph structure**, enabling the detection of suspicious patterns through relationships between transactions.

The system combines **spectral graph filtering and graph neural networks** to identify fraudulent transactions more effectively.

---

## Problem Statement

Financial fraud detection is challenging because:

* Fraudulent transactions are **rare (class imbalance)**.
* Fraud often occurs in **groups or transaction chains**.
* Attackers may create **camouflage connections** to hide illegal activity.

This project addresses these challenges using **graph-based learning techniques**.

---

## Dataset

The model is trained on the **Elliptic Bitcoin Transaction Dataset**.

### Dataset Properties

| Property                       | Value                 |
| ------------------------------ | --------------------- |
| Number of transactions (nodes) | 203,769               |
| Number of edges (connections)  | 234,355               |
| Features per transaction       | 165                   |
| Classes                        | Normal (0), Fraud (1) |

Each node represents a **Bitcoin transaction**, and edges represent **transaction flow between nodes**.

### Dataset Structure

```
Data(
x=[203769,165],
edge_index=[2,234355],
y=[203769],
train_mask=[203769],
test_mask=[203769]
)
```

Where:

* `x` = Node feature matrix
* `edge_index` = Graph connectivity matrix
* `y` = Transaction labels

---

## System Architecture

The current system pipeline is:

```
Transaction Graph
        ↓
Spectral Graph Filter
        ↓
Graph Convolutional Network (GCN)
        ↓
Weighted Cross Entropy Training
        ↓
Fraud Classification
```

### 1. Spectral Graph Filter

A spectral filtering layer smooths node features across the graph using Laplacian-based message passing. This reduces graph noise and improves feature consistency across connected transactions.

### 2. Graph Convolutional Network

The model uses Graph Convolution layers to aggregate information from neighboring transactions.

Architecture:

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

### 3. Class Imbalance Handling

Fraud cases are significantly fewer than normal transactions. To address this, the model uses **weighted cross entropy loss**.

```
weights = [1, 9]
```

This penalizes fraud misclassification more heavily.

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
│   (dataset should be placed here locally)
│
requirements.txt
README.md
.gitignore
```

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

Activate the environment:

Windows:

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Dataset Setup

The dataset is not included in the repository due to GitHub file size limits.

Download the **Elliptic Bitcoin dataset** and place the files inside:

```
data/raw/
```

---

## Running the Project

Run the training script:

```
python -m training.train
```

This will:

1. Load the transaction graph
2. Train the GNN model
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

The model successfully detects approximately **66% of fraudulent transactions**.

---

## Key Features

* Graph-based fraud detection
* Spectral graph filtering
* Graph Neural Network architecture
* Handling of class imbalance
* GPU accelerated training

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

This project is for academic and research purposes.
