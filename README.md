# 🚨 Crypto Fraud Prevention System (Graph + Temporal + Contrastive Learning)

## 📌 Overview

This project implements a **real-time fraud prevention system** using deep learning techniques that combine **graph modeling**, **temporal learning**, and **dual-view contrastive learning**.

Unlike traditional systems that only classify transactions, this system **takes actionable decisions** such as allowing, blocking, or requesting verification — making it a true **fraud prevention pipeline**.

---

## 🎯 Key Features

* Graph-based modeling using **Graph Attention Networks (GAT)**
* Temporal modeling using **LSTM**
* Dual-view contrastive learning
* Dynamic thresholding (AUC-PR optimized)
* Multi-level decision system:

  * ALLOW
  * OTP
  * BLOCK
  * SEND TO ANALYST
* Handles imbalanced fraud data
* Optimized for speed and stability

---

## 📊 Performance

* Accuracy: ~93%
* Fraud Precision: ~0.49
* Fraud Recall: ~0.55
* Fraud F1 Score: ~0.52

---

## 🧠 System Pipeline (Flow Diagram)

```
Raw Transaction Data
        ↓
Graph Construction (Nodes + Edges)
        ↓
Spectral Filtering
        ↓
Temporal Modeling (LSTM)
        ↓
Graph Attention Network (GAT)
        ↓
Dual-View Contrastive Learning
        ↓
Fraud Probability
        ↓
Dynamic Thresholding
        ↓
Decision Engine
        ↓
ALLOW / OTP / BLOCK / ANALYST
```

---

## 🏗️ System Architecture

```
            ┌──────────────────────┐
            │  Transaction Input   │
            └─────────┬────────────┘
                      ↓
            ┌──────────────────────┐
            │ Graph Construction   │
            └─────────┬────────────┘
                      ↓
            ┌──────────────────────┐
            │ Spectral Filtering   │
            └─────────┬────────────┘
                      ↓
        ┌─────────────┴─────────────┐
        ↓                           ↓
 ┌──────────────┐           ┌──────────────┐
 │   GAT (Graph)│           │  LSTM (Time) │
 └──────┬───────┘           └──────┬───────┘
        ↓                          ↓
        └──────────┬───────────────┘
                   ↓
        ┌──────────────────────────┐
        │ Contrastive Learning     │
        └─────────┬────────────────┘
                  ↓
        ┌──────────────────────────┐
        │ Fraud Probability Output │
        └─────────┬────────────────┘
                  ↓
        ┌──────────────────────────┐
        │ Decision Engine          │
        └─────────┬────────────────┘
                  ↓
   ┌────────┬────────┬────────┬────────┐
   │ ALLOW  │  OTP   │ BLOCK  │ ANALYST│
   └────────┴────────┴────────┴────────┘
```

---

## ⚙️ Model Architecture

### Graph View (GAT)

Captures relationships between transactions and detects network-based fraud.

### Temporal View (LSTM)

Captures behavioral patterns over time.

### Dual-View Learning

Aligns:
Graph Embedding ≈ Temporal Embedding

---

## 🔬 Mathematical Model

### Graph Attention

hᵢ' = σ( Σⱼ∈N(i) αᵢⱼ · W · hⱼ )

---

### Temporal Modeling

hₜ = LSTM(xₜ, hₜ₋₁)

z_temp = Linear(h_T)

---

### Contrastive Loss

L_contrast = || z_graph − z_temp ||²

---

### Classification Loss

L_cls = − Σ y log(ŷ)

---

### Total Loss

L = L_cls + λ · L_contrast

---

## 🧮 Decision Engine

Threshold selected using F1 optimization:
threshold = argmax(F1)

Adjusted:
threshold = 0.75 × threshold

Decision Rules:

* High probability → BLOCK
* Medium → OTP
* Low → ALLOW
* High uncertainty → ANALYST

---

## ⚡ Optimizations

* Removed redundant data copying
* Precomputed graph edges
* Reduced spectral computation frequency
* Class-weighted loss
* Dropout regularization
* Efficient GPU usage

---

## 🚀 Run the Project

```
python -m training.train
```

---

## 🧠 Key Insight

Traditional systems:
Prediction → End

This system:
Prediction → Risk Evaluation → Action

---

## 🔮 Future Scope

* Reinforcement learning for adaptive decisions
* Real-time streaming fraud detection
* Explainable AI integration
* Multi-agent fraud systems

---

## 👨‍💻 Author

Soham Mhetre
B.Tech CSE (AIML)
