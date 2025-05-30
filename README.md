# Temporal Link Prediction Using Graph Neural Networks

**Author:** Leman Zakaryayeva  
**Affiliation:** Department of Computer Engineering, Akdeniz University, Antalya, Turkey

## Overview

This project addresses **Temporal Link Prediction**, the task of forecasting the appearance of edges between nodes in a time‐evolving graph. We employ a combined Graph Convolutional Network (GCN) and Gated Recurrent Unit (GRU) model—**GConvGRU**—to capture both the spatial and temporal dynamics of the data. The model is trained and evaluated on two real‐world datasets; initial results point to promising directions but also indicate room for significant improvement.

---

## Introduction

Real‐world systems (social networks, communication graphs, etc.) often exhibit relationships that change over time. Predicting which new connections will form—Temporal Link Prediction—can power recommendation systems, anomaly detection, and more. In this work, we apply Graph Neural Networks (GNNs) to learn from past interactions and predict future edge formation.

---

## Methodology

### Data Preprocessing

- **Missing Node Features**  
  Replace any placeholder values (e.g., `-1`) in node feature vectors with `0` to handle missing data.

- **Edge‐Type Feature Aggregation**  
  For each edge type, compute the average of its associated features, mitigating sparsity.

- **Valid‐Node Filtering**  
  Retain only those nodes with complete feature vectors to ensure consistency in model input.

### Model Architecture

We implement **GConvGRU** using PyTorch Geometric Temporal:

1. **Graph Convolution (GCN)** layers capture structural information at each time step.  
2. **Gated Recurrent Units (GRU)** integrate temporal dependencies across successive graph snapshots.  
3. Final output layer produces a probability for each candidate edge.

### Training Process

- **Hyperparameter Tuning**  
  Learning rate, number of epochs, and other settings were tuned via grid search.

- **Optimizer & Scheduler**  
  Adopted the Adam optimizer (initial lr = 3e-4) with a dynamic learning‐rate scheduler.

- **Loss Function**  
  Binary Cross-Entropy with logits, weighted to counter class imbalance (few positive edges vs. many negatives).

### Evaluation

- **Metric:** AUC (Area Under the ROC Curve)  
- Predictions are scored on held‐out test sets, measuring ability to distinguish future edges.

### Optimization

Future enhancements include:

- Experimenting with **Graph Attention Networks (GAT)** or deeper GCN stacks.  
- Increasing dropout, batch‐norm layers, and exploring alternative loss formulations.  
- Advanced temporal feature engineering (e.g., time‐difference embeddings) and data augmentation.

---

## Results

| Dataset   | AUC Score |
|-----------|-----------|
| Dataset A | 0.5004    |

An AUC near 0.50 indicates performance comparable to random guessing—highlighting the need for more sophisticated architectures and tuning.

---

## Discussion & Future Work

- **Architectural Variants:** Try GAT, deeper GCN, or Transformer‐based modules.  
- **Hyperparameter Search:** Broaden the search space for learning rates, dropout, and hidden sizes.  
- **Temporal Augmentation:** Introduce sliding‐window or time‐warp data augmentation.  
- **Ensemble Methods:** Combine multiple GNN predictions to boost robustness.

---

## Conclusion

We have demonstrated a baseline application of **GConvGRU** for Temporal Link Prediction. While initial results are modest, the framework provides a solid foundation. Ongoing work will refine model design and training strategies to achieve stronger predictive performance.


