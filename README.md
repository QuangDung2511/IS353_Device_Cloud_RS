# Device-Cloud Collaborative Learning (DCCL) for Graph Neural Networks (GNNs)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![GNN](https://img.shields.io/badge/GNN-GraphSAGE-green.svg)](https://pytorch-geometric.readthedocs.io/)

This repository implements a professional **Device-Cloud Collaborative Learning (DCCL)** framework for modern recommender systems. By decoupling GNN components between the Cloud (global item knowledge) and Device (local user subgraphs), we achieve high-performance recommendations while prioritizing user privacy and reducing data transmission costs.

---

## 1. The Problem: The Privacy-Efficiency Tradeoff

Traditional recommender systems rely on a **Cloud-Centric** architecture where all user interaction data (clicks, views, reviews) is uploaded to a central server. This approach faces three critical challenges:

*   **Privacy Risks**: Uploading raw user behavior logs exposes sensitive personal information to interceptors or server breaches.
*   **Latency & Real-time Adaptivity**: Centralized models often struggle to adapt to instantaneous user interest shifts due to the round-trip time of data processing and model inference.
*   **Transmission Overhead**: Continuously transmitting massive streams of interaction logs consumes significant bandwidth for both the terminal (mobile device) and the server.

### The Solution: DCCL
Our framework addresses these issues by **decoupling the inference pipeline**:
*   **Cloud**: Learns global item embeddings from a massive catalog and overarching global patterns.
*   **Device**: Performs on-device computation using a **local subgraph** of the user's recent history. Only vector embeddings (not raw logs) are transmitted, significantly reducing bandwidth and enhancing privacy.

---

## 2. The Method: Hybrid Heterogeneous GNNs

The core architecture utilizes **GraphSAGE (Graph Sample and AggregatE)** within a heterogeneous graph environment to handle the complex relationships between users, items, and reviews.

### Graph Architecture
We model the recommendation task as a link prediction problem on a bipartite heterogeneous graph:
*   **Nodes**: `User` and `Item`.
*   **Edges**: `(User, Reviews, Item)` and `(Item, Also_Bought, Item)`.
*   **Message Passing**: We implement **Reverse Edges** `(Item, Rev_Reviews, User)` to ensure bipartite information flow during aggregation.

### The GraphSAGE Advantage
Unlike traditional GCNs that learn transductive node-specific embeddings, **GraphSAGE** learns **aggregation functions**. This is crucial for DCCL because:
1.  **Inductive Learning**: It generalizes to **unseen nodes** (hidden/new users) by aggregating features from their localized neighborhood.
2.  **On-Device Efficiency**: Devices only need to store the user's immediate 1-2 hop neighborhood, allowing for low-memory sub-graph inference.

**Mathematical Formulation**:
$$h_v^k = \sigma(W^k \cdot \text{concat}(h_v^{k-1}, \text{aggr}(\{h_u^{k-1}, \forall u \in \mathcal{N}(v)\})))$$
Where $h_v^k$ is the embedding of node $v$ at layer $k$, and the `mean` aggregator is used to synthesize neighbor information.

---

## 3. The Result: Performance & Inductive Capability

We evaluated the framework using the **Amazon Reviews (Baby Products)** dataset, comparing our Inductive GraphSAGE approach against a baseline Global GCN.

### Key Metrics (Link Prediction)

| Model | User Split | AUROC | Average Precision (AP) |
| :--- | :--- | :--- | :--- |
| **GraphSAGE (DCCL)** | **Seen Users** | **0.9817** | **0.9720** |
| **GraphSAGE (DCCL)** | **Hidden Users** | **0.9362** | **0.9202** |
| GCN Baseline | Seen Users | 0.9780 | 0.9650 |
| Random Baseline | - | 0.5128 | 0.5000 |

### Key Findings
*   **Inductive Generalization**: GraphSAGE shows a minimal "Inductive Gap," maintaining high accuracy (0.93+ AUROC) even for users never seen during the Cloud training phase.
*   **Scalability**: The use of `LinkNeighborLoader` allows the system to train on massive graphs by sampling manageable mini-batches of subgraphs.

---

## 4. How to Run

### 1. Prerequisites
Ensure you have Python 3.11+ and a virtual environment set up:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Preparation
Run the engineering notebooks to download and process the Amazon dataset:
- `01_data_engineering.ipynb`: Cleans raw review/metadata.
- `02_feature_encoding.ipynb`: Generates text/category embeddings.

### 3. Model Training & Evaluation
Execute the GNN pipeline in order:
- `03_hetero_data.ipynb` to `05_edge_decoder.ipynb`: Build the Heterogeneous graph data.
- `06_graphsage_train_eval.ipynb`: Train the DCCL model.
- `08_model_comparision.ipynb`: Generate final performance reports and visualizations.

---

## Project Structure
- `data/`: Processed parquets and `.pt` graph tensors (ignored by Git).
- `notebooks/`: Modular pipeline from data to model evaluation.
- `TODO.md`: Roadmap for future on-device deployment (MNN/TFLite).
- `requirements.txt`: Project dependencies.
