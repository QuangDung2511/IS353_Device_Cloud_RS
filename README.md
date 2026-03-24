# Device-Cloud Collaborative Learning for Graph Neural Networks in Modern Recommender Systems

## 1. Problem Introduction
**Background:** Traditional recommender systems upload all user data to the Cloud, which raises privacy concerns and consumes significant transmission resources.

**Solution:** This assignment focuses on decoupling the Graph Neural Network (GNN) model:
- **Cloud:** Stores the massive global graph and learns overarching global patterns (item embeddings).
- **Device (Mobile):** Performs localized computations on a personal subgraph. It updates user preferences in real-time without needing to transmit raw user behavior data back to the server.

## 2. Data
**Dataset:** Amazon Product Data (Review and Metadata).

**Key Characteristics:** 
- The dataset provides user-item interactions (reviews/clicks) along with contextual metadata and timestamps to simulate on-device processing.
- **Visualization Goal:** The project will include visual comparisons showing the massive reduction in data size when transmitting vector embeddings instead of raw interaction data.

**Why the Amazon Dataset?**
It provides rich product metadata for generating high-quality embeddings on the Cloud, and extensive historical user reviews that can be stored and processed dynamically as local subgraphs on the Device.

## 3. Theory & Model
### Collaborative Inference/Training via GraphSAGE
The core architecture is built upon the **GraphSAGE (Graph Sample and AggregatE)** algorithm:
- **Global Knowledge (Cloud):** GraphSAGE is a framework for inductive representation learning on large graphs. It samples and aggregates features from a node's local neighborhood across the entire user-item interaction graph. This allows the Cloud to generate robust, inductive item embeddings.
- **Local Subgraph (Device):** A user "Node" on the device only maintains connections to the items they have recently interacted with (1-depth or 2-depth subgraph). The device receives the pre-trained GraphSAGE item embeddings from the Cloud and applies the aggregation functions locally to infer personalized recommendations for unseen subgraphs.

**Key Reference Papers:**
1. *"Device-Cloud Collaborative Recommendation via Meta Learning"*
2. *"DCCL: Device-Cloud Collaborative Learning for Personalized Recommendation"*
3. *"Walle: An End-to-End, General-Purpose, and Large-Scale On-Device Deep Learning System"*
4. *"CoCorrRec: Device-Cloud Collaborative Correction for On-Device Recommendation"*
5. *"Graph Neural Networks for Recommender Systems: Challenges, Methods, and Directions"*

## 4. Implementation
### Suggested Frameworks & Libraries
- **On-Device Simulation:** **MNN (Mobile Neural Network)** by Alibaba or **TensorFlow Lite**. These lightweight inference engines simulate running the local subgraph encoder directly on mobile devices.
- **Cloud GNN:** **PyTorch Geometric (PyG)**. Used to construct the global GraphSAGE model on the Cloud and generate the embeddings to be pushed down to the local devices.

**Reference Architecture:**
- **Walle (Alibaba):** The first real-world system to deploy full-scale Device-Cloud Collaborative Learning for GNNs. [GitHub - Alibaba Walle Framework](https://github.com/alibaba/MNN)

### Data Structure for Demo
- **Global Knowledge (Cloud):** Maintains the global graph connecting items to learn general features.
- **Local Subgraph (Device):** Each user device stores a subgraph consisting of their direct historical interactions, using GraphSAGE's inductive aggregation to adapt to real-time changes instantly.

---

## Project Structure
- `data/`: Contains raw and processed datasets (e.g., `clean_meta.parquet`, `clean_reviews.csv`). This directory is ignored by Git to save space.
- `notebooks/`: Jupyter Notebooks for Exploratory Data Analysis (EDA), baseline model testing, and visualization.
- `requirements.txt`: Python package dependencies for reproducing the environment.
