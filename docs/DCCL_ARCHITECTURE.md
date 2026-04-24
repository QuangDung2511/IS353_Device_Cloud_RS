# Device-Cloud Collaborative Learning (DCCL) - Architecture & Pipeline

This document provides a comprehensive breakdown of the new implementation phases (Phases 4, 5, and 6) to help your team understand the workflow and the required directory structure to build the true Device-Cloud system.

---

## 1. The Big Picture: How the System Works

In a traditional recommendation system, the device sends all user clicks to the Cloud, and the Cloud computes the recommendation. 

In our **DCCL architecture**, we divide the labor to preserve privacy and reduce bandwidth:
1. **The Cloud Server (FastAPI):** Holds the heavy item-item global knowledge graph and the pre-computed embeddings of all products.
2. **The Mobile Device (TFLite script):** Holds the user's personal click history locally. It asks the Cloud *only* for the embeddings of the specific items the user interacted with, then runs the lightweight recommendation model (`SAGEConv`) directly on the phone.

---

## 2. Detailed Pipeline Flow

Here is the step-by-step pipeline across the new phases:

### A. Artifact Extraction & Conversion (Preparation)
- **Input:** Your trained `graphsage_link_pred.pt` model.
- **Process:** 
  - We extract the `Item` embeddings and save them as a dictionary or NumPy array for the Cloud.
  - We isolate the User `SAGEConv` layer and the `DotProductDecoder`. We convert this sub-model into a `.tflite` file so it can run efficiently on mobile.
  
### B. The Cloud API (Phase 4)
- **Process:** The FastAPI server boots up and loads the Item embeddings into memory.
- **API Flow:** It exposes a `GET /api/v1/items/` endpoint. If a device sends `[ItemA, ItemB]`, the server responds with `[VectorA, VectorB]`.

### C. On-Device Inference (Phase 5)
- **Process:** We simulate a user's phone.
- **Flow:**
  1. The phone reads its local database (e.g., a local JSON file) to see what the user clicked recently.
  2. The phone sends an API request to the Cloud to fetch the embeddings for those clicks.
  3. The Cloud returns the dense vectors.
  4. The phone passes these vectors into the `.tflite` model.
  5. The `.tflite` model outputs the final recommendation scores (Top-K Items).

### D. System Demonstration (Phase 6)
- **Process:** A Streamlit dashboard runs to prove the system works.
- **Flow:** The dashboard will have two halves. One half shows the internal workings of the "Phone", the other half shows the "Cloud Server" logs. We will also include a chart proving that downloading embeddings uses less bandwidth than uploading massive graphs.

---

## 3. Required Directory Structure

To keep the project organized and easy to split among teammates, we should create the following new folders in the project root. This separation cleanly divides the "Cloud" logic from the "Device" logic.

```text
Project/
├── data/                       # Existing: Raw and processed graph datasets
├── notebooks/                  # Existing: Jupyter notebooks for training
│
├── scripts/                    # NEW: Scripts for data processing & conversion
│   ├── extract_embeddings.py   # Extracts Item embeddings from PyTorch
│   ├── convert_to_tflite.py    # Converts SAGEConv to ONNX and TFLite
│   └── profile_bandwidth.py    # Calculates payload sizes for Phase 6.1
│
├── cloud_server/               # NEW: Everything related to Phase 4
│   ├── main.py                 # FastAPI application
│   ├── requirements.txt        # Server dependencies (fastapi, uvicorn)
│   └── artifacts/              # Extracted item embeddings (.npy or .pt)
│
├── device_client/              # NEW: Everything related to Phase 5
│   ├── client.py               # Simulated phone script (API calls + Inference)
│   ├── models/                 # The .tflite lightweight model
│   └── local_storage/          # JSON files simulating local device history
│
└── demo/                       # NEW: Everything related to Phase 6
    └── app.py                  # Streamlit Dashboard UI
```

## 4. Teammate Delegation Suggestion

Because of this clear folder structure, you can easily divide the work:
- **Teammate A (Data/ML Engineer):** Focuses on the `scripts/` folder. Handles extracting embeddings and figuring out the TFLite conversion.
- **Teammate B (Backend Engineer):** Focuses on the `cloud_server/` folder. Builds the FastAPI app to serve embeddings quickly.
- **Teammate C (Client/Frontend Engineer):** Focuses on the `device_client/` and `demo/` folders. Writes the simulated API requests, runs the TFLite inference, and builds the Streamlit UI.
