import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import requests

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

logger = logging.getLogger(__name__)

class DeviceClient:
    """
    Simulated mobile client for Device-Cloud Collaborative Learning (DCCL).
    Handles reading local interactions, requesting item embeddings from the Cloud,
    and running the lightweight SAGEConv user-branch via TFLite.
    """
    def __init__(self, user_id: str, local_storage_dir: str, model_path: str, cloud_api_url: str):
        self.user_id = user_id
        self.local_storage_dir = Path(local_storage_dir)
        self.model_path = Path(model_path)
        self.cloud_api_url = cloud_api_url
        
        self.in_channels = 256
        self.out_channels = 256
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"TFLite model not found at {self.model_path}")
            
        # Initialize Interpreter (allocation is deferred until shapes are known)
        self.interpreter = tflite.Interpreter(model_path=str(self.model_path))
        
    def _read_local_history(self) -> List[int]:
        """Step A: Read local interaction history (Item IDs) from JSON."""
        file_path = self.local_storage_dir / f"user_{self.user_id}.json"
        if not file_path.exists():
            return []
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        return data.get("history", [])

    def recommend(self, candidate_item_ids: List[int], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Executes the DCCL Phase 5 inference pipeline.
        Returns a list of (item_id, score) tuples.
        """
        # Step A: Read local history
        history_item_ids = self._read_local_history()
        
        # Step B: Fetch embeddings from Cloud
        all_ids_to_fetch = list(set(history_item_ids + candidate_item_ids))
        if not all_ids_to_fetch:
            return []
            
        params = {"item_ids": ",".join(map(str, all_ids_to_fetch))}
        response = requests.get(f"{self.cloud_api_url}/api/v1/items/", params=params)
        response.raise_for_status()
        
        data = response.json()
        fetched_ids = data.get("item_ids", [])
        fetched_embeddings = data.get("embeddings", [])
        
        # Map item_id -> embedding
        emb_map = {
            int(iid): np.array(emb, dtype=np.float32) 
            for iid, emb in zip(fetched_ids, fetched_embeddings)
        }
        
        # Separate neighbors (history) from candidates
        neighbor_embs = [emb_map[iid] for iid in history_item_ids if iid in emb_map]
        valid_candidate_ids = [iid for iid in candidate_item_ids if iid in emb_map]
        candidate_embs = [emb_map[iid] for iid in valid_candidate_ids]
        
        if not valid_candidate_ids:
            return []
            
        if not neighbor_embs:
            # Fallback if no history: provide a zero vector
            neighbor_embs = [np.zeros(self.in_channels, dtype=np.float32)]
            
        # Prepare inputs for TFLite
        user_x = np.zeros((1, self.in_channels), dtype=np.float32)
        neighbor_x = np.stack(neighbor_embs).astype(np.float32)
        candidate_emb = np.stack(candidate_embs).astype(np.float32)
        
        K = neighbor_x.shape[0]
        N = candidate_emb.shape[0]
        
        # Step C: Feed into TFLite interpreter
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        # Dynamically resize the input tensors to match K and N
        for d in input_details:
            name = d["name"]
            if "neighbor" in name:
                self.interpreter.resize_tensor_input(d["index"], [K, self.in_channels])
            elif "candidate" in name:
                self.interpreter.resize_tensor_input(d["index"], [N, self.out_channels])
                
        self.interpreter.allocate_tensors()
        
        # Set tensor data
        for d in input_details:
            name = d["name"]
            if "user_x" in name:
                self.interpreter.set_tensor(d["index"], user_x)
            elif "neighbor_x" in name:
                self.interpreter.set_tensor(d["index"], neighbor_x)
            elif "candidate_emb" in name:
                self.interpreter.set_tensor(d["index"], candidate_emb)
                
        # Step D: Inference and calculate Top-K
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(output_details[0]["index"])
        
        # Numpy argsort is ascending, so we reverse it
        top_indices = np.argsort(logits)[::-1][:top_k]
        
        results = [
            (valid_candidate_ids[idx], float(logits[idx]))
            for idx in top_indices
        ]
        
        return results

if __name__ == "__main__":
    # Example usage / Smoke test
    import argparse
    import random
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=str, required=True, help="User ID (e.g., A1GQRGB8FGSLIZ)")
    parser.add_argument("--cloud_url", type=str, default="http://localhost:8000", help="FastAPI Server URL")
    args = parser.parse_args()
    
    root_dir = Path(__file__).resolve().parent.parent
    storage_dir = root_dir / "device_client" / "local_storage"
    model_file = root_dir / "device_client" / "models" / "user_sage_decoder.tflite"
    
    # Load real items to act as candidates
    item_map_path = root_dir / "data" / "processed" / "item_mapping.json"
    with open(item_map_path, "r", encoding="utf-8") as f:
        item_mapping = json.load(f)
        
    all_item_ids = list(item_mapping.values())
    # Randomly sample 50 candidate items to rank
    candidates = random.sample(all_item_ids, k=min(50, len(all_item_ids)))
    
    client = DeviceClient(
        user_id=args.user_id,
        local_storage_dir=str(storage_dir),
        model_path=str(model_file),
        cloud_api_url=args.cloud_url
    )
    
    print(f"Running on-device inference for User: {args.user_id}...")
    print(f"Evaluating {len(candidates)} real candidate items...")
    
    try:
        recommendations = client.recommend(candidate_item_ids=candidates, top_k=5)
        print("\nTop 5 Recommendations:")
        for rank, (item_id, score) in enumerate(recommendations, 1):
            # Reverse lookup the item string ID for better readability
            item_str_id = next((k for k, v in item_mapping.items() if v == item_id), "Unknown")
            print(f"{rank}. Item {item_str_id} (Integer ID: {item_id}) | Score: {score:.4f}")
    except Exception as e:
        print(f"Error during inference: {e}")
