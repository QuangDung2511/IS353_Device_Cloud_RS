"""Script to extract local history for device simulation (Phase 5.1)"""
import json
import os
from pathlib import Path

import pandas as pd

def extract_history(parquet_path: Path, user_map_path: Path, item_map_path: Path, output_dir: Path):
    """
    Extracts the 1-hop interaction history for each unseen user and saves it as JSON
    to simulate local device storage.
    """
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    print("Loading ID mappings...")
    with open(user_map_path, "r") as f:
        user_mapping = json.load(f)
    with open(item_map_path, "r") as f:
        item_mapping = json.load(f)
        
    print("Mapping item string IDs to integer IDs...")
    # Map item IDs. If an item ID is missing from the mapping, we drop it
    df["item_id"] = df["asin"].map(item_mapping)
    
    # Drop rows with unmapped item IDs
    df = df.dropna(subset=["item_id"])
    
    # Convert to int after dropping NAs
    df["item_id"] = df["item_id"].astype(int)
    
    print(f"Grouping interactions for {df['reviewerID'].nunique()} unique unseen users...")
    grouped = df.groupby("reviewerID")["item_id"].apply(list).reset_index()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for _, row in grouped.iterrows():
        uid = str(row["reviewerID"])
        history = [int(i) for i in row["item_id"]]
        
        # Deduplicate history while preserving order
        history = list(dict.fromkeys(history))
        
        file_path = output_dir / f"user_{uid}.json"
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"user_id": uid, "history": history}, f, indent=2)
            
        count += 1
        
    print(f"Successfully generated {count} local storage files in {output_dir}")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "hidden_interactions_test.parquet"
    USER_MAP_PATH = PROJECT_ROOT / "data" / "processed" / "user_mapping.json"
    ITEM_MAP_PATH = PROJECT_ROOT / "data" / "processed" / "item_mapping.json"
    OUTPUT_DIR = PROJECT_ROOT / "device_client" / "local_storage"
    
    extract_history(PARQUET_PATH, USER_MAP_PATH, ITEM_MAP_PATH, OUTPUT_DIR)
