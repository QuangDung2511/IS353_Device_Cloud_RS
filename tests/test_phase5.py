import json
import os
import shutil
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from scripts.extract_local_history import extract_history
import sys
# Add project root to path so we can import device_client
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from device_client.client import DeviceClient
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOCK_DATA_DIR = PROJECT_ROOT / "tests" / "mock_data"
MOCK_STORAGE_DIR = PROJECT_ROOT / "tests" / "mock_storage"

@pytest.fixture(scope="session", autouse=True)
def setup_mock_data():
    MOCK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MOCK_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Mock user and item mappings
    user_mapping = {"userA": 0, "userB": 1, "userC": 2}
    item_mapping = {"itemX": 100, "itemY": 101, "itemZ": 102}
    
    with open(MOCK_DATA_DIR / "user_mapping.json", "w") as f:
        json.dump(user_mapping, f)
        
    with open(MOCK_DATA_DIR / "item_mapping.json", "w") as f:
        json.dump(item_mapping, f)
        
    # Mock hidden interactions
    df = pd.DataFrame({
        "reviewerID": ["userA", "userA", "userB", "userC", "userC", "userC"],
        "asin": ["itemX", "itemY", "itemX", "itemY", "itemZ", "itemX"]
    })
    df.to_parquet(MOCK_DATA_DIR / "hidden_interactions_test.parquet")
    
    yield
    
    # Teardown
    shutil.rmtree(MOCK_DATA_DIR, ignore_errors=True)
    shutil.rmtree(MOCK_STORAGE_DIR, ignore_errors=True)

def test_extract_local_history_generates_json():
    # Run the extraction function
    extract_history(
        parquet_path=MOCK_DATA_DIR / "hidden_interactions_test.parquet",
        user_map_path=MOCK_DATA_DIR / "user_mapping.json",
        item_map_path=MOCK_DATA_DIR / "item_mapping.json",
        output_dir=MOCK_STORAGE_DIR
    )
    
    # Check if files are created for each user
    assert (MOCK_STORAGE_DIR / "user_userA.json").exists()
    assert (MOCK_STORAGE_DIR / "user_userB.json").exists()
    assert (MOCK_STORAGE_DIR / "user_userC.json").exists()
    
    # Check content of userA
    with open(MOCK_STORAGE_DIR / "user_userA.json", "r") as f:
        data_0 = json.load(f)
        
    assert "user_id" in data_0
    assert "history" in data_0
    assert data_0["user_id"] == "userA"
    assert isinstance(data_0["history"], list)
    assert set(data_0["history"]) == {100, 101}
    
    # Check content of userB
    with open(MOCK_STORAGE_DIR / "user_userB.json", "r") as f:
        data_1 = json.load(f)
    assert data_1["user_id"] == "userB"
    assert set(data_1["history"]) == {100}

def test_tflite_model_exists():
    # Simple check for task 5.1 completion requirement
    model_path = PROJECT_ROOT / "device_client" / "models" / "user_sage_decoder.tflite"
    assert model_path.exists(), "TFLite model should exist as part of Phase 5.1"

@mock.patch("device_client.client.requests.get")
def test_device_client_inference(mock_get):
    model_path = PROJECT_ROOT / "device_client" / "models" / "user_sage_decoder.tflite"
    
    # Mock Cloud API response
    mock_response = mock.Mock()
    mock_response.json.return_value = {
        "item_ids": [100, 101, 200, 201],
        "embeddings": np.random.randn(4, 256).tolist(),
        "dim": 256
    }
    mock_response.raise_for_status = mock.Mock()
    mock_get.return_value = mock_response
    
    # Init client for userA
    client = DeviceClient(
        user_id="userA",
        local_storage_dir=str(MOCK_STORAGE_DIR),
        model_path=str(model_path),
        cloud_api_url="http://mock-cloud"
    )
    
    candidates = [200, 201]
    recommendations = client.recommend(candidate_item_ids=candidates, top_k=2)
    
    # Verification
    mock_get.assert_called_once()
    assert len(recommendations) == 2
    assert isinstance(recommendations[0], tuple)
    assert recommendations[0][0] in candidates
    # Scores should be floats
    assert isinstance(recommendations[0][1], float)
