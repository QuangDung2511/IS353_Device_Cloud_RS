import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cloud_server import retrieval
from cloud_server.main import app, _load_embeddings, _embeddings
from fastapi.testclient import TestClient

client = TestClient(app)

def test_load_graph_data():
    # Force load data
    retrieval.load_graph_data()
    
    # Check if adjacency list is populated
    assert len(retrieval._adjacency_list) > 0, "Adjacency list should not be empty"
    
    # Check popular fallback
    assert len(retrieval._popular_fallback) == 100, "Should exactly cache 100 popular fallback items"
    
    # Verify elements are distinct
    assert len(set(retrieval._popular_fallback)) == 100, "Fallback items must be distinct"

def test_get_item_neighbors():
    retrieval.load_graph_data()
    
    if len(retrieval._popular_fallback) == 0:
        pytest.skip("Graph data not found, skipping neighbor test")
        
    # Get a known item with edges. The most popular item definitely has edges!
    top_item = retrieval._popular_fallback[0]
    
    # Request neighbors
    neighbors = retrieval.get_item_neighbors([top_item], max_neighbors=15)
    
    # Top item should have neighbors
    assert len(neighbors) > 0
    assert len(neighbors) <= 15
    
    # The item itself should not be in its own neighbors
    assert top_item not in neighbors

def test_get_popular_fallback():
    retrieval.load_graph_data()
    
    exclude_set = {retrieval._popular_fallback[0], retrieval._popular_fallback[1]}
    
    candidates = retrieval.get_popular_fallback(exclude_set, count=10)
    
    assert len(candidates) == 10
    assert retrieval._popular_fallback[0] not in candidates
    assert retrieval._popular_fallback[1] not in candidates

def test_endpoint_candidates():
    # Force load components manually for testing
    retrieval.load_graph_data()
    _load_embeddings()
    
    from cloud_server import main as cloud_main
    
    if cloud_main._embeddings is None:
        pytest.skip("Embeddings not found")
        
    # Get a popular item to act as history
    history_id = retrieval._popular_fallback[0]
    
    # Test request
    response = client.get(f"/api/v1/candidates/?history_item_ids={history_id}&target_k=30")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "candidate_ids" in data
    assert "embeddings" in data
    assert "dim" in data
    
    assert len(data["candidate_ids"]) == 30
    assert len(data["embeddings"]) == 30
    assert data["dim"] == 256
    
    # Ensure history item isn't in candidates
    assert history_id not in data["candidate_ids"]
