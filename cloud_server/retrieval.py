import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Set

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EDGE_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "item_item_edge_index.pt"

# In-memory graph structures
_adjacency_list: Dict[int, Set[int]] = defaultdict(set)
_popular_fallback: List[int] = []
_loaded = False

def load_graph_data() -> None:
    """
    Loads the item-item edge index, builds the adjacency list,
    and computes the globally popular fallback items based on node degree.
    """
    global _adjacency_list, _popular_fallback, _loaded

    if _loaded:
        return
    
    if not EDGE_INDEX_PATH.exists():
        print(f"[WARN] Edge index not found at {EDGE_INDEX_PATH}. Graph retrieval will be empty.")
        _loaded = True
        return

    import torch

    print(f"Loading item-item graph from {EDGE_INDEX_PATH}...")
    edge_index = torch.load(EDGE_INDEX_PATH, map_location="cpu", weights_only=False)
    
    # edge_index is [2, num_edges]
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    
    # Reset
    _adjacency_list.clear()
    
    # We want undirected graph for neighbor lookup (since 'also_bought' goes both ways logically)
    for u, v in zip(src, dst):
        u, v = int(u), int(v)
        _adjacency_list[u].add(v)
        _adjacency_list[v].add(u)
        
    print(f"Built adjacency list for {len(_adjacency_list)} distinct items.")
    
    # Compute Popular Fallback (Top 100 items by degree)
    degrees = [(item, len(neighbors)) for item, neighbors in _adjacency_list.items()]
    degrees.sort(key=lambda x: x[1], reverse=True)
    
    _popular_fallback = [item for item, degree in degrees[:100]]
    print(f"Cached Top 100 popular items for fallback.")
    _loaded = True

def get_item_neighbors(item_ids: List[int], max_neighbors: int = 20) -> List[int]:
    """Prong 1: Graph Search (1-hop neighbors)"""
    neighbors = set()
    for item_id in item_ids:
        neighbors.update(_adjacency_list.get(item_id, set()))
        
    # Remove the seed items themselves
    for item_id in item_ids:
        neighbors.discard(item_id)
        
    # Cap at max_neighbors (we could sort by something, but random/arbitrary set pop is fine for now)
    return list(neighbors)[:max_neighbors]

def get_popular_fallback(exclude_items: Set[int], count: int) -> List[int]:
    """Prong 3: Popularity Fallback"""
    candidates = []
    for item in _popular_fallback:
        if item not in exclude_items:
            candidates.append(item)
        if len(candidates) >= count:
            break
    return candidates
