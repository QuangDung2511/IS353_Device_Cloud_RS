import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Set

import numpy as np

# Paths
ARTIFACT_ADJ_PATH = Path(__file__).resolve().parent / "artifacts" / "item_adj.json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EDGE_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "item_item_edge_index.pt"

# In-memory graph structures
_adjacency_list: Dict[int, Set[int]] = defaultdict(set)
_popular_fallback: List[int] = []


def load_graph_data() -> None:
    """
    Loads the item-item adjacency list from a lightweight JSON artifact and
    computes the globally popular fallback items based on node degree.

    If the JSON artifact is missing, the function will warn and exit without
    importing heavy dependencies. To generate the JSON from the original
    `.pt` file, run the helper script `scripts/convert_edge_index_to_json.py` locally.
    """
    global _adjacency_list, _popular_fallback

    if ARTIFACT_ADJ_PATH.exists():
        print(f"Loading adjacency JSON from {ARTIFACT_ADJ_PATH}...")
        with ARTIFACT_ADJ_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # data expected as {"item_id": [neighbor_ids, ...], ...}
        _adjacency_list.clear()
        for k, v in data.items():
            try:
                idx = int(k)
            except ValueError:
                continue
            _adjacency_list[idx] = set(int(x) for x in v)

        print(f"Built adjacency list for {len(_adjacency_list)} distinct items (from JSON).")
    else:
        # If JSON missing, do not import torch here (avoid heavy dependency/runtime OOM).
        if EDGE_INDEX_PATH.exists():
            print(f"[WARN] {ARTIFACT_ADJ_PATH} not found, but raw edge index exists at {EDGE_INDEX_PATH}.")
            print("Run scripts/convert_edge_index_to_json.py locally to generate the JSON artifact.")
        else:
            print(f"[WARN] No adjacency artifact found at {ARTIFACT_ADJ_PATH} and no edge index at {EDGE_INDEX_PATH}.")
        return

    # Compute Popular Fallback (Top 100 items by degree)
    degrees = [(item, len(neighbors)) for item, neighbors in _adjacency_list.items()]
    degrees.sort(key=lambda x: x[1], reverse=True)
    _popular_fallback = [item for item, degree in degrees[:100]]
    print(f"Cached Top 100 popular items for fallback.")


def get_item_neighbors(item_ids: List[int], max_neighbors: int = 20) -> List[int]:
    """Prong 1: Graph Search (1-hop neighbors)"""
    neighbors = set()
    for item_id in item_ids:
        neighbors.update(_adjacency_list.get(item_id, set()))

    # Remove the seed items themselves
    for item_id in item_ids:
        neighbors.discard(item_id)

    # Cap at max_neighbors
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
