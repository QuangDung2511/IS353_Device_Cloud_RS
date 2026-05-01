"""
Convert `data/processed/item_item_edge_index.pt` (PyTorch edge_index) into a
lightweight JSON adjacency list at `cloud_server/artifacts/item_adj.json`.

Run locally where `torch` is available:

python scripts/convert_edge_index_to_json.py

This script does not modify repository files except writing the JSON artifact.
"""
from pathlib import Path
import json
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EDGE_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "item_item_edge_index.pt"
OUT_PATH = PROJECT_ROOT / "cloud_server" / "artifacts" / "item_adj.json"

if not EDGE_INDEX_PATH.exists():
    raise SystemExit(f"Edge index not found: {EDGE_INDEX_PATH}")

print(f"Loading edge index from {EDGE_INDEX_PATH}...")
edge_index = torch.load(EDGE_INDEX_PATH, map_location="cpu")

# Expect edge_index as tensor shape [2, num_edges]
src = edge_index[0].numpy()
dst = edge_index[1].numpy()

adj = {}
for u, v in zip(src, dst):
    u_i = int(u)
    v_i = int(v)
    adj.setdefault(str(u_i), set()).add(v_i)
    adj.setdefault(str(v_i), set()).add(u_i)

# Convert sets to lists for JSON
adj_list = {k: list(v) for k, v in adj.items()}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(adj_list, f)

print(f"Wrote adjacency JSON to {OUT_PATH} ({len(adj_list)} nodes)")
