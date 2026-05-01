"""FastAPI Cloud Server for item embeddings (Phase 4.2)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager

try:
    # Works when service is started from repository root.
    from cloud_server.retrieval import load_graph_data, get_item_neighbors, get_popular_fallback
except ModuleNotFoundError:
    # Works when deploy root is cloud_server on Render.
    from retrieval import load_graph_data, get_item_neighbors, get_popular_fallback

def _load_embeddings_sync():
    # Helper to call the existing load function from lifespan
    _load_embeddings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_embeddings_sync()
    load_graph_data()
    yield

app = FastAPI(title="DCCL Cloud Server", version="0.1.0", lifespan=lifespan)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
NPY_PATH = ARTIFACT_DIR / "item_embeddings.npy"
JSON_PATH = ARTIFACT_DIR / "item_embeddings.json"

_embeddings: Optional[np.ndarray] = None
_source: Optional[str] = None


def _load_from_json(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return np.empty((0, 0), dtype=np.float32)

    max_id = max(int(item["item_id"]) for item in data)
    dim = len(data[0]["embedding"])
    arr = np.zeros((max_id + 1, dim), dtype=np.float32)

    for item in data:
        idx = int(item["item_id"])
        arr[idx] = np.asarray(item["embedding"], dtype=np.float32)

    return arr


def _load_embeddings() -> None:
    global _embeddings, _source

    if NPY_PATH.exists():
        _embeddings = np.load(NPY_PATH, mmap_mode="r")
        _source = "npy"
        return

    if JSON_PATH.exists():
        _embeddings = _load_from_json(JSON_PATH)
        _source = "json"
        return

    raise FileNotFoundError("No embeddings artifact found in cloud_server/artifacts")


def _parse_item_ids(raw_ids: list[str]) -> list[int]:
    parsed: list[int] = []
    for raw in raw_ids:
        parts = [p for p in raw.split(",") if p.strip()]
        for part in parts:
            try:
                parsed.append(int(part))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid item_id: {part}") from exc

    if not parsed:
        raise HTTPException(status_code=400, detail="item_ids is required")
    return parsed


# The startup event has been replaced by the lifespan context manager above.


@app.get("/health")
def health() -> dict:
    if _embeddings is None:
        return {"status": "error", "detail": "embeddings not loaded"}

    return {
        "status": "ok",
        "source": _source,
        "items": int(_embeddings.shape[0]),
        "dim": int(_embeddings.shape[1]) if _embeddings.ndim > 1 else 0,
    }


@app.get("/api/v1/items/")
def get_item_embeddings(
    item_ids: list[str] = Query(..., description="Item IDs (repeat or comma-separated)"),
    strict: bool = Query(False, description="Return 404 if any item_id is invalid"),
) -> dict:
    if _embeddings is None:
        raise HTTPException(status_code=503, detail="Embeddings not loaded")

    ids = _parse_item_ids(item_ids)
    max_index = _embeddings.shape[0] - 1

    valid: list[int] = []
    missing: list[int] = []
    for idx in ids:
        if 0 <= idx <= max_index:
            valid.append(idx)
        else:
            missing.append(idx)

    if missing and strict:
        raise HTTPException(status_code=404, detail={"missing_ids": missing})

    vectors = _embeddings[valid].tolist() if valid else []

    return {
        "item_ids": valid,
        "embeddings": vectors,
        "missing_ids": missing,
        "dim": int(_embeddings.shape[1]) if _embeddings.ndim > 1 else 0,
    }

@app.get("/api/v1/candidates/")
def retrieve_candidates(
    history_item_ids: list[str] = Query(..., description="User's local interaction history"),
    target_k: int = Query(50, description="Total number of candidates to return")
) -> dict:
    if _embeddings is None:
        raise HTTPException(status_code=503, detail="Embeddings not loaded")

    try:
        history_ids = _parse_item_ids(history_item_ids)
    except HTTPException:
        history_ids = []

    history_set = set(history_ids)
    candidates = []
    
    # Validation
    max_index = _embeddings.shape[0] - 1
    valid_history = [i for i in history_ids if 0 <= i <= max_index]

    if valid_history:
        # Prong 1: Graph Search
        graph_neighbors = get_item_neighbors(valid_history, max_neighbors=20)
        candidates.extend(graph_neighbors)
        
        # Prong 2: Vector Similarity
        # Compute mean embedding of history
        history_embs = _embeddings[valid_history]
        mean_emb = np.mean(history_embs, axis=0)
        
        # Compute cosine similarity (dot product since embeddings should be close to normalized)
        scores = np.dot(_embeddings, mean_emb)
        top_indices = np.argsort(scores)[::-1][:40] # Grab extra in case of overlaps
        
        vector_neighbors = [int(i) for i in top_indices]
        candidates.extend(vector_neighbors)

    # Deduplicate and remove history items
    unique_candidates = []
    seen = set(history_set) # Exclude items user already clicked
    
    for c in candidates:
        if c not in seen and 0 <= c <= max_index:
            unique_candidates.append(c)
            seen.add(c)
            
        if len(unique_candidates) >= target_k:
            break

    # Prong 3: Popularity Fallback (if we still need more items)
    if len(unique_candidates) < target_k:
        needed = target_k - len(unique_candidates)
        fallbacks = get_popular_fallback(seen, count=needed)
        unique_candidates.extend(fallbacks)

    # Final slice to exact target_k (just in case)
    final_candidates = unique_candidates[:target_k]
    
    # Extract embeddings for the final candidates
    final_embeddings = _embeddings[final_candidates].tolist() if final_candidates else []

    return {
        "candidate_ids": final_candidates,
        "embeddings": final_embeddings,
        "dim": int(_embeddings.shape[1]) if _embeddings.ndim > 1 else 0,
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
