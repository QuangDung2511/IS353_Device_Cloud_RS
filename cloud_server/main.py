"""FastAPI Cloud Server for DCCL item retrieval and demo recommendations."""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from contextlib import asynccontextmanager
from html import unescape
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from cloud_server.retrieval import (
    get_item_neighbors,
    get_popular_fallback,
    load_graph_data,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
NPY_PATH = ARTIFACT_DIR / "item_embeddings.npy"
JSON_PATH = ARTIFACT_DIR / "item_embeddings.json"
ITEM_MAPPING_PATH = PROJECT_ROOT / "data" / "processed" / "item_mapping.json"
ITEM_METADATA_CSV = PROJECT_ROOT / "data" / "clean_reviews.csv"
LOCAL_STORAGE_DIR = PROJECT_ROOT / "device_client" / "local_storage"
TFLITE_MODEL_PATH = PROJECT_ROOT / "device_client" / "models" / "user_sage_decoder.tflite"
GRAPH_SAGE_CHECKPOINT_PATH = PROJECT_ROOT / "data" / "processed" / "graphsage_link_pred.pt"

_embeddings: Optional[np.ndarray] = None
_source: Optional[str] = None
_item_id_to_asin: Optional[dict[int, str]] = None
_item_metadata: Optional[dict[str, dict[str, str]]] = None
_tflite_interpreter: Any = None
_graph_load_attempted = False
_torch_branch_weights: Optional[dict[str, np.ndarray]] = None


def _load_embeddings_sync() -> None:
    _load_embeddings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_embeddings_sync()
    yield


app = FastAPI(title="DCCL Cloud Server", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _require_embeddings() -> np.ndarray:
    if _embeddings is None:
        raise HTTPException(status_code=503, detail="Embeddings not loaded")
    return _embeddings


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


def _load_item_maps() -> tuple[dict[int, str], dict[str, dict[str, str]]]:
    global _item_id_to_asin, _item_metadata

    if _item_id_to_asin is None:
        with ITEM_MAPPING_PATH.open("r", encoding="utf-8") as f:
            asin_to_item_id = json.load(f)
        _item_id_to_asin = {int(item_id): asin for asin, item_id in asin_to_item_id.items()}

    if _item_metadata is None:
        metadata: dict[str, dict[str, str]] = {}
        if ITEM_METADATA_CSV.exists():
            csv.field_size_limit(sys.maxsize)
            with ITEM_METADATA_CSV.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    asin = row.get("asin", "")
                    if not asin or asin in metadata:
                        continue
                    title = unescape(row.get("title", "")).strip()
                    category = (row.get("main_cat") or row.get("category") or "").strip()
                    metadata[asin] = {
                        "title": title or f"Item {asin}",
                        "category": category or "Software",
                    }
        _item_metadata = metadata

    return _item_id_to_asin, _item_metadata


def _describe_item(item_id: int) -> dict[str, Any]:
    item_id_to_asin, item_metadata = _load_item_maps()
    asin = item_id_to_asin.get(int(item_id), f"ITEM_{item_id}")
    metadata = item_metadata.get(asin, {})
    category = metadata.get("category") or "Software"
    return {
        "item_id": int(item_id),
        "asin": asin,
        "title": metadata.get("title") or f"Item {asin}",
        "category": category,
        "tag": category.upper()[:14],
    }


def _list_local_user_ids() -> list[str]:
    if not LOCAL_STORAGE_DIR.exists():
        return []

    user_ids: list[str] = []
    for file_path in LOCAL_STORAGE_DIR.glob("user_*.json"):
        user_id = file_path.stem.removeprefix("user_")
        if user_id:
            user_ids.append(user_id)

    return sorted(user_ids)


def _resolve_user_id(user_id: Optional[str]) -> str:
    if isinstance(user_id, str):
        normalized = user_id.strip()
        if normalized and normalized.lower() != "random":
            return normalized

    user_ids = _list_local_user_ids()
    if not user_ids:
        raise HTTPException(status_code=503, detail=f"No local user histories found in {LOCAL_STORAGE_DIR}")

    return random.choice(user_ids)


def _read_local_history(user_id: str) -> list[int]:
    file_path = LOCAL_STORAGE_DIR / f"user_{user_id}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"No local history found for user_id={user_id}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return [int(item_id) for item_id in data.get("history", [])]


def _ensure_graph_data() -> None:
    global _graph_load_attempted

    if _graph_load_attempted:
        return

    try:
        load_graph_data()
    except Exception as exc:
        print(f"[WARN] Failed to load item graph: {exc}")
    finally:
        _graph_load_attempted = True


def _candidate_result(history_ids: list[int], target_k: int) -> dict[str, Any]:
    embeddings = _require_embeddings()
    target_k = max(1, min(int(target_k), 200))
    max_index = embeddings.shape[0] - 1
    valid_history = [i for i in history_ids if 0 <= i <= max_index]
    history_set = set(history_ids)

    graph_neighbors: list[int] = []
    vector_neighbors: list[int] = []
    candidates: list[int] = []

    if valid_history:
        _ensure_graph_data()
        graph_neighbors = get_item_neighbors(valid_history, max_neighbors=20)
        candidates.extend(graph_neighbors)

        history_embs = embeddings[valid_history]
        mean_emb = np.mean(history_embs, axis=0)
        scores = np.dot(embeddings, mean_emb)
        top_indices = np.argsort(scores)[::-1][:40]
        vector_neighbors = [int(i) for i in top_indices]
        candidates.extend(vector_neighbors)

    unique_candidates: list[int] = []
    seen = set(history_set)
    for candidate in candidates:
        if candidate not in seen and 0 <= candidate <= max_index:
            unique_candidates.append(int(candidate))
            seen.add(int(candidate))
        if len(unique_candidates) >= target_k:
            break

    fallback_count = 0
    if len(unique_candidates) < target_k:
        needed = target_k - len(unique_candidates)
        fallbacks = get_popular_fallback(seen, count=needed)
        fallback_count = len(fallbacks)
        unique_candidates.extend(fallbacks)

    final_candidates = unique_candidates[:target_k]
    final_embeddings = embeddings[final_candidates] if final_candidates else np.empty((0, embeddings.shape[1]))

    return {
        "candidate_ids": final_candidates,
        "embeddings": final_embeddings,
        "dim": int(embeddings.shape[1]) if embeddings.ndim > 1 else 0,
        "source_summary": {
            "graph_neighbors": len(graph_neighbors),
            "vector_neighbors": len(vector_neighbors),
            "fallback_items": fallback_count,
        },
    }


def _load_tflite_interpreter():
    global _tflite_interpreter

    if _tflite_interpreter is not None:
        return _tflite_interpreter

    if not TFLITE_MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"TFLite model not found: {TFLITE_MODEL_PATH}")

    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail="TFLite runtime not installed. Install tensorflow or tflite-runtime.",
            ) from exc

    try:
        _tflite_interpreter = tflite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to load TFLite model: {exc}") from exc

    return _tflite_interpreter


def _load_torch_branch_weights() -> dict[str, np.ndarray]:
    global _torch_branch_weights

    if _torch_branch_weights is not None:
        return _torch_branch_weights

    if not GRAPH_SAGE_CHECKPOINT_PATH.exists():
        raise RuntimeError(f"GraphSAGE checkpoint not found: {GRAPH_SAGE_CHECKPOINT_PATH}")

    import torch

    pack = torch.load(GRAPH_SAGE_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    enc_state = pack["encoder"] if isinstance(pack, dict) and "encoder" in pack else pack
    key_map = {
        "lin_l_weight": "convs2.convs.<item___rev_reviews___user>.lin_l.weight",
        "lin_l_bias": "convs2.convs.<item___rev_reviews___user>.lin_l.bias",
        "lin_r_weight": "convs2.convs.<item___rev_reviews___user>.lin_r.weight",
    }

    weights: dict[str, np.ndarray] = {}
    for name, key in key_map.items():
        if key not in enc_state:
            raise RuntimeError(f"Missing checkpoint key: {key}")
        weights[name] = enc_state[key].detach().cpu().numpy().astype(np.float32)

    _torch_branch_weights = weights
    return weights


def _score_with_torch_branch(neighbor_x: np.ndarray, candidate_emb: np.ndarray) -> np.ndarray:
    weights = _load_torch_branch_weights()
    user_x = np.zeros((1, candidate_emb.shape[1]), dtype=np.float32)
    aggr = neighbor_x.mean(axis=0, keepdims=True).astype(np.float32)
    h2 = (
        user_x @ weights["lin_l_weight"].T
        + weights["lin_l_bias"].reshape(1, -1)
        + aggr @ weights["lin_r_weight"].T
    )
    return (h2 * candidate_emb).sum(axis=-1).astype(np.float32).reshape(-1)


def _score_with_mean_dot(neighbor_x: np.ndarray, candidate_emb: np.ndarray) -> np.ndarray:
    user_vec = neighbor_x.mean(axis=0, keepdims=True).astype(np.float32)
    return (user_vec * candidate_emb).sum(axis=-1).astype(np.float32).reshape(-1)


def _set_tflite_tensor(interpreter, input_detail: dict[str, Any], tensors: dict[str, np.ndarray]) -> None:
    name = input_detail["name"]
    key = next((candidate for candidate in tensors if candidate in name), None)
    if key is None:
        return
    interpreter.set_tensor(input_detail["index"], tensors[key])


def _score_candidates(
    history_ids: list[int],
    candidate_ids: list[int],
    candidate_embeddings: np.ndarray,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    embeddings = _require_embeddings()
    max_index = embeddings.shape[0] - 1
    valid_history = [i for i in history_ids if 0 <= i <= max_index]

    if valid_history:
        neighbor_x = np.asarray(embeddings[valid_history], dtype=np.float32)
    else:
        neighbor_x = np.zeros((1, embeddings.shape[1]), dtype=np.float32)

    candidate_emb = np.asarray(candidate_embeddings, dtype=np.float32)
    if candidate_emb.size == 0 or not candidate_ids:
        return [], {
            "model": TFLITE_MODEL_PATH.name,
            "neighbor_count": int(neighbor_x.shape[0]),
            "candidate_count": 0,
            "user_vector_preview": [],
            "candidate_vector_preview": [],
            "dot_score_preview": 0.0,
        }

    model_name = TFLITE_MODEL_PATH.name
    runtime = "tflite"
    runtime_detail = "TensorFlow Lite interpreter"

    try:
        interpreter = _load_tflite_interpreter()
        input_details = interpreter.get_input_details()

        for detail in input_details:
            name = detail["name"]
            if "neighbor" in name:
                interpreter.resize_tensor_input(detail["index"], list(neighbor_x.shape))
            elif "candidate" in name:
                interpreter.resize_tensor_input(detail["index"], list(candidate_emb.shape))

        interpreter.allocate_tensors()

        tensors = {
            "user_x": np.zeros((1, embeddings.shape[1]), dtype=np.float32),
            "neighbor_x": neighbor_x,
            "candidate_emb": candidate_emb,
        }
        for detail in interpreter.get_input_details():
            _set_tflite_tensor(interpreter, detail, tensors)

        interpreter.invoke()
        output_details = interpreter.get_output_details()
        logits = np.asarray(interpreter.get_tensor(output_details[0]["index"]), dtype=np.float32).reshape(-1)
    except HTTPException as exc:
        try:
            logits = _score_with_torch_branch(neighbor_x, candidate_emb)
            model_name = GRAPH_SAGE_CHECKPOINT_PATH.name
            runtime = "torch_checkpoint_fallback"
            runtime_detail = f"TFLite unavailable ({exc.detail}); used GraphSAGE checkpoint weights"
        except Exception as torch_exc:
            logits = _score_with_mean_dot(neighbor_x, candidate_emb)
            model_name = "mean_neighbor_dot_product"
            runtime = "numpy_mean_dot_fallback"
            runtime_detail = (
                f"TFLite unavailable ({exc.detail}); PyTorch fallback unavailable ({torch_exc}); "
                "used mean-neighbor dot product"
            )

    top_k = max(1, min(int(top_k), len(candidate_ids)))
    top_indices = np.argsort(logits)[::-1][:top_k]
    top_scores = [float(logits[index]) for index in top_indices]
    min_score = min(top_scores) if top_scores else 0.0
    max_score = max(top_scores) if top_scores else 0.0
    score_range = max(max_score - min_score, 1e-6)

    recommendations: list[dict[str, Any]] = []
    for rank, index in enumerate(top_indices, start=1):
        item_id = int(candidate_ids[int(index)])
        score = float(logits[int(index)])
        item = _describe_item(item_id)
        recommendations.append(
            {
                "rank": rank,
                **item,
                "score": score,
                "score_percent": round(60 + ((score - min_score) / score_range) * 40, 1),
            }
        )

    mean_neighbor = neighbor_x.mean(axis=0)
    best_candidate = candidate_emb[int(top_indices[0])] if len(top_indices) else candidate_emb[0]
    inference = {
        "model": model_name,
        "runtime": runtime,
        "runtime_detail": runtime_detail,
        "neighbor_count": int(neighbor_x.shape[0]),
        "candidate_count": int(candidate_emb.shape[0]),
        "user_vector_preview": [round(float(v), 4) for v in mean_neighbor[:8]],
        "candidate_vector_preview": [round(float(v), 4) for v in best_candidate[:8]],
        "dot_score_preview": round(float(top_scores[0]), 4) if top_scores else 0.0,
    }
    return recommendations, inference


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
    embeddings = _require_embeddings()
    ids = _parse_item_ids(item_ids)
    max_index = embeddings.shape[0] - 1

    valid: list[int] = []
    missing: list[int] = []
    for idx in ids:
        if 0 <= idx <= max_index:
            valid.append(idx)
        else:
            missing.append(idx)

    if missing and strict:
        raise HTTPException(status_code=404, detail={"missing_ids": missing})

    vectors = embeddings[valid].tolist() if valid else []

    return {
        "item_ids": valid,
        "embeddings": vectors,
        "missing_ids": missing,
        "dim": int(embeddings.shape[1]) if embeddings.ndim > 1 else 0,
    }


@app.get("/api/v1/candidates/")
def retrieve_candidates(
    history_item_ids: list[str] = Query(..., description="User's local interaction history"),
    target_k: int = Query(50, ge=1, le=200, description="Total number of candidates to return"),
) -> dict:
    try:
        history_ids = _parse_item_ids(history_item_ids)
    except HTTPException:
        history_ids = []

    result = _candidate_result(history_ids, target_k)
    embeddings = result["embeddings"]
    return {
        "candidate_ids": result["candidate_ids"],
        "embeddings": embeddings.tolist() if len(result["candidate_ids"]) else [],
        "dim": result["dim"],
        "source_summary": result["source_summary"],
    }


@app.get("/api/v1/recommendations/")
def get_recommendations(
    user_id: Optional[str] = Query(None, description="Simulated device user ID; omit or use random for local random"),
    top_k: int = Query(5, ge=1, le=20, description="Recommendations to return"),
    target_k: int = Query(50, ge=1, le=200, description="Candidate pool size"),
) -> dict:
    embeddings = _require_embeddings()
    resolved_user_id = _resolve_user_id(user_id)
    history_ids = _read_local_history(resolved_user_id)
    candidate_result = _candidate_result(history_ids, target_k)
    recommendations, inference = _score_candidates(
        history_ids=history_ids,
        candidate_ids=candidate_result["candidate_ids"],
        candidate_embeddings=candidate_result["embeddings"],
        top_k=top_k,
    )

    return {
        "user_id": resolved_user_id,
        "history": [_describe_item(item_id) for item_id in history_ids],
        "cloud": {
            "candidate_ids": candidate_result["candidate_ids"],
            "candidate_count": len(candidate_result["candidate_ids"]),
            "embedding_dim": candidate_result["dim"],
            "embedding_source": _source,
            "catalog_items": int(embeddings.shape[0]),
            "source_summary": candidate_result["source_summary"],
        },
        "inference": inference,
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
