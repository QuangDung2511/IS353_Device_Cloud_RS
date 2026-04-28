"""
extract_embeddings.py
=====================
Nhiệm vụ 4.1 (Phase 4): Trích xuất Item embeddings từ GraphSAGE checkpoint đã huấn luyện
và lưu dưới dạng NumPy memory-mapped array để Cloud Server truy vấn O(1).

Output:
    cloud_server/artifacts/item_embeddings.npy   — shape [num_items, hidden_channels]
    cloud_server/artifacts/item_id_map.json      — mapping item_idx → asin
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, HeteroConv, to_hetero
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


# ──────────────────────────────────────────────────────────────────────────────
# Model definition (phải khớp với notebooks/06_graphsage_train_eval.ipynb)
# ──────────────────────────────────────────────────────────────────────────────

class HeteroSAGEEncoder(nn.Module):
    """
    2-layer Heterogeneous GraphSAGE Encoder.
    Kiến trúc này phải khớp chính xác với model đã lưu trong checkpoint.
    """

    def __init__(self, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = HeteroConv({
            ('user', 'reviews', 'item'):      SAGEConv((-1, -1), hidden_channels),
            ('item', 'rev_reviews', 'user'):  SAGEConv((-1, -1), hidden_channels),
            ('item', 'also_bought', 'item'):  SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('user', 'reviews', 'item'):      SAGEConv((-1, -1), out_channels),
            ('item', 'rev_reviews', 'user'):  SAGEConv((-1, -1), out_channels),
            ('item', 'also_bought', 'item'):  SAGEConv((-1, -1), out_channels),
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


class DotProductDecoder(nn.Module):
    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        return (user_emb * item_emb).sum(dim=-1)


class GraphSAGELinkPredictor(nn.Module):
    def __init__(self, hidden_channels: int = 64, out_channels: int = 64):
        super().__init__()
        self.encoder = HeteroSAGEEncoder(hidden_channels, out_channels)
        self.decoder = DotProductDecoder()

    def encode(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def decode(self, user_emb, item_emb):
        return self.decoder(user_emb, item_emb)


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction logic
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: str, device: torch.device, hidden_channels: int, out_channels: int, data: HeteroData) -> nn.Module:
    """
    Load GraphSAGE model từ checkpoint .pt file.

    Hỗ trợ 3 dạng checkpoint từ notebook 06:
        1. {"encoder": sd, "decoder": sd, "hparams": ...}  ← format notebook 06
        2. {"model_state_dict": sd, "epoch": ...}          ← format training loop thông thường
        3. raw state_dict  {layer_name: tensor, ...}
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = GraphSAGELinkPredictor(hidden_channels=hidden_channels, out_channels=out_channels)
    model = model.to(device)
    
    # Thực hiện DUMMY FORWARD PASS để khởi tạo các Lazy Modules (SAGEConv(-1, -1))
    model.eval()
    with torch.no_grad():
        x_dict = {node_type: data[node_type].x.to(device) for node_type in data.node_types}
        edge_index_dict = {
            edge_type: data[edge_type].edge_index.to(device)
            for edge_type in data.edge_types
        }
        model.encode(x_dict, edge_index_dict)

    if isinstance(state, dict) and "encoder" in state:
        # Format notebook 06: lưu encoder và decoder riêng
        hparams = state.get("hparams", {})
        print(f"[INFO] Checkpoint format: notebook-06  hparams={hparams}")
        # Ghép encoder + decoder state_dict thành 1 và sửa lỗi lệch key format
        combined: dict[str, torch.Tensor] = {}
        for k, v in state["encoder"].items():
            # Chỉ cần chuyển từ "convs1" sang "conv1" và "convs2" sang "conv2"
            k = k.replace("convs1.", "conv1.").replace("convs2.", "conv2.")
            combined[f"encoder.{k}"] = v
        for k, v in state["decoder"].items():
            combined[f"decoder.{k}"] = v
        model.load_state_dict(combined, strict=True)

    elif isinstance(state, dict) and "model_state_dict" in state:
        # Format: {"model_state_dict": ..., "epoch": ...}
        print(f"[INFO] Checkpoint format: model_state_dict  epoch={state.get('epoch', '?')}")
        model.load_state_dict(state["model_state_dict"])

    elif isinstance(state, dict):
        # Raw state_dict
        print("[INFO] Checkpoint format: raw state_dict")
        model.load_state_dict(state)

    elif isinstance(state, nn.Module):
        # Full model object (torch.save(model, path))
        print("[INFO] Checkpoint format: full model object")
        model = state

    else:
        raise ValueError(f"Unrecognized checkpoint format: {type(state)}")

    result: nn.Module = model  # explicit annotation để Pylance không complain
    return result


def load_graph_data(data_path: str, device: torch.device) -> HeteroData:
    """Load HeteroData graph từ file .pt."""
    data = torch.load(data_path, map_location=device, weights_only=False)
    # Thêm cạnh ngược (rev_reviews) bằng ToUndirected
    data = T.ToUndirected()(data)
    print(f"[INFO] Loaded HeteroData: {data}")
    return data


def extract_item_embeddings(
    model: nn.Module,
    data: HeteroData,
    device: torch.device,
) -> np.ndarray:
    """
    Forward pass trên toàn bộ graph để lấy item embeddings.
    Trả về numpy array shape [num_items, out_channels].
    """
    model.eval()
    with torch.no_grad():
        x_dict = {node_type: data[node_type].x.to(device) for node_type in data.node_types}
        edge_index_dict = {
            edge_type: data[edge_type].edge_index.to(device)
            for edge_type in data.edge_types
        }
        embeddings_dict = model.encode(x_dict, edge_index_dict)

    item_embeddings = embeddings_dict['item'].cpu().numpy()
    print(f"[INFO] Extracted item embeddings: shape={item_embeddings.shape}, dtype={item_embeddings.dtype}")
    return item_embeddings


def save_artifacts(
    item_embeddings: np.ndarray,
    output_dir: str,
    item_id_map: Optional[dict] = None,
):
    """Lưu embeddings và optional mapping file."""
    os.makedirs(output_dir, exist_ok=True)

    emb_path = os.path.join(output_dir, "item_embeddings.npy")
    np.save(emb_path, item_embeddings)
    print(f"[✓] Saved item embeddings → {emb_path}  ({item_embeddings.nbytes / 1024:.1f} KB)")

    if item_id_map is not None:
        map_path = os.path.join(output_dir, "item_id_map.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(item_id_map, f, ensure_ascii=False, indent=2)
        print(f"[✓] Saved item ID map    → {map_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Trích xuất Item embeddings từ GraphSAGE checkpoint (Phase 4.1)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/processed/graphsage_link_pred.pt",
        help="Đường dẫn tới file checkpoint .pt",
    )
    parser.add_argument(
        "--graph-data",
        type=str,
        default="data/processed/hetero_data.pt",
        help="Đường dẫn tới HeteroData graph .pt (dùng để forward pass)",
    )
    parser.add_argument(
        "--item-id-map",
        type=str,
        default=None,
        help="(Tùy chọn) JSON file mapping item_idx → asin. Nếu bỏ qua sẽ không lưu mapping.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cloud_server/artifacts",
        help="Thư mục lưu output artifacts",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=256,
        help="hidden_channels của model (phải khớp checkpoint)",
    )
    parser.add_argument(
        "--out-channels",
        type=int,
        default=256,
        help="out_channels của model (phải khớp checkpoint)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device để chạy inference",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # Kiểm tra files tồn tại
    for fpath, label in [(args.checkpoint, "checkpoint"), (args.graph_data, "graph data")]:
        if not os.path.exists(fpath):
            print(f"[ERROR] {label} file not found: {fpath}", file=sys.stderr)
            sys.exit(1)

    # Load
    print(f"\n{'='*60}")
    print("Phase 4.1 — Trích xuất Cloud Artifacts")
    print(f"{'='*60}")

    data  = load_graph_data(args.graph_data, device)
    model = load_checkpoint(args.checkpoint, device, args.hidden_channels, args.out_channels, data)

    # Extract
    item_embeddings = extract_item_embeddings(model, data, device)

    # Load optional ID map
    item_id_map = None
    if args.item_id_map and os.path.exists(args.item_id_map):
        with open(args.item_id_map, "r", encoding="utf-8") as f:
            item_id_map = json.load(f)
        print(f"[INFO] Loaded item ID map: {len(item_id_map)} entries")

    # Save
    save_artifacts(item_embeddings, args.output_dir, item_id_map)

    print(f"\n[✓] Done! Item embeddings ready for FastAPI Cloud Server.")
    print(f"    Shape: {item_embeddings.shape}  |  Size: {item_embeddings.nbytes/1024:.1f} KB")


if __name__ == "__main__":
    main()