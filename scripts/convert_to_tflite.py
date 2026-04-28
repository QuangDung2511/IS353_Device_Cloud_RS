"""
convert_to_tflite.py
====================
Phase 5.1 — Chuyển đổi nhánh User SAGEConv → ONNX → TFLite

Kiến trúc từ notebook 06:
  - IN_CHANNELS = 384  (sentence-transformers/all-MiniLM-L6-v2)
  - HIDDEN = 256, OUT_CH = 256
  - Checkpoint keys: {"encoder": state_dict, "decoder": state_dict, "metadata": ..., "hparams": ...}
  - Edge types: (user,reviews,item), (item,rev_reviews,user), (item,also_bought,item)

Tại sao không export HeteroConv trực tiếp?
  HeteroConv + SAGEConv dùng scatter_add trên dynamic edge_index → torch.onnx không trace
  được vì shape phụ thuộc runtime. Giải pháp: "freeze" aggregation bằng cách trích xuất
  trọng số lin_l / lin_r từ SAGEConv rồi build một nn.Module thuần Linear để export.

Pipeline:
  1. Load checkpoint → trích xuất weights SAGEConv (user branch)
  2. Build FrozenUserSAGEDecoder (thuần Linear + DotProduct, ONNX-friendly)
  3. Export → ONNX (opset 17, dynamic axes)
  4. ONNX → TFLite FlatBuffer (onnx2tf — thay thế onnx-tf đã bị abandon)
  6. Smoke test với tflite-runtime

Usage:
  # Chỉ export ONNX (chưa cần tensorflow):
  python scripts/convert_to_tflite.py --skip-tflite --cross-check

  # Full pipeline:
  python scripts/convert_to_tflite.py --smoke-test --cross-check

  # Chưa có checkpoint (test pipeline):
  python scripts/convert_to_tflite.py --no-transplant --skip-tflite

Requirements:
  pip install torch torch-geometric onnx onnxruntime onnx2tf tensorflow tf-keras onnx-graphsurgeon ai-edge-litert sng4onnx

Output:
  device_client/models/user_sage_decoder.onnx
  device_client/models/user_sage_decoder.tflite
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Hằng số — khớp notebook 06 và Phase 4
# ─────────────────────────────────────────────────────────────────────────────
IN_CHANNELS     = 256  # Sửa lại thành 256 vì Cloud trả về final embeddings (256-d)
HIDDEN_CHANNELS = 256  # Không còn dùng tới
OUT_CHANNELS    = 256

# ─────────────────────────────────────────────────────────────────────────────
# ONNX-exportable On-Device Model
# ─────────────────────────────────────────────────────────────────────────────

class FrozenUserSAGEDecoder(nn.Module):
    """
    Module thuần Linear, ONNX-friendly — không có scatter / dynamic index.

    Sửa đổi cấu trúc: Vì thiết bị di động tải về các "dense embeddings" (256-d) 
    từ Cloud cho các item đã tương tác, thiết bị sẽ chỉ chạy **Layer 2** của GraphSAGE 
    để tính toán embedding cho User.
    
        Layer 2:  user_h2 = lin_l2(user_h1) + lin_r2(mean(item_h1))
        Decode:   logits  = (user_h2 · candidate_emb).sum(dim=-1)

    Inputs (float32):
        user_x        [1, 256]    — feature vector (zeros cho hidden/new user)
        neighbor_x    [K, 256]    — item pre-trained embeddings từ cloud (đóng vai trò item_h1)
        candidate_emb [N, 256]    — pre-computed item embeddings cho candidate (từ cloud)
    Output:
        logits        [N]         — ranking scores, top-K → danh sách gợi ý
    """

    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int):
        super().__init__()
        # Layer 2 — (item, rev_reviews, user) branch
        # Dùng để cập nhật user từ item neighbors
        self.lin_l = nn.Linear(in_ch, out_ch, bias=True)   # self (has bias)
        self.lin_r = nn.Linear(in_ch, out_ch, bias=False)  # neighbor agg (no bias)

    def forward(
        self,
        user_x:        torch.Tensor,   # [1, in_ch]
        neighbor_x:    torch.Tensor,   # [K, in_ch]
        candidate_emb: torch.Tensor,   # [N, out_ch]
    ) -> torch.Tensor:
        # Neighbor aggregation (tính trung bình các item embeddings)
        aggr = neighbor_x.mean(dim=0, keepdim=True)         # [1, in_ch]
        
        # Layer 2: cập nhật user
        h2 = self.lin_l(user_x) + self.lin_r(aggr)          # [1, out_ch]
        # Không có ReLU sau Layer 2 (theo cấu trúc HeteroSAGEEncoder trong bài)

        # DotProduct decoder
        logits = (h2 * candidate_emb).sum(dim=-1)           # [N]
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Weight transplant
# ─────────────────────────────────────────────────────────────────────────────

def transplant_weights(model: FrozenUserSAGEDecoder, ckpt_path: str) -> dict:
    """
    Trích xuất weights SAGEConv từ checkpoint notebook 06 sang FrozenUserSAGEDecoder.
    Lưu ý: Để cập nhật User, ta phải dùng nhánh <item___rev_reviews___user>.
    """
    pack      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc_state = pack["encoder"]
    hparams   = pack.get("hparams", {})

    print("[INFO] Checkpoint hparams:", hparams)
    print("[INFO] Relevant encoder keys:")
    for k, v in enc_state.items():
        if "rev_reviews" in k:
            print(f"       {k}  {tuple(v.shape)}")

    # src key (checkpoint) → dst key (FrozenUserSAGEDecoder)
    key_map = {
        # Checkpoint thực tế dùng format: <item___rev_reviews___user>
        "convs2.convs.<item___rev_reviews___user>.lin_l.weight": "lin_l.weight",
        "convs2.convs.<item___rev_reviews___user>.lin_l.bias":   "lin_l.bias",
        "convs2.convs.<item___rev_reviews___user>.lin_r.weight": "lin_r.weight",
    }

    new_state   = model.state_dict()
    transferred = 0
    failed      = []

    for src_key, dst_key in key_map.items():
        if src_key not in enc_state:
            failed.append(f"  MISSING src key: {src_key}")
            continue
        if dst_key not in new_state:
            # bias không có nếu Linear(bias=False) — skip silently
            continue
        src_t = enc_state[src_key]
        dst_t = new_state[dst_key]
        if src_t.shape != dst_t.shape:
            failed.append(
                f"  SHAPE MISMATCH: {src_key}{tuple(src_t.shape)} → "
                f"{dst_key}{tuple(dst_t.shape)}"
            )
            continue
        new_state[dst_key] = src_t.clone()
        transferred += 1
        print(f"  [✓] {src_key} → {dst_key}  {tuple(src_t.shape)}")

    if failed:
        print("\n[WARN] Một số weights không transplant được:")
        for msg in failed:
            print(msg)

    model.load_state_dict(new_state)
    print(f"\n[INFO] Transplanted {transferred}/{len(key_map)} weight tensors.")
    return hparams


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Export ONNX
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx(
    model:     nn.Module,
    onnx_path: str,
    in_ch:     int = IN_CHANNELS,
    out_ch:    int = OUT_CHANNELS,
    K:         int = 10,
    N:         int = 100,
):
    model.eval()
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

    user_x        = torch.zeros(1, in_ch)
    neighbor_x    = torch.randn(K, in_ch)
    candidate_emb = torch.randn(N, out_ch)

    # Verify trước khi export
    with torch.no_grad():
        out = model(user_x, neighbor_x, candidate_emb)
    assert out.shape == (N,), f"Expected ({N},), got {out.shape}"
    assert torch.isfinite(out).all(), "Non-finite values trước export!"
    print(f"[INFO] Pre-export forward OK: logits shape={out.shape}, "
          f"min={out.min():.4f}, max={out.max():.4f}")

    torch.onnx.export(
        model,
        args=(user_x, neighbor_x, candidate_emb),
        f=onnx_path,
        input_names=["user_x", "neighbor_x", "candidate_emb"],
        output_names=["logits"],
        dynamic_axes={
            "neighbor_x":    {0: "num_neighbors"},
            "candidate_emb": {0: "num_candidates"},
            "logits":        {0: "num_candidates"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"[✓] ONNX exported → {onnx_path}  ({size_kb:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — ONNX → TFLite
# ─────────────────────────────────────────────────────────────────────────────

def convert_to_tflite(onnx_path: str, tflite_path: str):
    """
    ONNX → TFLite dùng onnx2tf (thay thế onnx-tf đã bị abandon từ 2022).

    Cài đặt:
        pip install onnx2tf tensorflow

    onnx2tf convert trực tiếp ONNX → TFLite mà không cần bước TF SavedModel trung gian,
    nhanh hơn và tương thích với onnx >= 1.13.
    """
    try:
        import onnx2tf
    except ImportError:
        print("\n[ERROR] onnx2tf chưa được cài.")
        print("  pip install onnx2tf tensorflow")
        sys.exit(1)

    os.makedirs(os.path.dirname(tflite_path) or ".", exist_ok=True)
    output_dir = os.path.dirname(tflite_path) or "."

    print("\n[...] ONNX → TFLite (onnx2tf) ...")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=output_dir,
        non_verbose=True,
        # output_integer_quantized_tflite=False,  # giữ float32
    )

    # onnx2tf tự đặt tên file theo tên model, cần rename về đúng tflite_path
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    generated_path = os.path.join(output_dir, f"{model_name}_float32.tflite")

    # Fallback nếu naming khác
    if not os.path.exists(generated_path):
        candidates = [f for f in os.listdir(output_dir) if f.endswith(".tflite")]
        if not candidates:
            print("[ERROR] onnx2tf không tạo ra file .tflite nào!")
            sys.exit(1)
        generated_path = os.path.join(output_dir, candidates[0])

    if os.path.abspath(generated_path) != os.path.abspath(tflite_path):
        import shutil
        shutil.move(generated_path, tflite_path)

    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"[✓] TFLite saved → {tflite_path}  ({size_kb:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Smoke test TFLite
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test_tflite(
    tflite_path: str,
    in_ch:  int = IN_CHANNELS,
    out_ch: int = OUT_CHANNELS,
    K: int = 8,
    N: int = 50,
):
    # tflite-runtime standalone và ai-edge-litert đều KHÔNG có wheel cho Python 3.9 Windows.
    # Cách duy nhất hoạt động: tf.lite.Interpreter built-in trong tensorflow.
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("[INFO] Using tensorflow.lite.Interpreter")
    except ImportError:
        print("[WARN] tensorflow chưa được cài. Chạy: pip install tensorflow")
        return

    print(f"\n[Smoke Test] {tflite_path}")
    interp = Interpreter(model_path=tflite_path)

    inp_details = interp.get_input_details()
    out_details = interp.get_output_details()

    # Resize dynamic axes
    for d in inp_details:
        name = d["name"]
        if "neighbor" in name:
            interp.resize_tensor_input(d["index"], [K, in_ch])
        elif "candidate" in name:
            interp.resize_tensor_input(d["index"], [N, out_ch])

    interp.allocate_tensors()

    # Feed dummy data
    dummy = {
        "user_x":        np.zeros((1, in_ch),   dtype=np.float32),
        "neighbor_x":    np.random.randn(K, in_ch).astype(np.float32),
        "candidate_emb": np.random.randn(N, out_ch).astype(np.float32),
    }
    for d in inp_details:
        name = d["name"]
        key  = next((k for k in dummy if k in name or name in k), None)
        tensor = dummy[key] if key else list(dummy.values())[d["index"]]
        interp.set_tensor(d["index"], tensor)

    interp.invoke()
    logits = interp.get_tensor(out_details[0]["index"])

    assert logits.shape[0] == N,         f"Expected {N} logits, got {logits.shape}"
    assert np.isfinite(logits).all(),    "Non-finite values trong TFLite output!"

    top5 = np.argsort(logits)[::-1][:5]
    print(f"  logits shape : {logits.shape}")
    print(f"  logits range : [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"  Top-5 indices: {top5.tolist()}")
    print("  [✓] Smoke test PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-check: PyTorch vs ONNX Runtime
# ─────────────────────────────────────────────────────────────────────────────

def cross_check_onnx(
    model:     nn.Module,
    onnx_path: str,
    in_ch:     int = IN_CHANNELS,
    out_ch:    int = OUT_CHANNELS,
    K: int = 10, N: int = 20,
    atol: float = 1e-5,
):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARN] onnxruntime không có, bỏ qua cross-check. "
              "pip install onnxruntime")
        return

    user_x        = torch.zeros(1, in_ch)
    neighbor_x    = torch.randn(K, in_ch)
    candidate_emb = torch.randn(N, out_ch)

    model.eval()
    with torch.no_grad():
        pt_out = model(user_x, neighbor_x, candidate_emb).numpy()

    sess    = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(
        ["logits"],
        {
            "user_x":        user_x.numpy(),
            "neighbor_x":    neighbor_x.numpy(),
            "candidate_emb": candidate_emb.numpy(),
        }
    )[0]

    max_diff = float(np.abs(pt_out - ort_out).max())
    status   = "PASSED" if max_diff < atol else "WARN — diff lớn, kiểm tra lại!"
    print(f"[Cross-check] PyTorch vs ONNX  max_diff={max_diff:.2e}  → {status}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 5.1 — PyTorch User SAGEConv Branch → ONNX → TFLite"
    )
    p.add_argument(
        "--checkpoint",
        default="data/processed/graphsage_link_pred.pt",
        help="Checkpoint từ notebook 06 (default: data/processed/graphsage_link_pred.pt)",
    )
    p.add_argument(
        "--onnx-path",
        default="device_client/models/user_sage_decoder.onnx",
    )
    p.add_argument(
        "--tflite-path",
        default="device_client/models/user_sage_decoder.tflite",
    )
    p.add_argument(
        "--skip-tflite",
        action="store_true",
        help="Chỉ export ONNX, bỏ qua TFLite (dùng khi chưa cài tensorflow/onnx-tf)",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Chạy smoke test TFLite sau khi convert",
    )
    p.add_argument(
        "--cross-check",
        action="store_true",
        help="So sánh output ONNX vs PyTorch (cần onnxruntime)",
    )
    p.add_argument(
        "--no-transplant",
        action="store_true",
        help="Không load weights (dùng random — để test pipeline khi chưa có checkpoint)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*65}")
    print("Phase 5.1 — PyTorch User Branch → ONNX → TFLite")
    print(f"{'='*65}")
    print(f"  in_channels     : {IN_CHANNELS}")
    print(f"  hidden_channels : {HIDDEN_CHANNELS}")
    print(f"  out_channels    : {OUT_CHANNELS}")
    print(f"  checkpoint      : {args.checkpoint}")
    print(f"  onnx_path       : {args.onnx_path}")
    print(f"  tflite_path     : {args.tflite_path}")

    # Build model
    model = FrozenUserSAGEDecoder(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS)

    # Transplant weights
    if not args.no_transplant:
        if os.path.exists(args.checkpoint):
            print(f"\n[...] Transplanting weights ...")
            transplant_weights(model, args.checkpoint)
        else:
            print(f"\n[WARN] Checkpoint không tìm thấy: {args.checkpoint}")
            print("[WARN] Dùng random weights — inference sẽ không có ý nghĩa.")
    else:
        print("\n[INFO] --no-transplant: dùng random weights.")

    model.eval()

    # Step 1: Export ONNX
    print(f"\n{'─'*65}")
    print("Step 1 — Export ONNX")
    export_onnx(model, args.onnx_path)

    if args.cross_check:
        print("\n[...] Cross-check PyTorch vs ONNX ...")
        cross_check_onnx(model, args.onnx_path)

    # Step 2: ONNX → TFLite
    if not args.skip_tflite:
        print(f"\n{'─'*65}")
        print("Step 2 — ONNX → TFLite")
        convert_to_tflite(args.onnx_path, args.tflite_path)

        if args.smoke_test:
            smoke_test_tflite(args.tflite_path)
    else:
        print("\n[INFO] Bỏ qua TFLite (--skip-tflite).")
        print("       Chạy lại không có flag đó khi đã cài: pip install onnx-tf tensorflow")

    print(f"\n{'='*65}")
    print("[✓] Done!")
    print(f"    ONNX   → {args.onnx_path}")
    if not args.skip_tflite:
        print(f"    TFLite → {args.tflite_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()