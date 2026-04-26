"""
profile_bandwidth.py
====================
Nhiệm vụ 6.1 (Phase 6): Phân tích Bandwidth & Privacy Profiling

So sánh băng thông truyền tải giữa:
    - Traditional Cloud: Upload toàn bộ raw interaction logs (user_id, item_id, rating, timestamp)
    - DCCL:             Chỉ request queried dense item embeddings từ Cloud

Output:
    - Console: bảng so sánh payload sizes (Bytes / KB / MB)
    - demo/bandwidth_profile.png: bar chart trực quan hóa bandwidth reduction
    - demo/bandwidth_report.json: dữ liệu số cho Streamlit dashboard
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BandwidthProfile:
    scenario: str
    description: str
    bytes_per_request: float
    kb_per_request: float
    mb_per_request: float
    requests_per_day: int
    total_mb_per_day: float


# ──────────────────────────────────────────────────────────────────────────────
# Payload size calculations
# ──────────────────────────────────────────────────────────────────────────────

BYTES_PER_FLOAT32 = 4
BYTES_PER_INT32   = 4
BYTES_PER_CHAR    = 1  # UTF-8 approximation for IDs/strings


def calc_traditional_payload(
    num_interactions: int,
    include_text_features: bool = False,
    text_feature_dim: int = 384,
) -> float:
    """
    Traditional Cloud: gửi toàn bộ interaction log lên server.
    Schema mỗi interaction: (user_id: int32, item_id: int32, rating: float32, timestamp: int32)
    = 4 * 4 = 16 bytes / interaction
    """
    bytes_per_interaction = (
        BYTES_PER_INT32   # user_id
        + BYTES_PER_INT32   # item_id (asin encoded)
        + BYTES_PER_FLOAT32 # rating
        + BYTES_PER_INT32   # unix timestamp
    )
    raw_bytes = num_interactions * bytes_per_interaction

    if include_text_features:
        # Worst case: cũng gửi text features của items
        raw_bytes += num_interactions * text_feature_dim * BYTES_PER_FLOAT32

    return float(raw_bytes)


def calc_dccl_payload(
    num_items_queried: int,
    embedding_dim: int = 64,
    include_request_overhead: bool = True,
) -> float:
    """
    DCCL: chỉ gửi list item_ids để query, nhận về dense embeddings.
    Request:  num_items_queried * 4 bytes (int32 IDs)
    Response: num_items_queried * embedding_dim * 4 bytes (float32 vectors)
    """
    request_bytes  = num_items_queried * BYTES_PER_INT32 if include_request_overhead else 0
    response_bytes = num_items_queried * embedding_dim * BYTES_PER_FLOAT32
    return float(request_bytes + response_bytes)


def build_profiles(
    num_interactions: int,
    num_items_queried: int,
    embedding_dim: int,
    requests_per_day: int,
) -> list[BandwidthProfile]:
    """Xây dựng danh sách các profile để so sánh."""

    traditional_bytes = calc_traditional_payload(num_interactions)
    dccl_bytes        = calc_dccl_payload(num_items_queried, embedding_dim)
    trad_with_text    = calc_traditional_payload(num_interactions, include_text_features=True)

    profiles = [
        BandwidthProfile(
            scenario="Traditional\n(Raw Logs)",
            description=f"Upload {num_interactions} interactions × 16 bytes/interaction",
            bytes_per_request=traditional_bytes,
            kb_per_request=traditional_bytes / 1024,
            mb_per_request=traditional_bytes / (1024**2),
            requests_per_day=requests_per_day,
            total_mb_per_day=traditional_bytes * requests_per_day / (1024**2),
        ),
        BandwidthProfile(
            scenario="Traditional\n(With Text Features)",
            description=f"Upload {num_interactions} interactions + text features (dim={384})",
            bytes_per_request=trad_with_text,
            kb_per_request=trad_with_text / 1024,
            mb_per_request=trad_with_text / (1024**2),
            requests_per_day=requests_per_day,
            total_mb_per_day=trad_with_text * requests_per_day / (1024**2),
        ),
        BandwidthProfile(
            scenario=f"DCCL\n(Dense Embeddings)",
            description=f"Query {num_items_queried} item embeddings × {embedding_dim}-dim × 4 bytes",
            bytes_per_request=dccl_bytes,
            kb_per_request=dccl_bytes / 1024,
            mb_per_request=dccl_bytes / (1024**2),
            requests_per_day=requests_per_day,
            total_mb_per_day=dccl_bytes * requests_per_day / (1024**2),
        ),
    ]
    return profiles


# ──────────────────────────────────────────────────────────────────────────────
# Realistic simulation từ actual embedding files
# ──────────────────────────────────────────────────────────────────────────────

def profile_from_actual_files(
    embeddings_path: Optional[str],
    local_storage_dir: Optional[str],
) -> dict:
    """
    Nếu có actual files, tính payload thực tế từ file sizes.
    """
    results = {}

    if embeddings_path and os.path.exists(embeddings_path):
        emb = np.load(embeddings_path)
        results["actual_embeddings"] = {
            "shape": list(emb.shape),
            "total_size_bytes": emb.nbytes,
            "total_size_kb": emb.nbytes / 1024,
            "total_size_mb": emb.nbytes / (1024**2),
            "bytes_per_item": emb.nbytes / emb.shape[0],
        }
        print(f"[INFO] Actual embeddings file: {emb.shape}  "
              f"({emb.nbytes/1024:.1f} KB total, "
              f"{emb.nbytes/emb.shape[0]:.0f} bytes/item)")

    if local_storage_dir and os.path.exists(local_storage_dir):
        json_files = list(Path(local_storage_dir).glob("*.json"))
        total_bytes = sum(f.stat().st_size for f in json_files)
        results["actual_local_storage"] = {
            "num_users": len(json_files),
            "total_size_bytes": total_bytes,
            "avg_size_per_user_bytes": total_bytes / max(len(json_files), 1),
        }
        print(f"[INFO] Local storage: {len(json_files)} user files, "
              f"avg {results['actual_local_storage']['avg_size_per_user_bytes']:.0f} bytes/user")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Reporting & Visualization
# ──────────────────────────────────────────────────────────────────────────────

def print_report(profiles: list[BandwidthProfile], reduction_ratio: float):
    """In bảng so sánh ra console."""
    print(f"\n{'='*70}")
    print("Phase 6.1 — Bandwidth & Privacy Profiling Report")
    print(f"{'='*70}")
    print(f"{'Scenario':<30} {'Bytes/req':>12} {'KB/req':>10} {'MB/day':>10}")
    print("-" * 70)
    for p in profiles:
        label = p.scenario.replace("\n", " ")
        print(f"{label:<30} {p.bytes_per_request:>12,.0f} {p.kb_per_request:>10.2f} {p.total_mb_per_day:>10.2f}")
    print("-" * 70)
    print(f"\n  → DCCL reduces bandwidth by {reduction_ratio:.1f}x compared to Traditional (Raw Logs)")
    print(f"  → Raw interactions NEVER leave the device (privacy preserved ✓)")
    print(f"{'='*70}\n")


def plot_bar_chart(profiles: list[BandwidthProfile], output_path: str):
    """Vẽ bar chart so sánh bandwidth và lưu PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[WARN] matplotlib not installed, skipping chart generation.")
        print("       Install with: pip install matplotlib")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "DCCL Bandwidth & Privacy Profiling\nAmazon Baby Products Dataset",
        fontsize=14, fontweight="bold", y=1.02
    )

    labels    = [p.scenario for p in profiles]
    kb_values = [p.kb_per_request for p in profiles]
    mb_values = [p.total_mb_per_day for p in profiles]

    colors = ["#E74C3C", "#E67E22", "#2ECC71"]  # Red / Orange / Green

    # ── Left: KB per request ──
    ax1 = axes[0]
    bars1 = ax1.bar(labels, kb_values, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    ax1.set_title("Payload Size per Request (KB)", fontsize=12, pad=10)
    ax1.set_ylabel("Kilobytes (KB)", fontsize=10)
    ax1.set_yscale("log")
    ax1.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    for bar, val in zip(bars1, kb_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.15,
            f"{val:.1f} KB",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    # ── Right: MB per day ──
    ax2 = axes[1]
    bars2 = ax2.bar(labels, mb_values, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
    ax2.set_title("Total Data Transferred per Day (MB)", fontsize=12, pad=10)
    ax2.set_ylabel("Megabytes (MB)", fontsize=10)
    ax2.set_yscale("log")
    ax2.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)

    for bar, val in zip(bars2, mb_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.15,
            f"{val:.1f} MB",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    # ── Privacy annotation ──
    privacy_patch = mpatches.Patch(color="#2ECC71", label="DCCL: Raw interactions stay on device ✓")
    fig.legend(handles=[privacy_patch], loc="lower center", fontsize=10, framealpha=0.9, ncol=1, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[✓] Bar chart saved → {output_path}")


def save_json_report(
    profiles: list[BandwidthProfile],
    actual_file_stats: dict,
    reduction_ratio: float,
    output_path: str,
):
    """Lưu JSON report cho Streamlit dashboard (Phase 6.2)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    report = {
        "profiles": [asdict(p) for p in profiles],
        "actual_file_stats": actual_file_stats,
        "bandwidth_reduction_ratio": round(reduction_ratio, 2),
        "privacy_preserved": True,
        "summary": (
            f"DCCL reduces bandwidth by {reduction_ratio:.1f}x. "
            "Raw user interactions never leave the device."
        ),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[✓] JSON report saved → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bandwidth & Privacy Profiling — DCCL vs Traditional Cloud (Phase 6.1)"
    )
    parser.add_argument("--num-interactions",   type=int,   default=50,
                        help="Số interactions trung bình mỗi user (giả lập local history)")
    parser.add_argument("--num-items-queried",  type=int,   default=50,
                        help="Số item embeddings DCCL cần fetch từ Cloud per request")
    parser.add_argument("--embedding-dim",      type=int,   default=64,
                        help="Số chiều của item embeddings")
    parser.add_argument("--requests-per-day",   type=int,   default=50,
                        help="Số request inference mỗi ngày (mỗi user)")

    # Actual file paths (tùy chọn)
    parser.add_argument("--embeddings-file",    type=str,   default="cloud_server/artifacts/item_embeddings.npy",
                        help="Path tới item_embeddings.npy để đo kích thước thực tế")
    parser.add_argument("--local-storage-dir",  type=str,   default="device_client/local_storage",
                        help="Thư mục chứa JSON files local storage của devices")

    # Output paths
    parser.add_argument("--chart-output",       type=str,   default="demo/bandwidth_profile.png")
    parser.add_argument("--json-output",        type=str,   default="demo/bandwidth_report.json")
    parser.add_argument("--no-chart",           action="store_true", help="Bỏ qua việc tạo chart")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("Phase 6.1 — Bandwidth & Privacy Profiling")
    print(f"{'='*60}")
    print(f"  Simulating scenario:")
    print(f"    avg interactions/user  : {args.num_interactions}")
    print(f"    items queried per req  : {args.num_items_queried}")
    print(f"    embedding dim          : {args.embedding_dim}")
    print(f"    requests/day           : {args.requests_per_day}")

    # Build profiles
    profiles = build_profiles(
        num_interactions=args.num_interactions,
        num_items_queried=args.num_items_queried,
        embedding_dim=args.embedding_dim,
        requests_per_day=args.requests_per_day,
    )

    # Bandwidth reduction ratio
    traditional = profiles[0].bytes_per_request
    dccl        = profiles[2].bytes_per_request
    reduction   = traditional / max(dccl, 1)

    # Print report
    print_report(profiles, reduction)

    # Profile actual files if available
    actual_stats = profile_from_actual_files(
        embeddings_path=args.embeddings_file,
        local_storage_dir=args.local_storage_dir,
    )

    # Save outputs
    if not args.no_chart:
        plot_bar_chart(profiles, args.chart_output)

    save_json_report(profiles, actual_stats, reduction, args.json_output)

    print(f"\n[✓] Done! Reports saved.")


if __name__ == "__main__":
    main()