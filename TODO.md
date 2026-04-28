# 📋 Project TODO

---

##  Data Engineering & Graph Construction

### Nhiệm vụ 1.1: Bipartite Edge Mapping & Inductive Split

- [x] Tải `clean_reviews.parquet` và `clean_meta.parquet`
- [x] Tạo continuous integer ID mappings (0 → N-1) cho tất cả `reviewerID` (Users) và `asin` (Items) duy nhất
- [x] Áp dụng **90/10 split** trên `reviewerID` → 10% users làm **unseen inductive holdout set**
- [x] Trích xuất **User-Item edges**: tensor `[2, num_interactions]` — bipartite interaction graph của training set
- [x] Trích xuất **Item-Item edges**: phân tích `also_buy` / `also_view` → tensor `[2, num_item_edges]` cho global item graph

### Nhiệm vụ 1.2: Textual Node Feature Encoding

- [x] Khởi tạo pre-trained NLP encoder (`sentence-transformers/all-MiniLM-L6-v2`)
- [x] Xử lý `title` + `description` của từng item → dense feature vectors `x_dict['item']` shape `[num_items, hidden_dim]`
- [x] Khởi tạo user features `x_dict['user']` (chiến lược: `torch.zeros([num_users, hidden_dim])` hoặc mean vector từ các items đã tương tác)

### Nhiệm vụ 1.3: PyG HeteroData Instantiation

- [x] Cấu trúc đối tượng `torch_geometric.data.HeteroData`
- [x] Gán node features: `data['user'].x` và `data['item'].x`
- [x] Gán edge indices: `data['user', 'reviews', 'item'].edge_index` và `data['item', 'also_bought', 'item'].edge_index`

---

##  Data Loader & Link Predictor Engineering

### Nhiệm vụ 1.1: Thiết lập Subgraph Mini-Batching

- [x] Khởi tạo `torch_geometric.loader.LinkNeighborLoader`
- [x] Xác định `edge_label_index` → edge type `('user', 'reviews', 'item')` cho training split
- [x] Cấu hình `num_neighbors` (ví dụ: `[15, 10]`) để giới hạn 2-hop computational graph
- [x] Triển khai **dynamic negative sampling** với `neg_sampling_ratio=1.0` bên trong loader

### Nhiệm vụ 1.2: Xây dựng Edge Decoder Module

- [x] Định nghĩa class `torch.nn.Module` (ví dụ: `DotProductDecoder`)
- [x] Triển khai `forward(user_emb, item_emb)`
- [x] Tính element-wise multiplication + sum (hoặc shallow MLP) → output 1D logits tensor shape `[batch_size]`

---

##  GraphSAGE Architecture & Inductive Evaluation

### Nhiệm vụ 2.1: Xây dựng GraphSAGE Encoder

- [x] Định nghĩa `torch.nn.Module` dùng `SAGEConv` trong `HeteroConv` cho từng loại cạnh (tương đương hetero GraphSAGE)
- [x] Triển khai kiến trúc **2-layer message-passing** với `in_channels` và `hidden_channels`
- [x] Áp dụng `relu` và `Dropout` giữa các convolution layers

### Nhiệm vụ 2.2: Vòng lặp huấn luyện GraphSAGE

- [x] Khởi tạo SAGE encoder + Edge Decoder
- [x] Khởi tạo `BCEWithLogitsLoss` và `Adam` optimizer
- [x] Thực hiện mini-batch training loop trên `LinkNeighborLoader` với backpropagation qua encoder-decoder

### Nhiệm vụ 2.3: Thực thi Inductive Inference

- [x] Áp dụng `torch.no_grad()` và chuyển model sang `eval()`
- [x] Trích xuất validation/test `HeteroData` subgraph chứa 10% inductive holdout users
- [x] Thực hiện forward pass → tính toán động representations cho **unseen user nodes**

**Triển khai:** [`notebooks/06_graphsage_train_eval.ipynb`](notebooks/06_graphsage_train_eval.ipynb) — encoder `HeteroConv` + `SAGEConv` (tương đương hetero GraphSAGE), cạnh `rev_reviews`, checkpoint `data/processed/graphsage_link_pred.pt`.

---

##  GCN Architecture & Transductive Baseline Evaluation

### Nhiệm vụ 3.1: Xây dựng GCN Encoder

- [x] Định nghĩa `torch.nn.Module` dùng `GCNConv` bọc trong `to_hetero`
- [x] Triển khai kiến trúc **2 tầng** với `hidden_channels` giống hệt GraphSAGE (để kiểm soát parameter counts)

### Nhiệm vụ 3.2: Vòng lặp huấn luyện GCN

- [x] Khởi tạo GCN encoder + Edge Decoder
- [x] Khởi tạo `BCEWithLogitsLoss` và `Adam` optimizer
- [x] Thực hiện training loop trên training topology đã cung cấp

### Nhiệm vụ 3.3: Kiểm chứng các hạn chế Transductive

- [x] Áp dụng `torch.no_grad()` và chuyển model sang `eval()`
- [x] Thử nghiệm forward pass trên cùng validation/test `HeteroData` subgraph của Thành viên 2
- [x] Ghi lại: structural dimension mismatches, OOM errors, hoặc embedding errors để chứng minh hạn chế của GCN's symmetric normalization matrix trên isolated/unseen subgraphs

---

## Phase 4: Cloud Service Architecture (Global Knowledge)

### Nhiệm vụ 4.1: Trích xuất Cloud Artifacts
- [ ] Export `Item` embeddings (`[num_items, hidden_channels]`) từ GraphSAGE checkpoint đã huấn luyện (`graphsage_link_pred.pt`).
- [ ] Trích xuất trọng số (weights/biases) của các `SAGEConv` layers thuộc nhánh User.
- [ ] Lưu trữ item embeddings dưới dạng memory-mapped array hoặc file cấu trúc (e.g., `NumPy` `.npy`, `HDF5`) để tối ưu tốc độ truy vấn (O(1) lookup).

### Nhiệm vụ 4.2: Xây dựng FastAPI Global Server
- [ ] Khởi tạo ứng dụng `FastAPI` đóng vai trò là Cloud Server.
- [ ] Triển khai REST Endpoint `GET /api/v1/items/` nhận tham số là danh sách `item_ids`.
- [ ] Trả về (serve) serialized dense embedding vectors tương ứng với requested items.
- [ ] (Tùy chọn) Triển khai Endpoint `POST /api/v1/model/update` để nhận gradient updates từ devices (Federated Learning).

---

## Phase 5: Device Simulation & On-Device Inference (TFLite)

### Nhiệm vụ 5.1: Subgraph Extraction & TFLite Conversion
- [ ] Tách `Hidden Users` (từ inductive split) làm tập thiết bị (mobile clients) giả lập.
- [ ] Lưu trữ local history (1-hop interactions của từng unseen user) thành các file JSON riêng biệt giả lập SQLite on-device storage.
- [ ] Chuyển đổi mô hình (Model Conversion): Isolate nhánh User `SAGEConv` aggregator và `DotProductDecoder`.
- [ ] Export PyTorch subgraph module sang **ONNX** → compile sang **TensorFlow Lite (`.tflite`)** để chạy trên mobile inference engine.

### Nhiệm vụ 5.2: Simulated Local Inference Pipeline
- [ ] Thiết lập Python script giả lập Device client sử dụng `tflite-runtime`.
- [ ] **Step A:** Đọc local interaction history (Item IDs) từ file JSON.
- [ ] **Step B:** Gửi HTTP request đến FastAPI Cloud để fetch các pre-trained item embeddings.
- [ ] **Step C:** Feed embeddings vào local `.tflite` interpreter thực hiện neighborhood aggregation.
- [ ] **Step D:** Tính toán offline dot-product logits để sinh ra ranked recommendation list (top-K items).

---

## Phase 6: System Demonstration & Evaluation (DCCL Requirements)

### Nhiệm vụ 6.1: Bandwidth & Privacy Profiling
- [ ] Viết script phân tích payload size (Bytes/Kilobytes).
- [ ] So sánh kích thước truyền tải: Truyền raw interaction logs (Traditional Cloud) vs Truyền queried dense embeddings (DCCL).
- [ ] Vẽ bar chart trực quan hóa **Bandwidth reduction** và chứng minh local privacy (raw interactions never leave the device).

### Nhiệm vụ 6.2: Interactive Streamlit Dashboard
- [ ] Xây dựng 2-panel UI bằng `Streamlit`.
- [ ] **Left Panel (Device Context):** Hiển thị local history, API call log (chỉ gửi ID), và Top-K inference results chạy qua TFLite.
- [ ] **Right Panel (Cloud Context):** Hiển thị server status, API throughput, và global graph architecture.
