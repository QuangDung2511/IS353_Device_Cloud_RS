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

- [ ] Định nghĩa `torch.nn.Module` dùng `SAGEConv` bọc trong `to_hetero` cho bipartite structure
- [ ] Triển khai kiến trúc **2-layer message-passing** với `in_channels` và `hidden_channels`
- [ ] Áp dụng `relu` và `Dropout` giữa các convolution layers

### Nhiệm vụ 2.2: Vòng lặp huấn luyện GraphSAGE

- [ ] Khởi tạo SAGE encoder + Edge Decoder
- [ ] Khởi tạo `BCEWithLogitsLoss` và `Adam` optimizer
- [ ] Thực hiện mini-batch training loop trên `LinkNeighborLoader` với backpropagation qua encoder-decoder

### Nhiệm vụ 2.3: Thực thi Inductive Inference

- [ ] Áp dụng `torch.no_grad()` và chuyển model sang `eval()`
- [ ] Trích xuất validation/test `HeteroData` subgraph chứa 10% inductive holdout users
- [ ] Thực hiện forward pass → tính toán động representations cho **unseen user nodes**

---

##  GCN Architecture & Transductive Baseline Evaluation

### Nhiệm vụ 3.1: Xây dựng GCN Encoder

- [ ] Định nghĩa `torch.nn.Module` dùng `GCNConv` bọc trong `to_hetero`
- [ ] Triển khai kiến trúc **2 tầng** với `hidden_channels` giống hệt GraphSAGE (để kiểm soát parameter counts)

### Nhiệm vụ 3.2: Vòng lặp huấn luyện GCN

- [ ] Khởi tạo GCN encoder + Edge Decoder
- [ ] Khởi tạo `BCEWithLogitsLoss` và `Adam` optimizer
- [ ] Thực hiện training loop trên training topology đã cung cấp

### Nhiệm vụ 3.3: Kiểm chứng các hạn chế Transductive

- [ ] Áp dụng `torch.no_grad()` và chuyển model sang `eval()`
- [ ] Thử nghiệm forward pass trên cùng validation/test `HeteroData` subgraph của Thành viên 2
- [ ] Ghi lại: structural dimension mismatches, OOM errors, hoặc embedding errors để chứng minh hạn chế của GCN's symmetric normalization matrix trên isolated/unseen subgraphs
