### 🌟 Bức tranh tổng thể (Overview)
Nếu vác toàn bộ cục `hetero_data` khổng lồ ở Bài 3 đút thẳng vào Card đồ họa (GPU) để train thì máy tính sẽ "cháy RAM" và sập ngay lập tức.
Nhiệm vụ của Notebook 4 là thiết kế một **"Phễu rót dữ liệu" (Data Loader)**. Nó chia nhỏ đồ thị lớn ra thành từng cụm nhỏ (Mini-batch) và mớm từ từ cho AI học. Bài toán mà mô hình AI sẽ giải quyết là **Link Prediction (Dự đoán liên kết)** - tức là đoán xem ông Khách A có khả năng "kéo một đường dây" tới Món Hàng B trong tương lai hay không.

---

### 🔍 Giải thích chi tiết từng bước (Step-by-Step Breakdown)

#### Bước 1. Chia tách Cạnh để Học và Thi (RandomLinkSplit) (Dòng 56 - 104)
Để dạy AI dự đoán xem User có mua Item hay không, ta phải giấu đi một số lượt mua trước mặt nó để còn lấy cái mà chấm điểm.
Hàm `RandomLinkSplit` đã xẻ các sợi dây nối thành 3 tập:
*   **Tập Train (80%):** Dùng để học.
*   **Tập Validation (10%):** Thi thử.
*   **Tập Test (10%):** Thi thật cuối kỳ.
*Điều đặc biệt ở Graph:* Ngay trong chính tập Train, nó lại phải cắt 20% cạnh ra làm **Supervision edges (Bài tập thực hành)**, và 80% cạnh để làm **Message-passing edges (Sách giáo khoa)**. Mô hình chỉ được nhìn "Sách" để giải "Bài tập" chứ không được ăn gian cạch chéo Bài tập.

#### Bước 2. Học từ Hàng xóm (LinkNeighborLoader) (Dòng 106 - 156)
Thay vì load cả triệu đỉnh, `LinkNeighborLoader` hoạt động theo nguyên lý "Ăn khoainh vùng": 
*   **Thuật toán Lấy mẫu (Num_neighbors = [15, 10]):** Giả sử muốn dạy AI về Khách A, nó chỉ bốc Khách A, cùng **tối đa 15** món đồ Khách A từng chạm (Hop 1). Từ 15 Món đồ đó, nó lại bốc lan ra thêm tối đa **10** thiết bị khác/người khác (Hop 2). AI sẽ học từ một "Cộng đồng nhỏ" (Subgraph) xung quanh người dùng đó thay vì nhìn cả trái đất.
*   **Negative Sampling (Dựng hiện trường giả):** Trong data thực tế chỉ có các nét vẽ CÓ mua hàng (Nhãn số `1`). Nếu AI chỉ thấy số 1, nó sẽ sinh tật hễ thấy ai cũng bảo "Có mua". Do đó, thuật toán lén cài cắm thêm các cạnh KHÔNG mua hàng (Nhãn `0`) với tỉ lệ `1:1`. Nó bốc bừa 1 ông khách và 1 món ông ấy chưa bao giờ mua để nhét vào mồm AI bắt AI phải nói "Không".

#### Bước 3. Mở xem thử các rổ Mini-batch (Dòng 159 - 186)
Notebook in thử ra màn hình 3 lượt mớm đầu tiên (Batch 0, 1, 2) để xem cái phễu rót ra được gì.
*   Khi thiết lập `BATCH_SIZE = 256`, tức là lấy ra 256 cạnh Có Mùa Hàng. 
*   Cộng thêm hệ thống nhân bản giả mạo `1:1` ở bước 2, ta thấy Terminal nhả ra `Positive edges: 256` và `Negative edges: 256`. 
*   Loader lấp đầy các mảng Tensor rồi xuất ra. Kỹ sư nhẩm tính thấy chuẩn bài nên Pass qua bước lưu trữ.

#### Bước 4. Sao lưu 3 tập (Dòng 205 - 213)
Cuối cùng, notebook lưu 3 tập `train_data.pt`, `val_data.pt`, `test_data.pt` xuống ổ cứng để chuẩn bị cắm vào Trái tim của hệ thống: Dàn AI Deep Learning ở bài sau.

---

### 💡 Gợi ý tóm tắt để bạn đi giảng lại
> *"Các bạn cứ hiểu Notebook 4 giống như một thầy giáo ra đề thi. Thầy giáo (RandomLinkSplit) xé bỏ và giấu đi 10% sợi dây trong mạng lưới để làm Đề thi tốt nghiệp (Test). Ở phần dữ liệu dạy học (Loader), vì lớp quá đông nên lớp trưởng phải chia thành nhóm nhỏ (batch). Cứ mỗi lần giải bài về một User, lớp trưởng sẽ chỉ mời khoảng 15 người/vật gần nhất làm 'hàng xóm' ném vào GPU cho đỡ nặng máy. Đặc biệt, để học sinh không học vẹt, thầy giáo cài cắm luật 'Negative Sampling': Cứ cho 1 câu 'Có mua' thì thầy bịa thêm 1 câu 'Không mua' để rèn AI phải thật sự thấu hiểu. Cuối cùng, cất giáo trình Train/Val/Test đi chuẩn bị bước lên thớt huấn luyện!"*