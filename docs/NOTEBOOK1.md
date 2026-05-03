### 🌟 Overview
Mục đích chính của notebook này là **Chuyển đổi dữ liệu thô (đã qua làm sạch) thành cấu trúc mạng lưới (Đồ thị - Graph)** để chuẩn bị cho việc huấn luyện một mô hình AI (Graph Neural Network - GNN). 
Điều đặc biệt là notebook này thiết kế dữ liệu theo kiến trúc phân tán **Cloud-Device** (Đám mây và Thiết bị cục bộ):
*   **Mô hình Cloud:** Sẽ học các hành vi chung từ phần lớn người dùng.
*   **Thiết bị cục bộ (Device):** Đại diện cho những người dùng hoàn toàn mới, giữ dữ liệu của họ một cách cục bộ, mô hình sẽ phải suy luận trực tiếp cho họ mà không cần gửi dữ liệu lên Cloud.

---

### 🔍 Step-by-Step Breakdown

#### Step 1: Initialization and Loading Data (Line 1 - 10)
Đầu tiên, notebook tạo một thư mục `data/processed` để chuẩn bị chứa các file kết quả cuối cùng. Sau đó nó tải 2 bảng dữ liệu đã được làm sạch trước đó:
*   **`clean_reviews`**: Chứa lịch sử đánh giá sản phẩm (User Tương tác với Item).
*   **`clean_meta`**: Chứa thông tin chi tiết về món hàng (Item Info).

#### Step 2: "Cloud vs Device" Data Split - Inductive Split (Line 12 - 26)
*Lưu ý: Đây là phần quan trọng nhất thể hiện tính chất "Cloud - Device" của dự án.*
*   **Cách làm quen thuộc:** Thông thường khi train AI, ta hay giấu đi 20% _lượt mua_ của một user để test.
*   **Cách làm trong Notebook này:** Thay vì cắt nhỏ lượt mua, hệ thống lấy ra toàn bộ danh sách khách hàng và **tách hẳn 10% số lượng khách hàng (Users) ra một góc**.
    *   **90% Seen Users (Khách hàng trên Cloud):** Dữ liệu của những người này sẽ được dùng để train trực tiếp trên Server trung tâm.
    *   **10% Hidden Users (Khách hàng trên Device):** Nhóm này được xem như những "thiết bị điện thoại mới". Hệ thống Cloud **hoàn toàn không biết họ là ai**. Về sau, mô hình sẽ phải dự đoán đồ cho họ dựa trên mô hình đã train ở Cloud đẩy xuống thiết bị.

#### Step 3: Node Mapping (Line 30 - 38)
Dữ liệu gốc có ID người dùng và ID món hàng là các chuỗi ký tự dài (VD: `A38NELQT98S4H8`).
*   Tuy nhiên, các thư viện Deep Learning (PyTorch) hoạt động cốt lõi bằng các phép nhân ma trận toán học, vì thế không thể lưu chữ cái vào ma trận được.
*   **Chức năng:** Quá trình "Mapping" giống như việc nhà trường cấp **Mã số sinh viên (Từ 0, 1, 2, ... đến N)** để dễ quản lý.
*   Notebook biến tất cả User ID và Item ID thành các số nguyên (integer), sau đó lưu "bảng đối chiếu" này vào file json. Đáng lưu ý là nó **chỉ cấp mã số cho 90% user của Cloud (seen_users)**, 10% user bí mật kia hoàn toàn không được cấp mã số để đảm bảo tính cô lập.

#### Step 4: Bipartite Edge Mapping (User-Item Graph) (Line 40 - 49)
Sử dụng dữ liệu của bộ 90% người dùng trên Cloud, notebook xây dựng nên các "cạnh" (edges) kết nối giữa User và Item:
*   Cứ mỗi khi khách hàng mua/đánh giá một món, ta vẽ một đường thẳng nối số thứ tự của khách hàng đó với số thứ tự của món hàng.
*   Kết quả thu được là một Tensor tên `user_item_edge_index` có dạng `[2, tổng_số_lượt_tương_tác]`. Mảng thứ 1 là mảng ID của User, mảng thứ 2 là mảng ID tương ứng của món hàng mà họ đã tương tác.

#### Step 5: Item-Item Edge Mapping ("Also Buy" Graph) (Line 51 - 75)
Ngoài việc nối Người với Sản Phẩm, hệ thống còn muốn hiểu rõ mối quan hệ giữa các món hàng với nhau.
*   Trong bộ Meta data, mỗi sản phẩm đi kèm một trường `also_buy` (Người mua món này cũng mua món kia).
*   Notebook tìm cách vẽ thêm đường chéo kết nối món A với món B nếu chúng hay được mua chung. 
*   Quá trình này bóc tách list (`also_buy`), đổi nó ra số nguyên thứ tự, và kết thành thêm một Tensor đồ thị `item_item_edge_index`.

#### Step 6: Save Artifacts (Line 93 - 96)
Cuối cùng, tất cả các công cụ đã được lắp ráp xong:
1.  Lưu `user_item_edge_index.pt`: Đồ thị (Graph) dạng PyTorch nối người và món.
2.  Lưu `item_item_edge_index.pt`: Đồ thị dạng PyTorch thể hiện mối quan hệ riêng biệt giữa các món hàng với nhau.
3.  Lưu `hidden_interactions_test.parquet`: Cất giấu các lượt mua của **10% khách hàng bí mật** vào một file riêng biệt. File này hoàn toàn vắng mặt trong tập Train và chỉ được lấy ra ở Test phase (mô phỏng giai đoạn Test trên End-point Devices).

---

### 💡 Elevator Pitch Summary
Bạn có thể tóm tắt cho mọi người bằng một ví dụ dễ hiểu như sau:

> *"Trong Notebook 1 này, ta đóng vai vị thần phân chia dữ liệu để thiết lập mô hình máy học chia cấp (Devive-Cloud). Ta lấy toàn bộ khách hàng, cắt riêng ra 10% khách hàng làm 'người dùng thiết bị cá nhân (device)' giấu đi để sau này thi cuối kỳ. Còn lại 90% khách được đưa vào trường huấn luyện 'Đám mây đám đông (Cloud)'. Đối với 90% này, ta cấp cho mỗi người và món hàng một số báo danh (0,1,2...). Cuối cùng ta căng dây chằng chịt: cứ người mua món nào thì căng dây từ người sang món, hai món hàng nào hay được mua lẻ tẻ cùng nhau ta cũng căng một dây nối chúng lại. Hệ thống dây nhợ này được lưu dạng PyTorch Tensor, sẵn sàng đút vào một mạng thần kinh đồ thị Graph Neural Networks ở Notebook sau."*