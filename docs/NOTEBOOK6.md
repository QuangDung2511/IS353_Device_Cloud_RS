### 🌟 Bức tranh tổng thể (Overview)
Notebook 6 có nhiệm vụ **Huấn luyện (Training)** mô hình AI có tên là **GraphSAGE**. 
Sự vĩ đại của thuật toán GraphSAGE nằm ở chỗ: Nhờ học cách "tổng hợp thông tin từ hàng xóm", nó không chỉ học thuộc lòng những khách hàng cũ (Cloud), mà còn có năng lực siêu phàm là **suy diễn (Inductive) ra sở thích của người dùng hoàn toàn mới (Device)** – những người chưa từng tồn tại trong Database lúc nó học.

---

### 🔍 Giải thích chi tiết từng bước (Step-by-Step Breakdown)

#### Bước 1. Thiết kế Não bộ (GraphSAGE Encoder) (Dòng 48 - 90)
GNN Encoder được xây từ 2 tầng `SAGEConv` (hàm tính toán mượn của thuật toán GraphSAGE).
*   *Bí quyết hai chiều:* Mũi tên tương tác ban đầu chỉ đi 1 chiều: `User -> Item`. Để não bộ học được hiệu quả, kỹ sư gọi hàm `add_reverse_reviews` (Dòng 92). Hàm này nhân bản thêm một đường ngược lại `Item -> User`. Nhờ vậy, User biết được thông tin của đồ vật, và Đồ vật cũng hấp thụ ngược lại được phong cách của User.

#### Bước 2. Vòng lặp Học tập (Training Loop) (Dòng 229 - 269)
*   **Optimizier & Loss:** Notebook dùng thuật toán tối ưu `Adam` và hàm lỗi `BCEWithLogitsLoss`.
*   **Quy trình 5 Kỷ nguyên (Epochs=5):** Quá trình huấn luyện diễn ra lặp đi lặp lại:
    1. Đưa cục dữ liệu (batch) qua **Encoder (GraphSAGE)** để tạo mảng não bộ 256 chiều.
    2. Ném mảng đó qua **Decoder (DotProduct)** để nó nhả ra điểm số.
    3. So sánh điểm số với đáp án thực (Label 1 hay 0).
    4. Cập nhật lại các "dây nơ-ron" (weights) để vòng sau đoán chuẩn hơn. 
*   Quá trình liên tục in ra các điểm số như `AUROC` (khả năng phân biệt) và `AP` (Độ chính xác) qua từng Epoch để kỹ sư theo dõi mô hình có đang khôn lên hay không.

#### Bước 3. Sao lưu Trí khôn (Checkpointing) (Dòng 275 - 294)
Sau khi AI học xong và pass qua bài thi trên tập khách hàng quen, toàn bộ "trọng số chất xám" của nó được cất vào két sắt mang tên `graphsage_link_pred.pt`. Đây chính là file Model quý giá trị giá hàng nghìn giờ bọc lá.

#### Bước 4. Bài Kiểm tra Cuối cấp: Năng lực Inductive (Dòng 297 - 410)
*Đây chính là bài test "Device-Cloud" thần thánh của dự án!* Ngay bên dưới dòng chữ `Inductive holdout users`, là lúc hệ thống trích xuất **10% khách hàng bí mật** từ file `hidden_interactions_test.parquet` ở Bài 1.
*   Con AI **chưa từng nhìn mặt** những khách hàng này một giây nào trên Cloud. Đặc trưng (Feature) ban đầu của 10% khách hàng này bị xóa sạch thành một **Mảng đục lỗ toàn số 0** (Dòng 344: `torch.zeros`).
*   **Điều kỳ diệu xảy ra:** Bằng cách nhét đồ thị mới của nhóm khách này qua Model GraphSAGE (không train lại), con AI tự động nhìn ngó xung quanh xem món hàng họ vừa cầm là món gì, rồi **suy luận ngược (Inductive reasoning)** ra não bộ của họ!
*   Khả năng xuất thần đó được đem đi thi test tỷ lệ AUROC. Để chứng minh AI không phải ăn rùa, hệ thống còn tung ra một bài test ngẫu nhiên (Random baseline) được tầm ~50% (đoán mò). Nếu điểm Inductive của SAGE lớn hơn hẳn 50%, tức là thuật toán đã thành công rực rỡ!

---

### 💡 Gợi ý tóm tắt để bạn đi giảng lại
> *"Mọi người hình dung Notebook 6 này chính là Lò Bát Quái. Chúng ta ném dữ liệu từ bài 4, thuật toán giải mã từ bài 5 vào đây và kích hoạt bộ não 2 tầng GraphSAGE. GraphSAGE không học vẹt, nó học 'cách' để chắt lọc thông tin từ những người xung quanh. Để chứng minh sức mạnh của nó, khúc cuối notebook tung ra một chiêu bài cực khó: Thả 10% Khách Hàng Hoàn Toàn Xa Lạ (tượng trưng cho 1 cái điện thoại Device mới toanh) vào mạng lưới với xuất phát điểm não bộ bằng 0. Chỉ bằng cách nhìn xem những khách hàng mới này cầm vào món đồ nào, GraphSAGE tự động suy luận ra được sở thích của họ và đoán cực kỳ chính xác xem họ sẽ mua gì tiếp theo. Đó chính là tinh hoa của Inductive Graph Learning!"*
