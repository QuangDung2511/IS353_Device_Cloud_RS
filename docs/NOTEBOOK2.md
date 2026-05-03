### 🌟 Bức tranh tổng thể (Overview)
Trong Notebook 1, chúng ta đã xây dựng được "Bộ khung xương" (các đường nối/cạnh đồ thị). Ở Notebook 2 này, nhiệm vụ là đắp "thịt" lên các Node (đỉnh). 
Để con AI GNN (Graph Neural Network) có thể học được, nó cần biết **đặc trưng riêng của từng Món hàng và từng Khách hàng** dưới dạng các phép toán (ma trận số). Quá trình biến đổi thông tin chữ viết (title, description) thành các mảng số liệu để AI hiểu gọi là **Feature Encoding (Mã hóa Đặc trưng)**.

---

### 🔍 Giải thích chi tiết từng bước (Step-by-Step Breakdown)

#### Bước 1. Đọc lại "Hồ sơ" từ bài trước (Dòng 10 - 26)
Notebook load lại các cuốn sổ địa chỉ ID (file json) và sơ đồ cột điện (đồ thị tương tác User-Item `edge_index` đã lưu ở bài 1). Nó cũng load lại thông tin `clean_meta` để lấy dữ liệu văn bản của hàng hóa.

#### Bước 2. Xử lý kịch bản cho Sản phẩm (Dòng 29 - 64)
*   Để máy tính hiểu một món hàng là gì, notebook đọc từng dòng thông tin của từng Item, cạo sửa sạch sẽ các lỗi đánh máy, lỗi cấu trúc mảng (list/array).
*   Sau đó nó **ghép "Tiêu đề" (title) và "Mô tả sản phẩm" (description) lại thành một đoạn văn duy nhất**. Dữ liệu lúc này đã sẵn sàng để đem đi dịch thuật cho máy hiểu.

#### Bước 3. Dùng "Não ngôn ngữ" đọc và mã hóa Item (Dòng 67 - 76)
*Lưu ý: Đây là phần cốt lõi (NLP - Xử lý ngôn ngữ tự nhiên).*
*   Máy tính không biết đọc chữ. Do đó, notebook tải một mô hình AI ngôn ngữ rất nổi tiếng tên là **`all-MiniLM-L6-v2`** (sử dụng thư viện `sentence-transformers`).
*   Mô hình này đóng vai trò như một "cỗ máy xay thịt": Nó nuốt các đoạn văn ghép ở Bước 2 vào, và "nhả" ra các vector (mảng số). Mỗi món hàng giờ đây được đại diện bởi **một dãy đúng 384 con số** (chiều không gian 384). Người ta gọi đây là dãy **Embedding**.
*   *Ý nghĩa:* Nhờ nhúng qua AI ngôn ngữ, hai món hàng có ý nghĩa tương đồng nhau (VD: "Áo khoác gió" và "Áo lạnh mùa đông") sẽ có hai mảng 384 con số rất giống nhau, giúp AI phân loại sau này. Kết quả này được lưu tạm vào `item_features`.

#### Bước 4. Mã hóa Người dùng theo định lý "Bạn là những gì bạn mua" (Dòng 79 - 109)
Vấn đề lớn: Dataset thường không có hồ sơ miêu tả Users (không biết tuổi tác, giới tính khách hàng). Vậy làm sao để có dải 384 con số đại diện cho User?
*   Notebook sử dụng một logic rất thông minh: **"Khách hàng được định nghĩa bởi lịch sử mua hàng của họ"**.
*   Thuật toán sẽ dò trên đồ thị ở Bài 1 xem Khách User_A đã tương tác với món đồ số 1, số 5 và số 9. 
*   Sau đó, nó lấy 3 mảng số 384-chiều của 3 món đồ đó (vừa tạo ở bước 3) cộng lại và **chia trung bình cộng (Mean Aggregation)**.
*   **Kết quả:** Con số trung bình đó chính là *Hồ sơ Sở thích (User Profile)* của User_A. Nếu khách hàng chưa từng mua gì (bị lỗi hoặc out of tập data), họ tạm thời mang một mảng gồm toàn số 0. Mảng này được lưu vào `user_features`.

#### Bước 5. Xuất Dữ liệu Đặc trưng (Dòng 112 - 114)
*Lắp ráp xong:*
1.  Lưu `x_dict_item.pt`: Chứa ma trận các đặc trưng của món hàng (chuyển thể từ text). Chữ `x` trong AI GNN thường dùng để ám chỉ Features (đặc điểm).
2.  Lưu `x_dict_user.pt`: Chứa ma trận các đặc trưng của người dùng (tính bằng trung bình các món họ mua).

---

### 💡 Gợi ý tóm tắt để bạn đi giảng lại
> *"Mọi người tưởng tượng Notebook 2 như một dây chuyền làm căn cước công dân cho Khách hàng và Món hàng. Máy tính không hiểu text, nên ta mướn một mô hình xử lý ngôn ngữ NLP đọc mô tả hàng hóa và đúc nó ra thành 384 con số. Vậy món hàng đã có ID căn cước đặc trưng! Còn khách hàng thì không có hồ sơ sinh học, ta định danh họ bằng cách bốc tất cả các món đồ họ từng mua, đem 384 con số của các món đó chia trung bình ra. Nhờ vậy, ta thu được ma trận não bộ của cả hai bên để chuẩn bị cắm vào mạng Graph Neural Network."*