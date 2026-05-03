### 🌟 Bức tranh tổng thể (Overview)
Notebook 1 đã tạo ra "Khung xương" (các cạnh nối). Notebook 2 đã đúc ra "Thịt và Não" (các ma trận đặc trưng 384-chiều).
Nhiệm vụ duy nhất của Notebook 3 là **Lắp ráp (Assembly) tất cả những mảnh ghép rời rạc đó lại thành một Thực thể hoàn chỉnh duy nhất**. 
Thực thể này được gọi là **Heterogeneous Graph (Đồ thị dị thể)**. Điểm khác biệt của Đồ thị dị thể là nó cho phép tồn tại *nhiều loại Đỉnh khác nhau* (User, Item) và *nhiều loại Cạnh khác nhau* (cạnh 'mua đồ' và cạnh 'mua kèm đồ') trong cùng một mạng lưới, thay vì chỉ có 1 loại đỉnh như đồ thị truyền thống.

---

### 🔍 Giải thích chi tiết từng bước (Step-by-Step Breakdown)

#### Bước 1. Lấy rổ linh kiện (Dòng 7 - 21)
Notebook tải lại toàn bộ 4 file kết quả từ 2 bài trước:
*   Mảng não bộ: `x_user.pt`, `x_item.pt`
*   Khung xương dây nối: `user_item_edge_index.pt`, `item_item_edge_index.pt`

#### Bước 2. Lắp ráp Robot HeteroData (Dòng 23 - 39)
Notebook khởi tạo một Đồ thị dị thể rỗng bằng lệnh `data = HeteroData()` của thư viện PyTorch Geometric, sau đó đính từng bộ phận vào đúng vị trí của nó với những "Cái tên" (Labels) cực kỳ dễ hiểu:
*   Gắn não cho User: `data['user'].x = x_user`
*   Gắn não cho Item: `data['item'].x = x_item`
*   Gắn dây thần kinh nối User với Item: Đặt tên đường nối là `'reviews'` -> `data['user', 'reviews', 'item'].edge_index = user_item_edge_index`
*   Gắn dây thần kinh nối hai Item với nhau: Đặt tên đường nối là `'also_bought'` -> `data['item', 'also_bought', 'item'].edge_index = item_item_edge_index`

#### Bước 3. Khám bệnh tổng quát (Sanity Checks) (Dòng 41 - 69)
Trước khi xuất xưởng, kỹ sư phải test xem mình cắm dây có bị nhầm hay lỏng lẻo không.
*   Hệ thống dùng lệnh `assert` kiểm tra: Mảng có bị lệch chiều không? Số dây nối có lỡ đâm xuyên qua một ID User không tồn tại (out of range) hay không?
*   Nếu mọi thứ in màu xanh `All sanity checks passed!`, nghĩa là robot lắp chuẩn.

#### Bước 4. Xuất xưởng Thành phẩm (Dòng 71 - 76)
Toàn bộ Thực thể Đồ thị này được nén lại vào 1 file duy nhất mang tên **`hetero_data.pt`**. Kể từ các notebook thiết kế Model sau này, ta chỉ cần load đúng một file này là có đầy đủ toàn bộ cả hệ sinh thái.

#### Bước 5. Trực quan hóa Mạng lưới Dị thể (Dòng 78 - 136)
*Lưu ý: Notebook này đã có sẵn code trực quan hóa ở cuối cùng!*
Nó đã lấy ra 200 nét vẽ review và 100 nét vẽ also_bought để in ra màn hình. Khi chạy cell này, bạn sẽ thấy một mạng lưới phân tầng tuyệt đẹp: User (màu xanh dương) sẽ chỉ mũi tên tấn công vào Item (màu cam), trong khi các Item (Màu cam) lại tiếp tục chỉ mũi tên sang các Item khác. Đây chính là biểu hiện rực rỡ nhất của thuật ngữ "Heterogeneous" (Đa dạng/Dị thể).

---

### 💡 Gợi ý tóm tắt để bạn đi giảng lại
> *"Các bạn tưởng tượng, ở Bài 1 ta đúc ra Khung xương, Bài 2 ta tạo ra Khối óc. Sang Bài 3, chúng ta giống như Bác sĩ Frankenstein, cắm tất cả mọi thứ vào một Siêu thực thể mang tên `HeteroData`. Thực thể dị thể (Heterogeneous) này rất thông minh vì nó phân biệt được rõ ràng đâu là xương tay (User đánh giá Item) và đâu là xương sườn (Item liên quan tới Item), chứ không bị lẫn lộn. Cuối cùng, ta vặn đinh ốc kiểm tra lỗi (Sanity Checks) và đóng gói trọn bộ Siêu thực thể này vào một vali duy nhất tên là `hetero_data.pt` để mang lên Mây (Cloud) cho AI tập luyện!"*