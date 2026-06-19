# BÁO CÁO PHÂN TÍCH KẾT QUẢ BENCHMARK (DAY 17 - TRACK 03)
## Đề tài: Hệ thống bộ nhớ cho AI Agent (Memory Systems for AI Agent)

---

## 1. Kết quả Benchmark Thực tế

Dưới đây là bảng số liệu thu thập được khi chạy benchmark song song cho hai Agent:

### 1.1. Standard Benchmark (Hội thoại ngắn/vừa - `data/conversations.json`)
| Tên Agent | Agent tokens sinh ra | Prompt tokens xử lý | Khả năng nhớ chéo phiên (Recall) | Chất lượng phản hồi | Tăng trưởng bộ nhớ (bytes) | Số lần nén (Compactions) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent** | 1,172 | 14,703 | 11.90% | 11.90% | 0 | 0 |
| **Advanced Agent** | 1,171 | 17,992 | **80.71%** | **80.71%** | 2,830 | 0 |

### 1.2. Long-Context Stress Benchmark (Hội thoại siêu dài - `data/advanced_long_context.json`)
| Tên Agent | Agent tokens sinh ra | Prompt tokens xử lý | Khả năng nhớ chéo phiên (Recall) | Chất lượng phản hồi | Tăng trưởng bộ nhớ (bytes) | Số lần nén (Compactions) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent** | 515 | 24,688 | 0.00% | 0.00% | 0 | 0 |
| **Advanced Agent** | 531 | **14,106** | **83.33%** | **83.33%** | 188 | **26** |

---

## 2. Phân tích Chi tiết các Chỉ số & Trade-offs

### 2.1. Vì sao Advanced Agent có khả năng ghi nhớ chéo phiên (Recall) vượt trội?
* **Cơ chế hoạt động**:
  * **Baseline Agent** chỉ giữ bộ nhớ ngắn hạn dạng "in-memory history" trong cùng một thread ID. Khi hệ thống bắt đầu một thread/session mới để hỏi câu hỏi recall, Baseline Agent hoàn toàn không có dữ kiện lịch sử, dẫn đến tỷ lệ recall gần như bằng 0 (chỉ đạt 11.90% ở Standard Benchmark do trùng từ khóa ngẫu nhiên và 0.00% ở Stress Benchmark).
  * **Advanced Agent** sử dụng lớp **Persistent Memory** (lưu trữ file `User.md` trên ổ đĩa). Mỗi khi người dùng chia sẻ thông tin, Agent sẽ trích xuất và cập nhật trực tiếp vào file cá nhân của họ. Khi sang thread mới, Advanced Agent tự động đọc file `User.md` này và chèn vào prompt, giúp đạt tỷ lệ recall xuất sắc từ **80.71% đến 83.33%**.

### 2.2. Vì sao Advanced Agent lại tiêu tốn nhiều prompt tokens hơn ở hội thoại ngắn?
* Ở Standard Benchmark, Advanced Agent tiêu thụ **17,992** prompt tokens, cao hơn Baseline Agent (**14,703** prompt tokens, chênh lệch khoảng +22%).
* **Nguyên nhân**: Ở các cuộc hội thoại ngắn, tổng số lượng token của lịch sử chat chưa vượt qua ngưỡng nén (`compact_threshold_tokens = 400`). Lúc này, cơ chế nén chưa hoạt động. Tuy nhiên, Advanced Agent luôn phải gánh thêm phần nội dung hồ sơ lưu trữ dài hạn (`User.md`) và chèn nội dung đó vào prompt hệ thống ở từng lượt chat. Phần overhead này làm tăng lượng context đầu vào ở mỗi lượt tương tác ngắn.

### 2.3. Vì sao cơ chế Compact giúp Advanced Agent tối ưu chi phí ở hội thoại dài?
* Ở Stress Benchmark (hội thoại siêu dài), tổng lượng prompt token của Baseline Agent phình to lên tới **24,688** tokens, trong khi Advanced Agent chỉ tiêu thụ **14,106** tokens (**tiết kiệm được 42.8%**).
* **Nguyên nhân**: 
  * Baseline Agent mang theo toàn bộ lịch sử hội thoại thô qua mỗi lượt chat mới. Chi phí prompt tăng theo hàm số mũ/bậc hai của chiều dài hội thoại.
  * Advanced Agent sử dụng **CompactMemoryManager** và đã thực hiện **26 lần nén** (compactions). Khi kích thước hội thoại vượt quá ngưỡng, agent tự động tóm tắt các hội thoại cũ thành một đoạn text ngắn và chỉ giữ lại 2 tin nhắn gần nhất dưới dạng đầy đủ. Nhờ đó, kích thước prompt context được khống chế ở mức ổn định, ngăn chặn hiện tượng phình to token.

### 2.4. Sự tăng trưởng của File Memory (`User.md`) và Rủi ro đi kèm
* **Xu hướng tăng trưởng**: 
  * Ở Standard Benchmark, file lưu trữ tăng trưởng **2,830 bytes** do tích lũy nhiều thông tin cá nhân của người dùng qua 10 phiên hội thoại.
  * Ở Stress Benchmark, mặc dù hội thoại rất dài nhưng dung lượng file chỉ tăng **188 bytes** do người dùng chỉ xoay quanh việc đính chính và lặp lại thông tin (và cơ chế lọc nhiễu của agent hoạt động hiệu quả).
* **Rủi ro đi kèm**:
  1. **Dữ liệu rác/Nhiễu**: Nếu không lọc tốt, agent sẽ lưu cả những thông tin mang tính tạm thời, các câu đùa hoặc thông tin sai lệch vào file dài hạn. Điều này làm tăng kích thước file và loãng ngữ cảnh.
  2. **Chi phí đọc đĩa (I/O) và Token tích lũy**: File `User.md` phình càng to thì chi phí đọc/ghi file càng lớn và lượng prompt token cơ bản ở mọi lượt chat đều tăng theo tuyến tính.
  3. **Xung đột thông tin (Conflicts)**: Nếu người dùng thay đổi trạng thái (ví dụ: chuyển từ backend sang MLOps, đổi nơi ở từ Huế sang Đà Nẵng), nếu agent chỉ thêm thông tin mới vào cuối file mà không sửa/ghi đè thông tin cũ, LLM sẽ nhận được hai dữ kiện mâu thuẫn và trả lời sai.

---

## 3. Các Giải pháp Nâng cao đã Triển khai (Bonus)

Để khắc phục các rủi ro trên, chúng tôi đã hiện thực hóa hai tính năng cốt lõi:
1. **Lọc nhiễu hội thoại (Noise Filtering)**: Agent tự động nhận diện và bỏ qua các câu đùa (ví dụ: chuyển sang làm product manager) hoặc thông tin tạm thời (đi công tác Hà Nội) dựa trên từ khóa ngữ cảnh tiếng Việt trước khi ghi vào hồ sơ.
2. **Giải quyết xung đột thông tin (Conflict Handling)**: Lớp lưu trữ cấu trúc hóa facts dưới dạng các khóa cặp (`Tên`, `Nơi ở`, `Nghề nghiệp`, `Đồ uống yêu thích`...). Khi có thông tin đính chính mới, agent tự động thay thế/ghi đè lên thông tin cũ tại khóa tương ứng trong `User.md`, đảm bảo hồ sơ luôn đồng nhất và gọn gàng.
