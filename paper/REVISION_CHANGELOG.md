# 🛠️ NHẬT KÝ SỬA ĐỔI TOÀN DIỆN BẢN THẢO BÀI BÁO (REVISION CHANGELOG)
## Bám sát 100% Feedback của GVHD (TS. Huỳnh Võ Trung Dũng)

> **Thời điểm cập nhật:** 24/10/2026  
> **Tác giả:** Đỗ Trọng Minh Quân  
> **File mã nguồn cập nhật:** `paper/LaTeX-Templates/main.tex`, `paper/overleaf_clean/main.tex`, `paper/Overleaf_ICIIT2027_Submission.zip`  

---

## 1. Bảng Tổng Hợp Đối Chiếu Trước & Sau Sửa Đổi (Before vs. After)

| Tiêu chí | Bản thảo ban đầu (Initial Draft) | Bản thảo sửa đổi theo Thầy (Revised Draft) | Trạng thái nghiệm thu |
| :--- | :--- | :--- | :---: |
| **Tiêu đề bài báo** | *... An Empirical Investigation on Edge-AI Feasibility* (Bị Thầy nhắc nhở vì chưa test phần cứng nhúng) | **`Multi-Frame Super-Resolution and Vision Transformers for Low-Resolution License Plate Recognition`** | ✅ Đã chuẩn hóa đúng thực tế |
| **Dung lượng Abstract** | 328 từ (Quá dài, vượt khung quy định) | **241 từ** (Nằm trọn vẹn trong dải chuẩn 200 – 300 từ) | ✅ Đạt chuẩn 200-300 từ |
| **Sử dụng Bullet Points** | 18 bullets rải rác ở Intro, Method, Findings (Dấu hiệu AI) | **0 bullet points** (Xóa sạch 100%; chuyển hoàn toàn sang văn xuôi học thuật liền mạch) | ✅ Triệt tiêu hoàn toàn bullet |
| **Cấu trúc Sections** | 6 sections chính + 14 subsections/subsubsections vụn vặt | **Đúng 4 Section chính** (+ Abstract & References) theo lệnh Thầy: `Introduction`, `Methodology`, `Results and Discussion`, `Conclusion` | ✅ Chuẩn hóa cấu trúc bài |
| **Số lượng Công thức** | 16 công thức (Nhiều công thức toán cơ bản, SGK làm loãng bài) | **Đúng 4 công thức then chốt** đại diện cho giải pháp kỹ thuật trực tiếp tạo ra kết quả thực nghiệm | ✅ Tinh gọn toán học thực chiến |
| **Nói quá về Edge-AI** | Tuyên bố triển khai thực tế trên Edge-AI | Chỉ báo cáo **Parameter size** và **Inference latency** trên thiết bị đo thực tế (MPS/GPU), loại bỏ mọi suy diễn thổi phồng | ✅ Trung thực 100% |
| **Thứ tự Trích dẫn** | Rải rác, không theo thứ tự xuất hiện | Được sắp xếp xuất hiện tuần tự theo mạch văn từ `[1]` đến `[13]` | ✅ Chuẩn trích dẫn ACM |
| **Giới hạn số trang** | Dự kiến 6-7 trang nếu dàn trải | Được cô đọng chặt chẽ trong **tối đa 5 trang** (chuẩn `sigconf` 2 cột) | ✅ Đạt trần $\le 5$ trang |

---

## 2. Chi Tiết Các Hạng Mục Đã Khắc Phục Cụ Thể

### Mục 1: Tiêu đề và Tuyên bố khoa học (Scientific Claims)
- **Hành động:** Loại bỏ hoàn toàn cụm từ *"An Empirical Investigation on Edge-AI Feasibility"*.
- **Cơ sở:** Chúng ta đo độ trễ trên chip Apple Silicon MPS và huấn luyện trên Kaggle T4, chưa cắm lên bo mạch nhúng Jetson Nano hay Raspberry Pi. Bài báo chỉ khiêm tốn nhận định kiến trúc Flow 3 "gọn nhẹ, tiết kiệm tham số và suy luận nhanh, mở ra tiềm năng cho hệ thống giám sát thực tế".

### Mục 2: Viết lại Abstract (241 từ)
- **Cấu trúc đoạn văn duy nhất:**
  1. Đặt vấn đề suy giảm ảnh biển số trong giám sát giao thông thực tế ($16 \times 64$ pixels, nhòe chuyển động).
  2. Hai kiến trúc đề xuất: Flow 2 (Frame-Parallel SR StackedSRNet kết hợp ResNet-34) và Flow 3 (Pre-LN Vision Transformer với Spatial-to-Sequence Height Pooling).
  3. Phương pháp hậu xử lý: Bayesian log-prior theo quy chuẩn vị trí Mercosur.
  4. Số liệu thực nghiệm trung thực: Flow 2 đạt $78.08\%$ Exact Match và $5.72\%$ CER, xóa bỏ $100\%$ lỗi `8 ↔ B`; Multi-Flow Synergy Ensemble (Flow 2 + Flow 3 + Mercosur syntax prior) đạt mốc kỷ lục thực tế **$80.38\%$** (803 / 999 tracks đúng) với CER giảm sâu xuống $5.22\%$; Flow 3 đạt $75.78\%$ với chỉ $14.4\text{ M}$ tham số (giảm $55\%$) và độ trễ $6.82\text{ ms}$.

### Mục 3: Xóa bỏ toàn bộ Bullet Points & Gom nhóm Sections
- **Tại Introduction:**
  - Chuyển 3 gạch đầu dòng về "Thách thức kỹ thuật" thành 1 đoạn văn xuôi phân tích nguyên nhân vật lý (spatial phase misalignment, gradient instability).
  - Chuyển 4 câu hỏi nghiên cứu ($RQ_1 \to RQ_4$) thành các câu văn phân tích học thuật gắn liền trong mạch văn.
  - Chuyển 4 gạch đầu dòng "Đóng góp" thành đoạn văn tổng kết đóng góp liền mạch, không đánh số hay dấu chấm tròn.
- **Gom nhóm mục:**
  - Không còn mục `Related Work` riêng lẻ; các nghiên cứu nền tảng được trích dẫn trực tiếp ngay trong `Introduction` và `Methodology`.
  - Gộp toàn bộ `Experiments` và `Discussion` thành **`Results and Discussion`** duy nhất.
  - Xóa toàn bộ các subsubsection chia nhỏ gây vụn bài.

### Mục 4: Tinh giản công thức toán học — Chỉ giữ lại những gì tạo ra kết quả
1. **$I_f^{SR} = \mathcal{G}_{SR}(\hat{I}_f; \Theta_{SR})$:** Định nghĩa toán tử Siêu phân giải song song từng khung hình, giải thích lý do tại sao triệt tiêu được hiện tượng bóng ma (ghosting).
2. **$\mathbf{s}_w = \frac{1}{H_{grid}} \sum_{h=1}^{H_{grid}} Z_{h, w}$:** Công thức Height Pooling — đóng góp then chốt cứu mạng Transformer không bị vỡ đường thẳng thời gian của CTC Loss.
3. **$\mathcal{L}_{total} = \mathcal{L}_{CTC} + \lambda_{SR}(\mathcal{L}_1 + \lambda_{perc}\mathcal{L}_{perc})$:** Hàm mục tiêu đa nhiệm kết hợp với hệ số bảo vệ gradient.
4. **$\log \tilde{P}(c_t) = \log P_{model}(c_t) + \beta \log P_{prior}(c_t | pos)$:** Công thức tích hợp phân phối tiên nghiệm vị trí ký tự Mercosur giải thích cho việc tăng $+3.80\%$ Exact Match mà không cần train lại.
*(Đã loại bỏ: Các công thức biến đổi affine 2D thông thường, định nghĩa Sigmoid/ReLU của RCAB, khai triển chuỗi CTC lý thuyết suông).*

### Mục 5: Kiểm soát độ dài tối đa 5 trang
- Tối ưu hóa khoảng cách hình vẽ `Figure 1` và 2 bảng số liệu `Table 1` (Master Benchmark) & `Table 2` (Confusion Matrix).
- Định dạng 2 cột `sigconf` chặt chẽ, không có khoảng trắng thừa, kết thúc trọn vẹn ở trang thứ 5 kèm đầy đủ danh mục tài liệu tham khảo.

---

## 3. Khắc Phục Triệt Để Toàn Bộ Cảnh Báo & Lỗi Trong Overleaf Log

1. **Lỗi `Package hyperxmp Error: hyperref must be loaded before hyperxmp`:**
   - **Nguyên nhân:** File mẫu `acmart.cls` bản cũ (2023) nạp gói `hyperxmp` trước `hyperref`. TeX Live 2026 trên Overleaf yêu cầu bắt buộc `hyperref` phải được nạp trước `hyperxmp`.
   - **Khắc phục:** Đã đảo thứ tự nạp trong file `acmart.cls` mới trong `overleaf_clean/`. *(Hoặc bạn chỉ cần xóa file `acmart.cls` trên Overleaf thì Overleaf sẽ tự động dùng bản mới nhất của hệ thống).*
2. **Cảnh báo `A possible image without description`:**
   - **Khắc phục:** Bổ sung thẻ tiếp cận `\Description{...}` cho `Figure 1` theo đúng chuẩn của ACM.
3. **Cảnh báo `Overfull \hbox` (Bảng 1 và Bảng 2 bị tràn lề):**
   - **Khắc phục:** Thêm `\setlength{\tabcolsep}{4.5pt}` cho Bảng 1 (toàn trang) và bọc `\resizebox{\columnwidth}{!}{...}` cho Bảng 2 (1 cột), giúp bảng co giãn vừa khít 100% không tràn lề.
4. **Loạt cảnh báo BibTeX (`empty publisher`, `empty address`, `no volume/pages`):**
   - **Khắc phục:** Đã bổ sung đầy đủ nhà xuất bản (`IEEE`, `Springer`, `ACM`), địa chỉ xuất bản và số trang chuẩn cho toàn bộ 13 tài liệu tham khảo trong `references.bib`. Xóa sạch mọi cảnh báo BibTeX.
