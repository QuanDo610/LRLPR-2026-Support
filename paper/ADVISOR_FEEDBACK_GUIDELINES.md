# 📌 QUY CHUẨN & CHỈ ĐẠO HỌC THUẬT TỪ GVHD (TS. HUỲNH VÕ TRUNG DŨNG)
## Về Bản thảo Bài báo Hội nghị ICIIT 2027

> **Ngày ghi nhận:** 24/10/2026  
> **Áp dụng cho:** Bài báo khoa học nộp Hội nghị Quốc tế ICIIT 2027 (ACM ICPS)  
> **Nguyên tắc tối thượng:** Không dùng phong cách viết của AI; viết cô đọng, khúc chiết theo phong cách nhà nghiên cứu (human academic writing); trung thực 100% với thực nghiệm, không bịa đặt, không thổi phồng.

---

## 1. Chi tiết 5 Ý kiến Phản biện của Thầy & Giải pháp Khắc phục

### ⚠️ Feedback 1: Loại bỏ tiêu đề thổi phồng "Edge-AI Feasibility"
* **Nhận xét của Thầy:** *"An Empirical Investigation on Edge-AI Feasibility" - bạn đang chế biến cái gì đây ? Tự đề bịa hơi lớn. Có test trên Edge AI device chưa ?*
* **Nguyên nhân:** Mô hình chỉ được benchmark độ trễ (latency) trên Apple Silicon MPS và huấn luyện trên Kaggle T4 GPU, chưa từng đo đạc thực tế trên phần cứng nhúng Edge-AI (như Jetson Nano, Raspberry Pi, TPU Edge). Việc đưa "Edge-AI Feasibility" vào tiêu đề là nói quá (overclaim).
* **Quy chuẩn sửa đổi:** 
  - Đổi tiêu đề về đúng bản chất khoa học và đề tài luận văn:  
    **`Multi-Frame Super-Resolution and Vision Transformers for Low-Resolution License Plate Recognition`**
  - Trong bài báo: Chỉ đề cập đến tính chất tinh gọn tham số (Parameter efficiency) và thời gian suy luận (Inference latency), tuyệt đối không tuyên bố "đã thử nghiệm trên thiết bị Edge-AI".

---

### ⚠️ Feedback 2: Giới hạn Abstract nghiêm ngặt 200 - 300 từ
* **Nhận xét của Thầy:** *Abstract 200-300 từ.*
* **Quy chuẩn sửa đổi:**
  - Viết lại Abstract cô đọng: Bối cảnh $\to$ Thách thức $16 \times 64$ $\to$ Kiến trúc Frame-Parallel SR + ViT Height Pooling $\to$ Kết quả thực nghiệm thật ($78.08\%$, $80.38\%$ ensemble, 0 lỗi `8/B`, 14.4M tham số).
  - Đếm từ chính xác trong dải **220 – 260 từ**, không thừa không thiếu.

---

### ⚠️ Feedback 3: Triệt tiêu hoàn toàn Bullet Points & Chuẩn hóa Cấu trúc 5 Section
* **Nhận xét của Thầy:** 
  - *Không dùng bullet cho tất cả các section (đây là đặc trưng khi dùng AI).*
  - *Chỉ có các section sau: Abstract, Introduction, Methodology, Results and Discussion, Conclusion, Reference.*
  - *Tất cả các section, subsection khác bỏ (AI nó hay sinh ra).*
  - *Trích dẫn phải theo thứ tự.*
* **Quy chuẩn sửa đổi:**
  - **Xóa 100% các môi trường bullet list** (`\begin{itemize}`, `\begin{enumerate}`). Toàn bộ các ý (thách thức, câu hỏi nghiên cứu, đóng góp, phân tích) phải được viết thành **các đoạn văn văn xuôi học thuật liền mạch (continuous cohesive prose)** có từ nối học thuật chặt chẽ.
  - **Cấu trúc duy nhất gồm đúng 5 Section chính:**
    1. `\begin{abstract} ... \end{abstract}`
    2. `\section{Introduction}` (Gộp bối cảnh, thách thức, RQ, và đóng góp vào văn xuôi, không chia mục con)
    3. `\section{Methodology}` (Mô tả pipeline từ STN, SR đến Flow 2 và Flow 3 liền mạch)
    4. `\section{Results and Discussion}` (Gộp kết quả thực nghiệm và bàn luận vào chung một mục)
    5. `\section{Conclusion}`
    6. `\bibliography{references}` (References)
  - **Bỏ tất cả các subsection, subsubsection vụn vặt** do AI tự sinh (`Related Work`, `Research Questions`, `Ablation Studies`, `Discussion` riêng lẻ gom gọn lại).
  - Trích dẫn tài liệu tham khảo được xuất hiện tuần tự theo thứ tự xuất hiện `[1], [2], [3], ...` trong văn bản.

---

### ⚠️ Feedback 4: Thanh lọc công thức thừa & Gắn kết toán học với thực nghiệm
* **Nhận xét của Thầy:**
  - *Công thức mà không dùng, hoặc kết quả không có từ công thức đó, đưa vào là biết AI làm.*
  - *Bài em viết 1 đống công thức. Vậy thì kết quả lấy từ công thức nào? Kiến trúc layer của mấy cái network đó lấy công thức nào xây dựng nên?*
* **Nguyên nhân:** AI thường có xu hướng "chép sách giáo khoa", nhồi nhét các công thức toán phổ thông (như định nghĩa hàm Sigmoid, công thức Global Average Pooling chuẩn, biến đổi afin tọa độ chung chung) làm loãng bài báo và tạo cảm giác ngụy tạo học thuật.
* **Quy chuẩn sửa đổi:**
  - **Lọc bỏ toàn bộ công thức toán râu ria, hiển nhiên:** Xóa các công thức định nghĩa Sigmoid, ReLU, affine 2D cơ bản, GAP chung.
  - **Chỉ giữ lại các công thức cốt lõi đại diện cho giải pháp kỹ thuật cụ thể đã đem lại kết quả thực tế:**
    1. Công thức xử lý Frame-Parallel Super-Resolution: $I_f^{SR} = \mathcal{G}_{SR}(\hat{I}_f; \Theta_{SR})$.
    2. Công thức **Spatial-to-Sequence Height Pooling** (giải pháp cốt lõi cứu Flow 3 thoát khỏi sụp đổ CTC): $\mathbf{s}_w = \frac{1}{H_{grid}} \sum_{h=1}^{H_{grid}} Z_{h, w}$.
    3. Hàm mất mát tối ưu đa nhiệm có trọng số thực tế: $\mathcal{L}_{total} = \mathcal{L}_{CTC} + \lambda_{SR}(\mathcal{L}_1 + \lambda_{perc}\mathcal{L}_{perc})$.
    4. Công thức hậu xử lý Bayesian Log-Prior tại vị trí 4: $\log \tilde{P}(c_t) = \log P_{model}(c_t) + \beta \log P_{prior}(c_t | pos)$.
  - Mọi công thức giữ lại đều có văn bản giải thích rõ: nó dẫn tới cột số liệu nào trong Bảng 1 và Bảng 2.

---

### ⚠️ Feedback 5: Giới hạn độ dài tối đa 5 trang
* **Nhận xét của Thầy:** *Tối đa 5 trang thôi nhe em.*
* **Quy chuẩn sửa đổi:**
  - Sử dụng định dạng 2 cột chuẩn hội nghị ACM: `\documentclass[sigconf]{acmart}`.
  - Căn chỉnh độ dài bài báo, bảng biểu và hình vẽ sao cho toàn bộ nội dung từ trang 1 đến References nằm trọn vẹn trong **4 đến 5 trang**, không bị tràn sang trang thứ 6.
