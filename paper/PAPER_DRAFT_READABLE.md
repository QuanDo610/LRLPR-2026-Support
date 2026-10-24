# 📘 BẢN DỊCH & GIẢI THÍCH CHI TIẾT TỪNG ĐOẠN TỪ `main.tex`
## Bài Báo Khoa Học Hội Nghị Quốc Tế ICIIT 2027 (ACM ICPS)

> **Đề tài:** Multi-Frame Super-Resolution and Vision Transformers for Low-Resolution License Plate Recognition  
> *(Nhận diện biển số xe độ phân giải thấp đa khung hình bằng Siêu phân giải và Vision Transformers)*  
> **Tác giả:** Đỗ Trọng Minh Quân (First Author) & TS. Huỳnh Võ Trung Dũng (Advisor & Corresponding Author)  
> **Đơn vị:** Viện Quản trị & Công nghệ FSB, Trường Đại học FPT & Trường ĐH Quốc Tế, ĐHQG-HCM  
> **Tài liệu nguồn:** [`paper/overleaf_clean/main.tex`](file:///Users/admin/Desktop/LowResolutionImage-Quan/paper/overleaf_clean/main.tex)

---

## 📌 TIÊU ĐỀ BÀI BÁO (TITLE) & THÔNG TIN TÁC GIẢ

### 🇬🇧 Bản gốc tiếng Anh:
```latex
\title{Multi-Frame Super-Resolution and Vision Transformers for Low-Resolution License Plate Recognition}

\author{Do Trong Minh Quan}
\affiliation{%
  \institution{School of Business and Technology (FSB), FPT University}
  \city{Ho Chi Minh City}
  \country{Vietnam}
}
\email{quandtm0610@gmail.com}

\author{Huynh Vo Trung Dung}
\authornote{Corresponding author and Research Advisor (Dr. Steven Huynh).}
\affiliation{%
  \institution{International University, Vietnam National University Ho Chi Minh City}
  \city{Ho Chi Minh City}
  \country{Vietnam}
}
\email{hvtdung@hcmiu.edu.vn}
```

### 🇻🇳 Bản dịch tiếng Việt:
- **Tiêu đề:** Nhận diện biển số xe độ phân giải thấp đa khung hình kết hợp Siêu phân giải và Vision Transformers.
- **Tác giả thứ nhất:** Đỗ Trọng Minh Quân (Viện Quản trị & Công nghệ FSB, Trường Đại học FPT).
- **Giảng viên hướng dẫn & Tác giả liên hệ:** TS. Huỳnh Võ Trung Dũng (Dr. Steven Huynh, Trường Đại học Quốc tế, Đại học Quốc gia TP. Hồ Chí Minh).

💡 **Ý nghĩa học thuật:**
Tiêu đề bám sát 100% đề tài luận văn thạc sĩ của bạn, nêu bật 2 trụ cột kỹ thuật cốt lõi: **Multi-Frame Super-Resolution** (Siêu phân giải đa khung hình) và **Vision Transformers**. Toàn bộ cụm từ thổi phồng *"Edge-AI Feasibility"* đã được loại bỏ hoàn toàn theo góp ý của Thầy.

---

## 📌 TÓM TẮT BÀI BÁO (ABSTRACT)

### 🇬🇧 Bản gốc tiếng Anh (258 từ):
> Automatic License Plate Recognition (ALPR) in real-world traffic surveillance frequently encounters severe degradation when vehicle crops exhibit low spatial resolutions ($16 \times 64$ pixels), motion blur, and video compression artifacts. Under these constrained conditions, conventional sequence recognition networks suffer from feature collapse and stroke confusion, especially among morphologically similar alphanumeric characters such as `8' versus `B' and `0' versus `D'. In this paper, we present an end-to-end framework combining multi-frame super-resolution and vision transformers to restore character legibility and improve transcription accuracy. We develop two distinct model workflows: Flow 2, a frame-parallel Super-Resolution network (StackedSRNet) utilizing Residual Channel Attention Blocks (RCAB) coupled with an affine Spatial Transformer Network (STN) and a temporal-attention ResNet-34 backbone; and Flow 3, a compact Pre-LN Vision Transformer (ViT) incorporating Spatio-Temporal Cross-Attention and a specialized Spatial-to-Sequence Height Pooling operator designed to eliminate Connectionist Temporal Classification (CTC) divergence. Additionally, we integrate domain-specific Mercosur syntax calibration via a positional Bayesian log-prior and multi-flow synergy. Evaluated on the unconstrained blind test set of 999 video tracks from the ICPR 2026 Challenge (Scenario-B), our end-to-end Flow 2 model achieves 78.08\% Exact Match accuracy and a Character Error Rate (CER) of 5.72\%, completely eliminating all confusion between `8' and `B'. Combining Flow 2 and Flow 3 into a Multi-Flow Synergy Ensemble with Mercosur syntax calibration further boosts Exact Match accuracy to 80.38\% and reduces CER to 5.22\%. Meanwhile, our compact Flow 3 architecture attains 75.78\% Exact Match with only 14.4M parameters (a 55.1\% reduction) and an inference latency of 6.82 ms, demonstrating high architectural efficiency for practical traffic monitoring.

### 🇻🇳 Bản dịch tiếng Việt chuẩn xác:
> Hệ thống Nhận diện Biển số xe Tự động (ALPR) trong giám sát giao thông thực tế thường xuyên đối mặt với sự suy giảm chất lượng nghiêm trọng khi các vùng ảnh cắt xe có độ phân giải không gian rất thấp ($16 \times 64$ điểm ảnh), nhòe do chuyển động và nhiễu nén video. Dưới những điều kiện ngặt nghèo này, các mạng nhận dạng chuỗi truyền thống thường bị sụp đổ đặc trưng và nhầm lẫn nét chữ, đặc biệt là giữa các ký tự chữ-số có hình thái tương đồng như `8` với `B` và `0` với `D`. Trong bài báo này, chúng tôi đề xuất một khung kiến trúc đầu-cuối (end-to-end) kết hợp giữa siêu phân giải đa khung hình và mạng vision transformer nhằm phục hồi độ rõ nét của ký tự và nâng cao độ chính xác giải mã. Chúng tôi phát triển hai luồng mô hình riêng biệt: Flow 2, một mạng Siêu phân giải xử lý song song từng khung hình (StackedSRNet) sử dụng các khối Residual Channel Attention Block (RCAB) kết hợp với mạng biến đổi không gian afin (STN) và mạng xương sống ResNet-34 tích hợp cơ chế chú ý theo thời gian; và Flow 3, một mạng Vision Transformer (ViT) Pre-LN tinh gọn tích hợp cơ chế Cross-Attention Không-Thời gian cùng toán tử Height Pooling chuyển đổi không gian sang chuỗi nhằm triệt tiêu hiện tượng phân kỳ của hàm mất mát CTC. Ngoài ra, chúng tôi tích hợp cơ chế hiệu chuẩn ngữ pháp vị trí chuẩn Mercosur thông qua phân phối tiên nghiệm Bayes dạng logarit và hiệp đồng đa luồng. Được đánh giá trên tập kiểm thử độc lập gồm 999 vệt video thực tế thuộc Thử thách ICPR 2026 (Scenario-B), mô hình Flow 2 của chúng tôi đạt độ chính xác Đúng Tuyệt Đối (Exact Match) là 78.08% và Tỷ lệ Lỗi Ký tự (CER) là 5.72%, triệt tiêu 100% các nhầm lẫn giữa `8` và `B`. Việc kết hợp Flow 2 và Flow 3 thành mô hình Multi-Flow Synergy Ensemble cùng hiệu chuẩn ngữ pháp Mercosur tiếp tục nâng độ chính xác Exact Match lên mức 80.38% và giảm CER xuống 5.22%. Đồng thời, kiến trúc Flow 3 tinh gọn đạt 75.78% Exact Match với chỉ 14.4 triệu tham số (giảm 55.1%) và độ trễ suy luận siêu nhanh 6.82 ms, chứng minh hiệu quả kiến trúc vượt trội cho việc giám sát giao thông thực tế.

💡 **Điểm nhấn cốt lõi của Abstract:**
- Nêu bật ngay bài toán khó: Ảnh $16 \times 64$ px, xe chạy rung lắc, nhầm `8/B` và `0/D`.
- Đưa ra 2 vũ khí: **Flow 2** (Frame-Parallel SR + ResNet-34) và **Flow 3** (ViT + Height Pooling).
- Báo cáo số liệu thực nghiệm trung thực: Flow 2 đạt **$78.08\%$**, kết hợp Ensemble đa luồng đạt mốc **$80.38\%$**, xóa sạch $100\%$ lỗi `8 ↔ B`, và Flow 3 siêu nhẹ chỉ $14.4\text{ M}$ tham số / $6.82\text{ ms}$.

---

## 📌 MỤC 1: GIỚI THIỆU (SECTION 1: INTRODUCTION)

### Đoạn 1.1: Bối cảnh giám sát giao thông & Thách thức ảnh cực mờ ($16 \times 64$)
> **🇬🇧 English:**  
> Automatic License Plate Recognition (ALPR) serves as a core component in modern Intelligent Transportation Systems (ITS), electronic toll collection, and urban security monitoring [1]. While existing deep learning models achieve high accuracy on high-resolution frontal captures captured under favorable illumination, practical surveillance cameras routinely record low-quality footage [2]. Vehicles recorded at long distances or high velocities yield small license plate regions, often constrained to spatial dimensions of approximately $16 \times 64$ pixels [3]. At this low spatial resolution, character strokes merge into indistinct clusters, causing catastrophic misclassifications between structurally identical character pairs such as `8' versus `B', `0' versus `D', and `2' versus `7'. Standard geometric upsampling, such as bicubic interpolation, merely blurs pixel neighborhoods without recovering missing high-frequency textual contours [4, 5].

> **🇻🇳 Dịch tiếng Việt:**  
> Nhận diện Biển số xe Tự động (ALPR) đóng vai trò là thành phần cốt lõi trong các Hệ thống Giao thông Thông minh (ITS) hiện đại, hệ thống thu phí không dừng và giám sát an ninh đô thị [1]. Mặc dù các mô hình học sâu hiện nay đạt độ chính xác rất cao trên ảnh chụp chính diện độ nét cao dưới điều kiện ánh sáng lý tưởng, các camera giám sát thực tế lại thường xuyên ghi lại những đoạn phim chất lượng thấp [2]. Phương tiện di chuyển ở cự ly xa hoặc tốc độ cao tạo ra các vùng ảnh biển số bị thu nhỏ, thường bị giới hạn ở kích thước không gian chỉ khoảng $16 \times 64$ điểm ảnh [3]. Ở độ phân giải cực thấp này, các nét ký tự bị hòa lẫn vào nhau thành những cụm điểm ảnh mờ mịt, gây ra các lỗi nhận dạng thảm họa giữa những cặp ký tự có cấu trúc tương đồng như `8` với `B`, `0` với `D`, và `2` với `7`. Các phương pháp phóng to hình học truyền thống, chẳng hạn như nội suy song lập phương (bicubic), thực chất chỉ làm mờ các vùng lân cận mà không thể phục hồi được các đường nét viền tần số cao đã bị mất của văn bản [4, 5].

💡 **Ý nghĩa:** Giải thích lý do tại sao không thể dùng phép phóng to thông thường (Bicubic). Vì Bicubic chỉ làm mượt ảnh (low-pass filter), không sinh thêm chi tiết, làm dính các nét của số 8 vào chữ B.

---

### Đoạn 1.2: Các trở ngại kỹ thuật của dữ liệu đa khung hình (Multi-Frame Dilemmas)
> **🇬🇧 English:**  
> To address these limitations, multi-frame video tracks containing consecutive frames ($F = 5$) have become increasingly important for accumulating complementary spatial-temporal visual information [3]. However, processing multi-frame low-resolution tracks introduces major technical obstacles. First, naively concatenating raw video frames across color channels (e.g., stacking 5 frames into 15 channels) creates severe double-edge and ghosting artifacts because vehicle movement and camera vibrations cause inter-frame spatial misalignment. Second, jointly optimizing high-magnification Super-Resolution (SR) networks with Connectionist Temporal Classification (CTC) sequence decoders [6] frequently destabilizes training, as high CTC gradients can corrupt early convolutional filters. Third, directly applying Vision Transformers (ViTs) [7] to CTC decoders by flattening 2D feature patch grids breaks the horizontal spatial monotonicity required by CTC alignment, leading to training divergence.

> **🇻🇳 Dịch tiếng Việt:**  
> Để khắc phục các hạn chế này, các vệt video đa khung hình gồm các khung hình liên tiếp ($F = 5$) ngày càng trở nên quan trọng nhằm tích lũy thông tin thị giác bổ trợ lẫn nhau theo cả không gian và thời gian [3]. Tuy nhiên, việc xử lý chuỗi đa khung hình độ phân giải thấp lại vấp phải những trở ngại kỹ thuật lớn.  
> - **Thứ nhất**, việc ghép nối thô các khung hình video theo các kênh màu (ví dụ: ghép 5 frame thành 15 channels) tạo ra hiện tượng viền đôi và bóng ma (ghosting artifacts) nghiêm trọng, do chuyển động của xe và rung lắc của camera gây ra sự lệch pha không gian giữa các frame.  
> - **Thứ hai**, việc huấn luyện kết hợp đồng thời giữa mạng Siêu phân giải (SR) phóng đại cao với bộ giải mã chuỗi CTC [6] thường làm mất ổn định quá trình tối ưu hóa, do gradient có biên độ lớn từ CTC có thể phá hỏng các bộ lọc tích chập ban đầu của mạng SR.  
> - **Thứ ba**, việc nạp trực tiếp mạng Vision Transformer (ViT) [7] vào bộ giải mã CTC bằng cách duỗi phẳng lưới patch 2D theo thứ tự quét mành (raster-scan) sẽ phá vỡ tính đơn điệu không gian theo trục ngang mà thuật toán căn chỉnh CTC đòi hỏi, dẫn đến hiện tượng phân kỳ và sụp đổ huấn luyện.

💡 **Ý nghĩa:** Đây là đoạn văn học thuật xuất sắc thay thế hoàn toàn cho các gạch đầu dòng AI trước đó. Bạn chỉ ra chính xác 3 nguyên nhân vật lý/toán học giải thích tại sao các nghiên cứu trước bị kẹt độ chính xác: bóng ma lệch pha, gradient CTC phá bộ lọc SR, và lưới 2D của ViT xé rách nét chữ.

---

### Đoạn 1.3: Mục tiêu đề tài & Tóm tắt đóng góp giải pháp
> **🇬🇧 English:**  
> In this work, we investigate these challenges by formulating concrete research questions regarding the viability of multi-frame super-resolution, fine-grained morphological character disambiguation, compact transformer sequence modeling, and the impact of domain-specific syntactic priors. To this end, we design and evaluate a unified framework featuring a frame-parallel Super-Resolution module (StackedSRNet) with Residual Channel Attention Blocks (RCAB) [4] that reconstructs individual frames independently prior to temporal fusion, thus avoiding ghosting artifacts. We compare two sequence recognition models: Flow 2, which pairs a pre-trained ResNet-34 backbone with temporal attention and a Transformer encoder, and Flow 3, a compact Pre-LN Vision Transformer utilizing a novel Spatial-to-Sequence Height Pooling operator that preserves temporal order for CTC decoding. Finally, we formulate a post-hoc Bayesian log-prior based on Mercosur plate syntax to refine predictions without extra training cost.

> **🇻🇳 Dịch tiếng Việt:**  
> Trong công trình này, chúng tôi nghiên cứu các thách thức trên bằng việc xây dựng các câu hỏi nghiên cứu cụ thể xoay quanh tính khả thi của siêu phân giải đa khung hình, khả năng khử nhầm lẫn hình thái ký tự chi tiết, mô hình hóa chuỗi bằng mạng transformer tinh gọn, và tác động của các phân phối tiên nghiệm ngữ pháp chuyên miền. Để đạt được điều này, chúng tôi thiết kế và đánh giá một khung làm việc thống nhất gồm một khối Siêu phân giải song song từng khung hình (StackedSRNet) với các khối Residual Channel Attention Block (RCAB) [4] giúp tái tạo từng khung hình độc lập trước khi hợp nhất thời gian, nhờ đó triệt tiêu hoàn toàn hiện tượng bóng ma. Chúng tôi so sánh hai mô hình nhận dạng chuỗi: Flow 2 kết hợp mạng xương sống ResNet-34 tiền huấn luyện với cơ chế chú ý thời gian và bộ mã hóa Transformer; và Flow 3 là một mạng Vision Transformer Pre-LN tinh gọn áp dụng toán tử mới Height Pooling (chuyển đổi không gian sang chuỗi) giúp bảo toàn thứ tự thời gian cho bộ giải mã CTC. Cuối cùng, chúng tôi thiết lập phân phối tiên nghiệm Bayes dạng logarit dựa trên quy chuẩn ngữ pháp biển số Mercosur để tinh chỉnh kết quả dự đoán mà không tốn thêm bất kỳ chi phí huấn luyện lại nào.

💡 **Ý nghĩa:** Nêu rõ giải pháp kỹ thuật bạn đã làm: **StackedSRNet** xử lý từng frame song song (trọng số chia sẻ), hai nhánh mô hình **Flow 2 & Flow 3**, và bộ lọc **Bayesian Log-Prior**.

---

### Đoạn 1.4: Tóm tắt kết quả thực nghiệm đạt được
> **🇬🇧 English:**  
> Our empirical findings on 999 unconstrained video tracks from the ICPR 2026 Challenge (Scenario-B) demonstrate that Flow 2 achieves 78.08\% Exact Match accuracy and lowers the Character Error Rate (CER) to 5.72\%, completely eliminating the 19 baseline misclassifications between `8' and `B'. Combining Flow 2 and Flow 3 in a multi-flow ensemble with Mercosur syntax calibration pushes Exact Match accuracy to **80.38\%** (803 out of 999 tracks correctly recognized) and lowers CER to 5.22\%. In parallel, our compact Flow 3 architecture achieves 75.78\% Exact Match accuracy while using only 14.4M parameters (55.1\% fewer than Flow 2) with an inference latency of 6.82 ms, confirming its efficiency for practical traffic surveillance systems.

> **🇻🇳 Dịch tiếng Việt:**  
> Các kết quả thực nghiệm của chúng tôi trên 999 vệt video thực tế không ràng buộc từ Thử thách ICPR 2026 (Scenario-B) chứng minh rằng Flow 2 đạt độ chính xác Đúng Tuyệt Đối là 78.08% và hạ thấp Tỷ lệ Lỗi Ký tự (CER) xuống 5.72%, triệt tiêu hoàn toàn 19 lỗi nhận dạng nhầm giữa `8` và `B` của mô hình đường cơ sở. Việc kết hợp Flow 2 và Flow 3 thành mô hình ensemble đa luồng cùng bộ hiệu chuẩn ngữ pháp Mercosur đã đẩy độ chính xác Exact Match lên mức **80.38%** (nhận diện đúng 803 trên tổng số 999 vệt xe) và giảm CER xuống 5.22%. Song song với đó, kiến trúc Flow 3 tinh gọn đạt độ chính xác 75.78% Exact Match với chỉ 14.4 triệu tham số (ít hơn 55.1% so với Flow 2) cùng độ trễ suy luận là 6.82 ms, xác nhận hiệu năng vượt trội cho các hệ thống giám sát giao thông thực tế.

💡 **Ý nghĩa:** Khẳng định các kết quả thật: Flow 2 đạt **$78.08\%$**, xóa $100\%$ lỗi `8 ↔ B`, Ensemble đạt **$80.38\%$**, và Flow 3 chỉ nặng **$14.4\text{ M}$ params** với tốc độ **$6.82\text{ ms}$**.

---

## 📌 MỤC 2: PHƯƠNG PHÁP THỰC HIỆN (SECTION 2: METHODOLOGY)

### Đoạn 2.1: Mô hình hóa toán học bài toán
> **🇬🇧 English:**  
> The multi-frame low-resolution recognition task is defined over an input sequence $\mathcal{X} = \{I_1, I_2, \dots, I_F\}$ of $F = 5$ consecutive crops, where each frame $I_f \in \mathbb{R}^{3 \times 16 \times 64}$. The goal is to transcribe the true alphanumeric sequence $\mathcal{Y} = (y_1, y_2, \dots, y_L)$ of length $L \in \{7, 8\}$ over an alphabet $\mathcal{C}$ containing 36 classes (26 uppercase letters and 10 digits).

> **🇻🇳 Dịch tiếng Việt:**  
> Bài toán nhận diện độ phân giải thấp đa khung hình được định nghĩa trên một chuỗi đầu vào $\mathcal{X} = \{I_1, I_2, \dots, I_F\}$ gồm $F = 5$ khung hình liên tiếp cắt từ vệt chuyển động của xe, trong đó mỗi khung hình $I_f \in \mathbb{R}^{3 \times 16 \times 64}$ có 3 kênh màu (RGB), chiều cao 16 điểm ảnh và chiều rộng 64 điểm ảnh. Mục tiêu là giải mã chính xác chuỗi ký tự chữ-số nhãn ground-truth $\mathcal{Y} = (y_1, y_2, \dots, y_L)$ có độ dài $L \in \{7, 8\}$ trên tập bảng chữ cái $\mathcal{C}$ gồm 36 lớp (26 chữ cái viết hoa `A-Z` và 10 chữ số `0-9`).

---

### Đoạn 2.2: Căn chỉnh STN & Khối Frame-Parallel Super-Resolution
> **🇬🇧 English:**  
> The overall pipeline begins with an affine Spatial Transformer Network (STN) [8] that estimates perspective correction parameters for each frame to rectify rotational and perspective skews caused by oblique camera angles. To eliminate ghosting artifacts associated with channel-stacked inputs, each rectified frame $\hat{I}_f$ is processed independently through a shared Super-Resolution generator $\mathcal{G}_{SR}$:
> \begin{equation}
> I_f^{SR} = \mathcal{G}_{SR}(\hat{I}_f; \Theta_{SR}) \in \mathbb{R}^{3 \times 32 \times 128}, \quad \forall f \in [1, F]
> \label{eq:sr_parallel}
> \end{equation}
> The generator $\mathcal{G}_{SR}$ consists of an initial shallow convolution, 8 cascaded Residual Channel Attention Blocks (RCAB) [4] that selectively amplify high-frequency character boundary features, and a sub-pixel convolution layer (PixelShuffle) [5] that upsamples the spatial resolution by $2\times$ with a global bicubic residual skip connection.

> **🇻🇳 Dịch tiếng Việt:**  
> Toàn bộ luồng xử lý bắt đầu bằng một mạng Spatial Transformer Network (STN) [8] nhằm ước lượng các tham số hiệu chỉnh afin cho từng khung hình để nắn thẳng các góc xoay nghiêng và góc phối cảnh do camera đặt chéo. Để xóa bỏ hiện tượng bóng ma khi ghép kênh, từng khung hình sau khi nắn thẳng $\hat{I}_f$ được xử lý độc lập qua một mạng sinh Siêu phân giải dùng chung trọng số $\mathcal{G}_{SR}$:
> $$I_f^{SR} = \mathcal{G}_{SR}(\hat{I}_f; \Theta_{SR}) \in \mathbb{R}^{3 \times 32 \times 128}, \quad \forall f \in [1, F] \quad \text{(Phương trình 1)}$$
> Mạng sinh $\mathcal{G}_{SR}$ bao gồm một tầng tích chập nông ban đầu, 8 khối Residual Channel Attention Block (RCAB) nối tiếp nhau [4] nhằm khuếch đại chọn lọc các đặc trưng biên nét tần số cao của con chữ, và một tầng tích chập sub-pixel (PixelShuffle) [5] giúp phóng đại kích thước không gian lên $2\times$ ($32 \times 128$ điểm ảnh) kết hợp với đường truyền tắt cộng dư toàn cục từ ảnh phóng to bicubic.

💡 **Giải thích Công thức (1):**  
Đây là công thức cốt tử giải quyết hiện tượng bóng ma: Thay vì nhồi 5 frame vào 15 kênh (làm mạng bị loạn khi xe di chuyển lệch vị trí giữa các frame), ta cho từng frame chạy qua mạng SR độc lập $\mathcal{G}_{SR}$ nhưng **chia sẻ cùng bộ trọng số $\Theta_{SR}$**. Nhờ đó mỗi frame được phóng to $2\times$ lên $32 \times 128$ một cách sắc nét và độc lập.

---

### Đoạn 2.3: Luồng nhận dạng Flow 2 (ResNet-34 + Attention Fusion)
> **🇬🇧 English:**  
> Following super-resolution, we study two distinct recognition pipelines. In Flow 2, all $F$ reconstructed frames $\{I_f^{SR}\}_{f=1}^F$ are fed into a shared ImageNet-pretrained ResNet-34 backbone [9]. The resulting feature maps from each frame are aggregated along the temporal dimension using an attention-weighted pooling layer, serialized horizontally, and processed by a 2-layer Transformer encoder [10] with temporal stride $T = 32$ before projecting onto CTC linear prediction heads.

> **🇻🇳 Dịch tiếng Việt:**  
> Sau giai đoạn siêu phân giải, chúng tôi khảo sát hai luồng nhận dạng khác biệt. Trong Flow 2, toàn bộ $F = 5$ khung hình đã phục hồi $\{I_f^{SR}\}_{f=1}^F$ được nạp vào một mạng xương sống ResNet-34 dùng chung trọng số tiền huấn luyện ImageNet [9]. Các bản đồ đặc trưng thu được từ mỗi frame được gộp lại theo chiều thời gian thông qua một tầng Attention-weighted Pooling (gộp có trọng số chú ý), duỗi tuần tự theo chiều ngang và được xử lý bởi một bộ mã hóa Transformer gồm 2 lớp [10] với độ dài bước thời gian $T = 32$ trước khi chiếu lên đầu phân loại tuyến tính của CTC.

---

### Đoạn 2.4: Luồng nhận dạng Flow 3 (ViT + Height Pooling)
> **🇬🇧 English:**  
> In Flow 3, we implement a compact Vision Transformer [7] alternative. The super-resolved image is split into non-overlapping patches of size $4 \times 8$, producing a spatial grid of $H_{grid} = 8$ rows and $W_{grid} = 16$ columns ($N = 128$ tokens). After linear projection and positional encoding, tokens across all $F = 5$ frames are fused through Spatio-Temporal Cross-Attention, using the central frame ($f = 3$) tokens as Query and all concatenated frame tokens as Key and Value. The fused tokens are processed by a 6-layer Pre-LayerNorm Transformer encoder. To resolve the conflict between 2D raster-scan patch arrangements and the 1D temporal monotonicity required by CTC decoders, we introduce a Spatial-to-Sequence Height Pooling operator that averages tokens across the vertical height dimension:
> \begin{equation}
> \mathbf{s}_w = \frac{1}{H_{grid}} \sum_{h=1}^{H_{grid}} Z_{(h-1)W_{grid} + w}, \quad \forall w \in [1, W_{grid}]
> \label{eq:height_pool}
> \end{equation}
> The aggregated sequence $\mathbf{s} \in \mathbb{R}^{16 \times D}$ maps directly to 16 horizontal time steps, enabling smooth CTC sequence alignment.

> **🇻🇳 Dịch tiếng Việt:**  
> Trong Flow 3, chúng tôi triển khai một phương án mạng Vision Transformer [7] tinh gọn thay thế. Ảnh sau siêu phân giải ($32 \times 128$) được chia thành các mảnh vá (patches) không chồng lấn có kích thước $4 \times 8$, tạo ra một lưới không gian gồm $H_{grid} = 8$ hàng và $W_{grid} = 16$ cột ($N = 128$ tokens). Sau khi chiếu tuyến tính và cộng mã hóa vị trí, các token trên toàn bộ 5 khung hình được hợp nhất thông qua cơ chế Cross-Attention Không-Thời gian, sử dụng các token của khung hình trung tâm ($f = 3$) làm Query và toàn bộ token ghép nối của cả 5 frame làm Key và Value. Các token hợp nhất được xử lý qua 6 lớp Transformer Encoder Pre-LayerNorm. Để giải quyết xung đột giữa cách bố trí patch 2D dạng quét mành với tính đơn điệu 1D theo thời gian mà bộ giải mã CTC đòi hỏi, chúng tôi giới thiệu toán tử Spatial-to-Sequence Height Pooling (gộp trung bình theo chiều cao):
> $$\mathbf{s}_w = \frac{1}{H_{grid}} \sum_{h=1}^{H_{grid}} Z_{(h-1)W_{grid} + w}, \quad \forall w \in [1, W_{grid}] \quad \text{(Phương trình 2)}$$
> Chuỗi đặc trưng thu gọn $\mathbf{s} \in \mathbb{R}^{16 \times D}$ ánh xạ trực tiếp thành đúng 16 bước thời gian theo phương ngang, giúp thuật toán CTC căn chỉnh chuỗi mượt mà không bị sụp đổ.

💡 **Giải thích Công thức (2) - Height Pooling:**  
Đây là phát minh kỹ thuật quan trọng nhất của đề tài giúp cứu Flow 3: Lưới 2D gồm 8 hàng $\times$ 16 cột patch. Nếu đọc theo kiểu zíc-zắc từ trái sang phải, từ trên xuống dưới (raster-scan), nửa trên của chữ `B` sẽ cách nửa dưới của chữ `B` tới 16 bước thời gian, làm CTC bị "loạn nhịp" và sụp đổ accuracy về 0%. Bằng cách **cộng trung bình toàn bộ 8 patch theo chiều dọc ở mỗi cột** qua Công thức (2), ta thu về đúng 16 bước thời gian đơn điệu từ trái qua phải, mỗi bước đại diện trọn vẹn cho chiều cao của ký tự tại cột đó. Nhờ vậy CTC hội tụ trơn tru và đạt $75.78\%$!

---

### Đoạn 2.5: Hàm mất mát đa nhiệm & Kỹ thuật huấn luyện ổn định
> **🇬🇧 English:**  
> The entire system is trained using a multi-task objective that balances CTC recognition loss with super-resolution image reconstruction:
> \begin{equation}
> \mathcal{L}_{total} = \mathcal{L}_{CTC} + \lambda_{SR} \left(\mathcal{L}_1 + \lambda_{perc} \mathcal{L}_{perc}\right)
> \label{eq:loss_total}
> \end{equation}
> where $\lambda_{SR} = 0.5$ and $\lambda_{perc} = 0.01$. The perceptual loss $\mathcal{L}_{perc}$ is evaluated using features from a pre-trained VGG-19 network [11, 12]. To ensure numerical stability, the generator outputs are clamped and normalized from $[-1, 1]$ to $[0, 1]$ prior to standard ImageNet normalization, preventing feature distribution shifts. Furthermore, training follows a two-stage decoupled warmstart protocol where the SR generator is frozen during the initial 15 epochs, followed by joint fine-tuning with an isolated learning rate scale ($\eta_{SR} = 0.2 \times \eta_{OCR}$) and gradient clipping on the STN parameters.

> **🇻🇳 Dịch tiếng Việt:**  
> Toàn bộ hệ thống được huấn luyện bằng hàm mục tiêu đa nhiệm giúp cân bằng giữa hàm mất mát nhận dạng chuỗi CTC và hàm mất mát tái tạo ảnh siêu phân giải:
> $$\mathcal{L}_{total} = \mathcal{L}_{CTC} + \lambda_{SR} \left(\mathcal{L}_1 + \lambda_{perc} \mathcal{L}_{perc}\right) \quad \text{(Phương trình 3)}$$
> trong đó $\lambda_{SR} = 0.5$ và $\lambda_{perc} = 0.01$. Hàm mất mát tri giác $\mathcal{L}_{perc}$ được đánh giá bằng các đặc trưng trích xuất từ mạng VGG-19 tiền huấn luyện [11, 12]. Để đảm bảo sự ổn định số học, đầu ra của mạng sinh được kẹp giá trị (clamp) và chuẩn hóa từ dải $[-1, 1]$ về $[0, 1]$ trước khi đưa qua phép chuẩn hóa ImageNet, tránh làm lệch phân phối đặc trưng. Hơn nữa, quá trình huấn luyện tuân thủ giao thức khởi động tách biệt hai giai đoạn (Decoupled Warmstart): mạng SR được đóng băng trong 15 epoch đầu tiên, sau đó toàn bộ mạng được tinh chỉnh đồng thời với tỷ lệ learning rate được cô lập cho nhánh SR ($\eta_{SR} = 0.2 \times \eta_{OCR}$) và áp dụng kẹp gradient (clipping) cho các tham số của STN.

💡 **Giải thích Công thức (3) & Bài học thực tế:**  
Hàm loss kết hợp giữa CTC (để đọc đúng chữ) và $L_1$ + Perceptual Loss (để ảnh nét). Đoạn văn này đưa vào phát hiện kỹ thuật đắt giá: Nếu đưa thẳng ảnh $[-1, 1]$ vào VGG-19 (vốn quen nhận ảnh $[0, 1]$), gradient sẽ phát nổ làm phá vỡ toàn bộ mô hình. Việc chuẩn hóa về $[0, 1]$ và chia tỷ lệ learning rate cho SR bằng $0.2$ lần OCR giúp hệ thống học ổn định.

---

### Đoạn 2.6: Hậu xử lý xác suất tiên nghiệm Bayes theo luật Mercosur
> **🇬🇧 English:**  
> Finally, we apply a post-hoc Bayesian log-prior to enforce Mercosur positional syntax rules without retraining. Under the Brazilian Mercosur standard (three letters, one digit, one letter, two digits), the character at index 4 (0-indexed) is strictly restricted to the alphabetic set $\mathcal{C}_{alpha} = \{\text{`A'}-\text{`Z'}\}$. The model output log-probabilities are adjusted by:
> \begin{equation}
> \log \tilde{P}(c_t) = \log P_{model}(c_t) + \beta \log P_{prior}(c_t | \text{pos}(t))
> \label{eq:bayesian_prior}
> \end{equation}
> where $\beta = 1.0$ and $P_{prior}(c_t \in \mathcal{C}_{num} | \text{pos} = 4) = 0$, deterministically filtering out invalid digit predictions at that position.

> **🇻🇳 Dịch tiếng Việt:**  
> Cuối cùng, chúng tôi áp dụng bộ hiệu chuẩn tiên nghiệm Bayes dạng logarit để áp đặt các quy tắc ngữ pháp vị trí chuẩn Mercosur sau suy luận mà không cần huấn luyện lại. Theo quy chuẩn biển số Mercosur của Brazil (gồm 3 chữ cái, 1 chữ số, 1 chữ cái, 2 chữ số: `LLL-NLNN`), ký tự tại vị trí chỉ số 4 (đếm từ 0, tức vị trí thứ 5) bị giới hạn nghiêm ngặt chỉ được phép thuộc tập chữ cái $\mathcal{C}_{alpha} = \{\text{`A'}-\text{`Z'}\}$. Xác suất log đầu ra của mô hình được điều chỉnh theo:
> $$\log \tilde{P}(c_t) = \log P_{model}(c_t) + \beta \log P_{prior}(c_t \mid \text{pos}(t)) \quad \text{(Phương trình 4)}$$
> trong đó $\beta = 1.0$ và xác suất tiên nghiệm của mọi chữ số tại vị trí thứ 5 bị ép về 0: $P_{prior}(c_t \in \mathcal{C}_{num} \mid \text{pos} = 4) = 0$, giúp loại bỏ một cách tiền định mọi dự đoán chữ số sai lầm tại vị trí này.

💡 **Giải thích Công thức (4):**  
Theo luật biển số Mercosur, vị trí thứ 5 chắc chắn là **chữ cái**, không bao giờ là chữ số. Mô hình đôi khi phân vân giữa chữ cái `O` và số `0`, hay chữ `B` và số `8`. Bằng cách cộng thêm xác suất tiên nghiệm $\log P_{prior}$ (ép số bằng 0), mô hình tự động chọn đúng chữ cái mà không tốn công train lại.

---

## 📌 MỤC 3: KẾT QUẢ THỰC NGHIỆM & BÀN LUẬN (SECTION 3: RESULTS AND DISCUSSION)

### Đoạn 3.1: Thiết lập thực nghiệm (Experimental Setup)
> **🇬🇧 English:**  
> We evaluate our models on the official ICPR 2026 Challenge benchmark~\cite{icpr2026lrlpr}. Training was conducted on 19,001 tracks using dual NVIDIA Tesla T4 GPUs with the AdamW optimizer, batch size 64, and an initial learning rate of $3 \times 10^{-4}$. Evaluation was performed strictly on the 999 blind validation tracks of Scenario-B, where each track contains 5 video frames downsampled to $16 \times 64$ pixels. We report Exact Match (EM) accuracy, Character Error Rate (CER), Macro/Micro F1 scores, parameter counts, and 95th-percentile (P95) inference latency measured over 100 consecutive runs.

> **🇻🇳 Dịch tiếng Việt:**  
> Chúng tôi đánh giá các mô hình trên bộ chuẩn dữ liệu chính thức của cuộc thi ICPR 2026 Challenge [3]. Quá trình huấn luyện được thực hiện trên 19,001 vệt xe bằng cụm GPU kép NVIDIA Tesla T4 với bộ tối ưu hóa AdamW, kích thước batch 64 và tốc độ học ban đầu là $3 \times 10^{-4}$. Quá trình đánh giá được thực hiện nghiêm ngặt trên đúng 999 vệt video kiểm thử ẩn (blind validation) thuộc Scenario-B, trong đó mỗi vệt gồm 5 khung hình liên tiếp bị giảm độ phân giải xuống còn $16 \times 64$ điểm ảnh. Chúng tôi báo cáo các chỉ số: độ chính xác Đúng Tuyệt Đối (Exact Match - EM), Tỷ lệ Lỗi Ký tự (CER), điểm số F1 Macro/Micro, số lượng tham số mô hình, và độ trễ suy luận ở phân vị 95 (P95 Latency) được đo đạc qua 100 lần chạy liên tiếp.

---

### Bảng 1: Bảng kết quả thực nghiệm tổng thể (Master Benchmark)
*(Khớp 100% với Table 1 trong `main.tex`)*

| Luồng / Mô hình kiến trúc | Tham số (Params) | Độ trễ P95 (ms) | Exact Match (%) | CER (%) | Macro F1 (%) | Micro F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Flow 1: CRNN Baseline (+STN) [13] | 9.04 M | 6.30 ms | 76.28% | 6.56% | 85.12% | 86.85% |
| Flow 1: ResTran34 Baseline (+STN) | 31.08 M | 7.97 ms | 77.90% | 6.20% | 86.43% | 88.75% |
| Flow 1: ResTran34 (Không có STN) | 30.79 M | 7.51 ms | 75.80% | 7.12% | 84.90% | 86.20% |
| Flow 2: Train từ đầu không warmstart (Unstable) | 31.87 M | 8.48 ms | 59.16% | 11.90% | 81.30% | 83.10% |
| Flow 2: Huấn luyện tách biệt Stage 1 (15 epochs) | 35.20 M | 8.85 ms | 76.68% | 6.49% | 92.48% | 93.84% |
| **Flow 2: Huấn luyện kết hợp Joint End-to-End** | 32.10 M | 8.92 ms | **77.68%** | **5.81%** | **92.94%** | **94.43%** |
| **Flow 2: Tối ưu hóa Joint (Handoff v2.2)** | 32.10 M | 8.92 ms | **78.08%** | **5.72%** | **93.05%** | **94.52%** |
| **Multi-Flow Synergy Ensemble (Flow 2 + 3 + Luật)** | 46.50 M | 9.85 ms | **80.38%** | **5.22%** | **93.82%** | **94.78%** |
| Flow 3: Pure ViT (Quét mành bị sụp đổ CTC) | 5.85 M | 15.11 ms | 43.74% | 13.80% | 78.40% | 80.10% |
| **Flow 3: Pre-LN ViT + Height Pooling (Đề xuất)** | **14.40 M** | **6.82 ms** | **75.78%** | **6.16%** | **92.41%** | **94.04%** |

---

### Đoạn 3.2: Phân tích so sánh hiệu năng các luồng mô hình
> **🇬🇧 English:**  
> Table 1 presents the comparative results across all configurations. When training Flow 2 naively from scratch without decoupled stabilization, the model achieved only 59.16\% Exact Match due to gradient conflicts that damaged early RCAB filters. In contrast, our decoupled warmstart and joint fine-tuning enabled Flow 2 to achieve 78.08\% Exact Match and a CER of 5.72\%, outperforming the decoupled baseline by 1.40\% EM and reducing CER by 0.77\%. Qualitative inspection in Figure 1 confirms that the super-resolved frames generated by StackedSRNet recover sharp stroke boundaries and clear alphanumeric separations compared to blurred low-resolution inputs.

> **🇻🇳 Dịch tiếng Việt:**  
> Bảng 1 trình bày kết quả so sánh đối đầu trên tất cả các cấu hình. Khi huấn luyện Flow 2 từ đầu một cách đơn thuần mà không có cơ chế khởi động tách biệt (decoupled stabilization), mô hình chỉ đạt độ chính xác khiêm tốn 59.16% Exact Match do xung đột gradient làm hỏng các bộ lọc RCAB ban đầu. Ngược lại, chiến lược khởi động tách rời và tinh chỉnh kết hợp của chúng tôi đã giúp Flow 2 đạt độ chính xác 78.08% Exact Match và giảm CER xuống 5.72%, vượt qua mô hình cơ sở tách rời +1.40% EM và giảm 0.77% CER. Khảo sát trực quan ở Hình 1 khẳng định rằng các khung hình siêu phân giải do StackedSRNet tạo ra đã phục hồi được các đường biên nét sắc sảo và tách biệt rõ ràng các ký tự so với các khung ảnh đầu vào mờ nhòe.

---

### Bảng 2: Phân tích ma trận nhầm lẫn các cặp ký tự tương đồng
*(Khớp 100% với Table 2 trong `main.tex`)*

| Cặp ký tự dễ nhầm lẫn | Số lỗi mô hình Bicubic (Baseline) | Số lỗi mô hình Flow 2 (Ours) | Mức độ giảm lỗi (%) |
| :---: | :---: | :---: | :---: |
| **`8` $\leftrightarrow$ `B`** | 19 lỗi | **0 lỗi** | **-100.0% (Triệt tiêu hoàn toàn)** |
| **`0` $\leftrightarrow$ `D`** | 24 lỗi | 7 lỗi | **-70.8%** |
| **`2` $\leftrightarrow$ `7`** | 14 lỗi | 4 lỗi | **-71.4%** |
| **`5` $\leftrightarrow$ `S`** | 11 lỗi | 3 lỗi | **-72.7%** |

---

### Đoạn 3.3: Phân tích sâu ma trận nhầm lẫn ký tự khó ($RQ_2$)
> **🇬🇧 English:**  
> A closer examination of the character confusion matrix in Table 2 reveals the exact source of performance gains. In the bicubic baseline, the digit `8' and letter `B' were confused 19 times because low-resolution blur filled in the central inner loops. Our Frame-Parallel StackedSRNet reconstructs character curvatures and inner boundaries, completely eliminating `8' versus `B' misclassifications down to 0 errors. Similar reductions exceeding 70\% were observed for `0' versus `D' (from 24 to 7 errors) and `2' versus `7' (from 14 to 4 errors).

> **🇻🇳 Dịch tiếng Việt:**  
> Xem xét kỹ lưỡng ma trận nhầm lẫn ký tự ở Bảng 2 đã làm sáng tỏ nguồn gốc trực tiếp của sự cải thiện hiệu năng. Ở mô hình cơ sở dùng nội suy bicubic, chữ số `8` và chữ cái `B` bị nhận dạng nhầm lẫn 19 lần vì hiện tượng mờ nhòe ở độ phân giải thấp đã lấp đầy các vòng cung khép kín bên trong. Mạng Frame-Parallel StackedSRNet của chúng tôi đã tái cấu trúc thành công độ cong của ký tự và ranh giới nét bên trong, triệt tiêu hoàn toàn các lỗi nhầm giữa `8` và `B` về đúng 0 lỗi. Các mức độ giảm lỗi tương tự vượt trên 70% cũng được ghi nhận đối với cặp `0` và `D` (từ 24 xuống còn 7 lỗi) và cặp `2` và `7` (từ 14 xuống còn 4 lỗi).

💡 **Điểm sáng học thuật cực mạnh:**  
Đây là con số thực nghiệm thuyết phục nhất trong bài: Bạn chứng minh được Siêu phân giải không phải để "làm đẹp mắt người", mà nó thực sự tách được nét eo giữa của số 8 và nét thẳng lưng của chữ B, đưa số lần nhầm lẫn từ 19 lần về đúng 0 lần!

---

### Đoạn 3.4: Hiệu năng của Flow 3 & Đóng góp của Height Pooling ($RQ_3$)
> **🇬🇧 English:**  
> For Flow 3, directly applying a raster-scan serialization to 2D patch tokens caused CTC divergence, resulting in a low accuracy of 43.74\% and 13.80\% CER. Introducing our Spatial-to-Sequence Height Pooling operator resolved this geometric misalignment, raising Flow 3 performance to 75.78\% Exact Match and 6.16\% CER. Notably, Flow 3 requires only 14.4M parameters, representing a 55.1\% reduction compared to Flow 2 (32.1M parameters), while running with an inference latency of 6.82 ms (23.5\% faster).

> **🇻🇳 Dịch tiếng Việt:**  
> Đối với Flow 3, việc áp dụng trực tiếp phép tuần tự hóa theo thứ tự quét mành (raster-scan) lên các token patch 2D đã gây ra hiện tượng phân kỳ hàm mất mát CTC, dẫn đến độ chính xác sụt giảm xuống mức 43.74% và CER lên tới 13.80%. Việc đưa vào toán tử Spatial-to-Sequence Height Pooling đã giải quyết triệt để sự lệch pha hình học này, nâng vọt hiệu năng của Flow 3 lên 75.78% Exact Match và hạ CER xuống 6.16%. Đáng chú ý, Flow 3 chỉ cần 14.4 triệu tham số, tương đương mức giảm 55.1% kích thước so với Flow 2 (32.1 triệu tham số), đồng thời vận hành với độ trễ suy luận chỉ 6.82 ms (nhanh hơn 23.5%).

---

### Đoạn 3.5: Đỉnh cao kết quả Ensemble $80.38\%$ & Triệt tiêu thành phần (Ablation)
> **🇬🇧 English:**  
> Combining the log-probabilities of Flow 2 and Flow 3 in a Multi-Flow Synergy Ensemble along with the Mercosur positional syntax prior establishes our peak validation performance of \textbf{80.38\% Exact Match} (803 correctly transcribed tracks out of 999) and reduces CER to \textbf{5.22\%}. Ablation on spatial transformation shows that removing the STN module decreased ResTran34 accuracy by 2.10\% (from 77.90\% to 75.80\%), confirming the necessity of perspective rectification for unconstrained traffic camera angles.

> **🇻🇳 Dịch tiếng Việt:**  
> Việc kết hợp xác suất log của Flow 2 và Flow 3 trong mô hình Multi-Flow Synergy Ensemble cùng với phân phối tiên nghiệm ngữ pháp vị trí Mercosur đã xác lập đỉnh cao hiệu năng thực nghiệm của chúng tôi là **80.38% Exact Match** (nhận diện chính xác 803 trên 999 vệt xe) và giảm tỷ lệ lỗi ký tự CER xuống còn **5.22%**. Nghiên cứu triệt tiêu thành phần (Ablation) đối với phép biến đổi không gian cho thấy việc loại bỏ khối STN làm sụt giảm độ chính xác của ResTran34 đi 2.10% (từ 77.90% xuống 75.80%), khẳng định tính cấp thiết của việc nắn chỉnh góc phối cảnh đối với các góc quay tự do của camera giao thông.

---

### Đoạn 3.6: Ba bài học kỹ thuật thực tế (Engineering Insights)
> **🇬🇧 English:**  
> During system development, three key technical insights were established. First, computing perceptual loss requires precise dynamic range handling; passing raw $[-1, 1]$ tensors directly into pre-trained VGG-19 networks without scaling to $[0, 1]$ introduces severe gradient distortions that hinder generator training. Second, when fine-tuning across refactored architectures using non-strict checkpoint loading, automated parameter key remapping is vital to prevent accidental random weight re-initializations. Third, height-wise token pooling provides a principled bridge connecting 2D vision transformer representations to 1D CTC sequence decoders.

> **🇻🇳 Dịch tiếng Việt:**  
> Trong quá trình phát triển hệ thống, ba phát hiện kỹ thuật then chốt đã được đúc kết.  
> - **Thứ nhất**, việc tính toán hàm mất mát tri giác (Perceptual Loss) đòi hỏi phải xử lý dải động giá trị một cách chuẩn xác; việc nạp trực tiếp các tensor ở dải $[-1, 1]$ vào mạng VGG-19 tiền huấn luyện mà không co giãn về $[0, 1]$ sẽ tạo ra các biến dạng gradient nghiêm trọng cản trở quá trình hội tụ của mạng sinh.  
> - **Thứ hai**, khi tinh chỉnh giữa các kiến trúc được tái cấu trúc bằng cờ tải checkpoint không nghiêm ngặt (`strict=False`), cơ chế ánh xạ lại tên khóa tham số (key remapping) tự động là tối quan trọng nhằm ngăn chặn việc các khối trọng số cốt lõi bị khởi tạo ngẫu nhiên ngoài ý muốn.  
> - **Thứ ba**, cơ chế gộp token theo chiều cao (Height Pooling) tạo ra một cây cầu nối toán học chuẩn mực kết nối các biểu diễn patch 2D của Vision Transformer với các bộ giải mã chuỗi 1D CTC.

💡 **Ý nghĩa:** Đây chính là đoạn văn ghi điểm cao nhất với các Thầy và Reviewer: Bạn không giấu giếm các khó khăn kỹ thuật mà đúc kết thành các nguyên tắc kỹ thuật chuẩn xác, chứng minh tính trung thực và chiều sâu thực nghiệm.

---

## 📌 MỤC 4: KẾT LUẬN & HƯỚNG PHÁT TRIỂN (SECTION 4: CONCLUSION)

### 🇬🇧 Bản gốc tiếng Anh:
> In this paper, we presented an empirical study of multi-frame super-resolution and vision transformer architectures for low-resolution license plate recognition under severe video degradation. Our proposed frame-parallel StackedSRNet eliminates inter-frame ghosting artifacts and completely resolves confusable alphanumeric pairs, reducing `8' versus `B' classification errors to zero and achieving 78.08\% Exact Match accuracy, which increases to 80.38\% when synergized in a multi-flow ensemble with Mercosur syntax calibration. Furthermore, our Spatial-to-Sequence Height Pooling mechanism enables compact Vision Transformers to achieve 75.78\% Exact Match accuracy with a 55.1\% parameter reduction and an inference latency of 6.82 ms. Future work will investigate 8-bit integer quantization and unsupervised adaptation across multi-national plate formats.

### 🇻🇳 Bản dịch tiếng Việt chuẩn xác:
> Trong bài báo này, chúng tôi đã trình bày một nghiên cứu thực nghiệm chuyên sâu về các kiến trúc siêu phân giải đa khung hình và vision transformer cho bài toán nhận diện biển số xe độ phân giải thấp dưới sự suy thoái video nghiêm trọng. Mạng Frame-Parallel StackedSRNet đề xuất của chúng tôi đã loại bỏ hoàn toàn các hiện tượng bóng ma giữa các khung hình và giải quyết triệt để các cặp ký tự dễ gây nhầm lẫn, đưa số lỗi phân loại giữa `8` và `B` về mức 0 và đạt độ chính xác 78.08% Exact Match, và đạt mốc 80.38% khi hiệp đồng trong mô hình ensemble đa luồng kết hợp với hiệu chuẩn ngữ pháp Mercosur. Hơn nữa, cơ chế Spatial-to-Sequence Height Pooling của chúng tôi giúp mạng Vision Transformer tinh gọn đạt được độ chính xác 75.78% Exact Match với việc cắt giảm 55.1% số lượng tham số và đạt độ trễ suy luận chỉ 6.82 ms. Hướng phát triển tương lai sẽ tập trung nghiên cứu kỹ thuật lượng tử hóa số nguyên 8-bit (INT8) và cơ chế thích nghi không giám sát cho các định dạng biển số xe đa quốc gia.

---

## 📚 TÀI LIỆU THAM KHẢO CHUẨN XÁC (REFERENCES)

Toàn bộ 13 bài báo dưới đây đều được trích dẫn tuần tự và là các công trình kinh điển:

1. **[1] Laroca et al. (2021)**: *"An efficient and layout-independent automatic license plate recognition system based on the YOLO detector"*, **IET Intelligent Transport Systems**.
2. **[2] Gonçalves et al. (2018)**: *"Real-time license plate detection and recognition using deep convolutional neural networks"*, **Journal of Electronic Imaging**.
3. **[3] ICPR Organizers (2026)**: *"ICPR 2026 Challenge on Low-Resolution License Plate Recognition in Unconstrained Traffic Surveillance"*.
4. **[4] Zhang et al. (2018)**: *"Image super-resolution using very deep residual channel attention networks (RCAN)"*, **ECCV 2018** *(Phát minh khối RCAB)*.
5. **[5] Shi et al. (2016)**: *"Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network"*, **CVPR 2016** *(Phát minh tầng PixelShuffle)*.
6. **[6] Graves et al. (2006)**: *"Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks"*, **ICML 2006** *(Phát minh hàm CTC Loss)*.
7. **[7] Dosovitskiy et al. (2021)**: *"An image is worth 16x16 words: Transformers for image recognition at scale"*, **ICLR 2021** *(Bài báo gốc mạng Vision Transformer - ViT)*.
8. **[8] Jaderberg et al. (2015)**: *"Spatial transformer networks"*, **NeurIPS 2015** *(Phát minh mạng STN căn chỉnh không gian)*.
9. **[9] He et al. (2016)**: *"Deep residual learning for image recognition"*, **CVPR 2016** *(Mạng ResNet-34)*.
10. **[10] Vaswani et al. (2017)**: *"Attention is all you need"*, **NeurIPS 2017** *(Khai sinh cơ chế Self-Attention & Transformer)*.
11. **[11] Simonyan & Zisserman (2015)**: *"Very deep convolutional networks for large-scale image recognition"*, **ICLR 2015** *(Mạng VGG-19)*.
12. **[12] Johnson et al. (2016)**: *"Perceptual losses for real-time style transfer and super-resolution"*, **ECCV 2016** *(Phát minh hàm Perceptual Loss)*.
13. **[13] Shi, Bai & Yao (2016)**: *"An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition"*, **IEEE TPAMI** *(Mạng CRNN Baseline)*.
