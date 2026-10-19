# Báo Cáo Tiến Độ Huấn Luyện (Training Progress Report)

Tài liệu này là file báo cáo **duy nhất** theo dõi liên tục tiến độ các tiến trình huấn luyện (training) đang chạy ngầm trong hệ thống. Mỗi lần kiểm tra (khi người dùng gõ `report` hoặc `report tiến độ`), một dòng mới sẽ được cập nhật tiếp nối vào bảng bên dưới.

---

### Bảng Theo Dõi Tiến Độ Huấn Luyện

| STT | Thời điểm kiểm tra | Tên luồng training | Tiến độ (%) | Chi tiết trạng thái (PID / Thời gian chạy / Giới hạn) |
| :---: | :---: | :--- | :---: | :--- |
| 1 | 2026-10-16 13:48:30 | `train_decoupled.py` (Model: `restran`) | **25.0%** | PID: `51798` \| Elapsed: ~10m / 39m cap (0.65h) \| Checkpoint: `decoupled_restran_last.pth` (Epoch 1 xong, đang chạy Epoch 2) |
| 2 | 2026-10-16 15:41:08 | `train_decoupled.py` (Model: `restran`) | **100.0%** (Hoàn thành) | PID `51798` đã kết thúc lúc 14:09 (chạy 30.09m). Hoàn tất đủ 15/15 epochs (5 frozen + 10 joint). Kết quả: **Best Acc: 74.27%**, **Best CER: 7.18%** (Checkpoint: `results/decoupled_restran_best.pth`). Hiện không có tiến trình ngầm nào khác đang chạy. |
| 3 | 2026-10-16 19:55:00 | `train_decoupled.py` (Model: `restran` - Joint Stage 2) | **9.4%** | PID: `57576` \| Chạy ~2m 15s / 24m cap (0.4h) \| Warmstart từ `decoupled_restran_best.pth` (0 frozen, 10 joint epochs), đang chạy Epoch 01/10. |
| 4 | 2026-10-16 20:11:42 | `train_decoupled.py` (Model: `restran` - Joint Stage 2) | **85.0%** | PID: `57576` \| Chạy ~19m 20s / 24m cap (0.4h) \| Đang chạy Epoch 09/10. Vừa phá kỷ lục tại Epoch 08: **Val Acc: 75.08%**, **CER: 6.99%** (lưu đè checkpoint best). |
| 5 | 2026-10-16 20:19:32 | `train_decoupled.py` (Model: `sr_vit` - Flow 3) | **33.0%** | PID `57576` (`restran`) đã hoàn tất 100% lúc 20:15 với kỷ lục mới **Best Acc: 75.38%**, **CER: 6.79%** (10/10 epochs). Luồng mới PID `58192` (`sr_vit`) bắt đầu lúc 20:16 (chạy ~3.5m / 24m cap), hiện đang ở Epoch 04/10 [Stage 1: Frozen SR]. |
| 6 | 2026-10-16 21:08:58 | `train_decoupled.py` (Model: `restran` - Push 80%+) | **5.0%** | PID `58192` (`sr_vit`) đã hoàn tất 10/10 epochs lúc 20:31. Khởi động chiến dịch mới PID `59444` lúc 21:06 (15 joint epochs, 8192 samples/ep, batch 48, margin loss 4.0, cap 1.1h). Đang chạy Epoch 01/15 (batch >73%). |
| 7 | 2026-10-16 22:11:51 | `train_decoupled.py` (Model: `restran` - Push 80%+) | **100.0%** (Hoàn thành) | PID `59444` đã hoàn tất trọn vẹn đủ 15/15 epochs lúc 22:05 (chạy 58.63m). Thiết lập đỉnh cao mới: **Best Exact Match: 75.58%**, **CER: 6.81%**, **Edit Acc: 93.19%** (Checkpoint: `results/decoupled_restran_best.pth`). Hiện không còn tiến trình ngầm nào đang chạy. |
| 8 | 2026-10-17 01:41:39 | `train_decoupled.py` (Model: `restran` - Overnight Full Train) | **18.5%** | PID: `61973` (bọc `caffeinate`). Chạy Full Dataset (`samples=0`, 792 batches/ep, batch 48, cap 5.5h). Đã chạy 57m12s (~17.3% time cap), đang ở Epoch 03/15 (batch 675/792 ~85%). Val Acc Ep1: 74.57%, Ep2: 74.27%. |
| 9 | 2026-10-17 05:40:14 | `train_decoupled.py` (Model: `restran` - Overnight Full Train) | **100.0%** (Hoàn thành) | Hoàn tất trọn vẹn đủ 15/15 epochs trên 100% Full Dataset (38,002 tracks/epoch, batch 48). Đạt đỉnh cao kỷ lục mới: **Exact Match: 76.68%**, **CER: 6.49%**, **Normalized Edit Acc: 93.51%**, **Macro F1: 92.48%**, **Micro F1: 93.84%** (Checkpoint: `results/decoupled_restran_best.pth`). Đánh giá toàn diện và lưu log tại `results/run_overnight_full_train.log`. Hiện không còn tiến trình ngầm nào đang chạy. |
| 10 | 2026-10-18 18:08:25 | `train.py` (Model: `sr_restran_flow2`) | **13.0%** | PID: `81533` (4 workers fork). Huấn luyện Flow 2 chuẩn 30 epochs (`batch_size=64`, 594 batches/ep, lr 3e-4). Đã chạy 1h46m. Đang ở Epoch 04/30 (batch 500/594 ~84%). Checkpoint Best: `results/sr_restran_flow2_best.pth` (Val Acc: 65.67%). Log: `results/train_flow2_local.log`. |
| 11 | 2026-10-18 19:03:52 | `train.py` (Model: `sr_restran_flow2`) | **14.0%** | PID: `81533` (4 workers fork). Đã chạy 2h41m. Vừa hoàn thành Epoch 04/30 (Val Acc: 61.46%), hiện đang ở Epoch 05/30 (batch 62/594 ~10%). Best Val Acc hiện tại: **65.67%** (Checkpoint: `results/sr_restran_flow2_best.pth`). Log: `results/train_flow2_local.log`. |
| 12 | 2026-10-18 21:01:17 | `train.py` (Model: `sr_restran_flow2`) | **20.0%** | PID: `81533` (4 workers fork). Đã chạy 4h38m. Hoàn tất 5/30 epochs (Epoch 5 Val Acc: 57.26%), hiện đang chạy Epoch 06/30 (batch 507/594 ~85%). Checkpoint Best: `results/sr_restran_flow2_best.pth` (65.67%). Log: `results/train_flow2_local.log`. |
| 13 | 2026-10-19 08:36:17 | `train.py` (Model: `sr_restran_flow2`) | **95.0%** | PID: `81533` (4 workers fork). Chạy qua đêm bền bỉ **16h14m**. Đã hoàn tất 28/30 epochs, hiện đang chạy Epoch 29/30 (batch 284/594 ~48%). **Thiết lập đỉnh cao mới: Val Acc: 69.07%** tại Epoch 27 (Checkpoint: `results/sr_restran_flow2_best.pth`). Log: `results/train_flow2_local.log`. |
| 14 | 2026-10-19 09:42:24 | `train.py` (Model: `sr_restran_flow2`) | **100.0%** (Hoàn thành) | PID `81533` đã hoàn tất trọn vẹn đủ 30/30 epochs lúc 09:38 (chạy 17h16m). Đạt đỉnh cao tại Epoch 30: **Best Val Acc: 69.87%** (Checkpoint: `results/sr_restran_flow2_best.pth`). Đã lưu kết quả suy luận tại `results/submission_sr_restran_flow2.txt`. Hiện không còn tiến trình ngầm nào đang chạy. |
| 15 | 2026-10-19 11:42:27 | `train.py` (Model: `sr_restran_flow2` - Warmstart 76.68%) | **23.0%** | PID: `2722` (4 workers fork). Bắt đầu lúc 10:19 (đã chạy 1h24m). Huấn luyện tinh chỉnh 20 epochs trên Full Dataset (`batch_size=64`, lr 1.5e-4) warmstart từ checkpoint 76.68%. Đã xong 4 epochs, hiện đang ở Epoch 05/20 (batch 375/594 ~63%). Log: `results/train_flow2_warmstart76.log`. |
| 16 | 2026-10-19 14:46:36 | `train.py` (Model: `sr_restran_flow2` - Warmstart 76.68%) | **76.0%** | PID: `2722` (4 workers fork). Đã chạy **4h28m**. Hoàn tất 15/20 epochs, hiện đang chạy Epoch 16/20 (batch 56/594 ~9%). Vừa vượt mốc 70%: **Val Acc đạt 70.07%** tại Epoch 15 (Train Loss: 0.1880). Log: `results/train_flow2_warmstart76.log`. |
| 18 | 2026-10-19 16:13:50 | `train.py` (Model: `sr_restran_flow2` - Warmstart 76.68%) | **100.0%** (Hoàn thành) | PID: `2722` đã hoàn tất trọn vẹn đủ 20/20 epochs (chạy 5h54m). Val Acc đạt đỉnh 70.57% ở Epoch 16, kết thúc Epoch 20 đạt 69.07% (Train Loss: 0.1709). Best Checkpoint bảo toàn mốc **76.68% Exact Match**, **CER 6.49%**, **Normalized Edit Acc 93.51%** (`results/sr_restran_flow2_best.pth`). Log: `results/train_flow2_warmstart76.log`. |
| 19 | 2026-10-19 16:28:12 | `train.py` (Model: `sr_vit` - Flow 3 ViT) | **5.0%** | PID: `3482` (4 workers fork). Bắt đầu lúc 16:28. Huấn luyện 40 epochs theo Handoff Spec (`batch_size=64`, lr 3e-4) warmstart SR từ `results/sr_net_best.pth`. Đã hoàn tất Epoch 1 (Train Loss: 3.3553), đang chạy Epoch 2/40. Log: `results/train_flow3_local.log`. |

---















