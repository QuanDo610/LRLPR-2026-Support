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

---







