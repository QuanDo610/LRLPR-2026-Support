#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================================================="
echo "🚀 Chuẩn bị và Đẩy Kaggle Kernel: Flow 2 Joint Training +10 Epochs (v15)"
echo "=========================================================================="

# 1. Sync updated codebase and best checkpoint to Kaggle Dataset
echo "📦 Bước 1: Đồng bộ Codebase & Checkpoint 77.68% lên Kaggle Dataset..."
bash scripts/kaggle_sync.sh

# 2. Chuẩn bị Kernel Metadata & Template
echo "📦 Bước 2: Tạo thư mục cấu hình kernel kaggle_push_kernel..."
mkdir -p kaggle_push_kernel
cd kaggle_push_kernel

cat << 'JSON_EOF' > kernel-metadata.json
{
  "id": "dotrongminhquan/lrlpr-flow-2-kaggle-plus10",
  "title": "lrlpr-flow-2-kaggle-plus10",
  "code_file": "kaggle_kernel_template.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "dataset_sources": [
    "qunnguynm/lrlpr-2026",
    "dotrongminhquan/lrlpr-codebase"
  ],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
JSON_EOF

cat << 'PY_EOF' > kaggle_kernel_template.py
import os
import sys
import subprocess
import glob
import time

# Chống nuốt log trên Kaggle
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

print("=" * 70)
print("🚀 KAGGLE GPU: FLOW 2 (SR-RESTRAN) FINE-TUNING +10 EPOCHS")
print("=" * 70)

print("\n📦 1. Khôi phục mã nguồn từ dataset...")
zip_paths = glob.glob('/kaggle/input/**/codebase.zip', recursive=True)
if zip_paths:
    print(f"   Unzipping {zip_paths[0]} to /kaggle/working/...")
    os.system(f"unzip -o -q {zip_paths[0]} -d /kaggle/working/")
else:
    print("❌ Không tìm thấy codebase.zip! Thử tìm train.py...")
    codebase_paths = glob.glob('/kaggle/input/**/train.py', recursive=True)
    if codebase_paths:
        codebase_dir = os.path.dirname(codebase_paths[0])
        print(f"   Copy từ: {codebase_dir}")
        os.system(f"cp -r {codebase_dir}/* /kaggle/working/")
    else:
        print("❌ LỖI NGHIÊM TRỌNG: Không tìm thấy mã nguồn!")
        sys.exit(1)

os.system("mkdir -p /kaggle/working/data /kaggle/working/results")

print("\n🔗 2. Liên kết dữ liệu huấn luyện và kiểm định...")
kaggle_train_paths = glob.glob('/kaggle/input/**/train', recursive=True)
if kaggle_train_paths:
    train_dir = kaggle_train_paths[0]
    os.system(f"ln -sfn {train_dir} /kaggle/working/data/train")
    print(f"   ✅ Linked train data: {train_dir} -> /kaggle/working/data/train")
else:
    print("❌ LỖI: Không tìm thấy thư mục train trong /kaggle/input!")
    sys.exit(1)

kaggle_test_paths = glob.glob('/kaggle/input/**/public_test', recursive=True) or glob.glob('/kaggle/input/**/*test*', recursive=True)
if kaggle_test_paths:
    test_dir = kaggle_test_paths[0]
    os.system(f"ln -sfn {test_dir} /kaggle/working/data/public_test")
    print(f"   ✅ Linked test data: {test_dir} -> /kaggle/working/data/public_test")

val_track_paths = glob.glob('/kaggle/input/**/val_tracks.json', recursive=True)
if val_track_paths:
    os.system(f"cp {val_track_paths[0]} /kaggle/working/data/val_tracks.json")
    print(f"   ✅ Copied val_tracks.json: {val_track_paths[0]}")

# Tìm checkpoint 77.68%
ckpt_paths = glob.glob('/kaggle/input/**/sr_restran_flow2_plus10_best.pth', recursive=True)
target_ckpt = "/kaggle/working/results/sr_restran_flow2_plus10_best.pth"
if ckpt_paths:
    os.system(f"cp {ckpt_paths[0]} {target_ckpt}")
    print(f"   ✅ Checkpoint 77.68% copied from: {ckpt_paths[0]}")
elif os.path.exists(target_ckpt):
    print(f"   ✅ Checkpoint 77.68% already present at {target_ckpt}")
else:
    print("⚠️ Không tìm thấy checkpoint sr_restran_flow2_plus10_best.pth! Quét thư mục results...")
    os.system("find /kaggle/input -name '*.pth' -exec ls -lh {} \;")

print("\n📦 3. Cài đặt thư viện phụ thuộc...")
os.system("pip install -q albumentations")

print("\n🔥 4. BẮT ĐẦU HUẤN LUYỆN FLOW 2 (10 EPOCHS GPU): train.py -m sr_restran...")
os.chdir("/kaggle/working")

train_cmd = [
    "python", "-u", "train.py",
    "-m", "sr_restran",
    "--checkpoint", target_ckpt,
    "--experiment-name", "sr_restran_flow2_kaggle_plus10",
    "--epochs", "10",
    "--lr", "1.0e-4",
    "--ocr-lr-scale", "0.8",
    "--batch-size", "64",
    "--num-workers", "2",
    "--max-hours", "2.0",
    "--require-gpu"
]

print("Lệnh chạy:", " ".join(train_cmd))
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
ret = subprocess.run(train_cmd, env=env)

if ret.returncode != 0:
    print("❌ Training failed with exit code:", ret.returncode)
    sys.exit(ret.returncode)

print("\n" + "=" * 70)
print("📊 5. ĐÁNH GIÁ MÔ HÌNH VỚI evaluate.py...")
print("=" * 70)
eval_cmd = [
    "python", "-u", "evaluate.py",
    "-m", "sr_restran",
    "--weights", "results/sr_restran_flow2_kaggle_plus10_best.pth",
    "--data-dir", "data/train",
    "--batch-size", "32",
    "--output-json", "results/eval_flow2_kaggle_plus10_best.json"
]
print("Lệnh đánh giá:", " ".join(eval_cmd))
subprocess.run(eval_cmd, env=env)

print("\n🎉 HOÀN THÀNH HUẤN LUYỆN VÀ ĐÁNH GIÁ FLOW 2 TRÊN KAGGLE!")
os.system("ls -lh results/")
PY_EOF

echo "☁️ Đang push Kaggle Kernel..."
yes | "$PROJECT_ROOT/.venv/bin/kaggle" kernels push -p .

echo "🎉 Xong! Kernel Flow 2 đã được đẩy lên Kaggle."
echo "🔗 Link: https://www.kaggle.com/code/dotrongminhquan/lrlpr-flow-2-kaggle-plus10"
