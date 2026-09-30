#!/bin/bash
set -e

echo "🚀 Chuẩn bị Kaggle Kernel cho Flow 2 (ResTran) Two-Stage Joint Training..."

mkdir -p kaggle_push_kernel
cd kaggle_push_kernel

cat << 'JSON_EOF' > kernel-metadata.json
{
  "id": "dotrongminhquan/lrlpr-flow-2-joint-restran-v12",
  "title": "lrlpr-flow-2-joint-restran-v12",
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
import subprocess
import glob

print("📦 Dò tìm mã nguồn từ dataset...")
codebase_paths = glob.glob('/kaggle/input/**/train_decoupled.py', recursive=True)
if codebase_paths:
    codebase_dir = os.path.dirname(codebase_paths[0])
    os.system(f"cp -r {codebase_dir}/* /kaggle/working/")
else:
    os.system("unzip -o -q /kaggle/input/*/codebase.zip -d /kaggle/working/")

os.system("mkdir -p /kaggle/working/data")

kaggle_train_paths = glob.glob('/kaggle/input/**/train', recursive=True)
if kaggle_train_paths:
    train_dir = kaggle_train_paths[0]
    os.system(f"ln -sfn {train_dir} /kaggle/working/data/train")

kaggle_test_paths = glob.glob('/kaggle/input/**/public_test', recursive=True) or glob.glob('/kaggle/input/**/*test*', recursive=True)
if kaggle_test_paths:
    test_dir = kaggle_test_paths[0]
    os.system(f"ln -sfn {test_dir} /kaggle/working/data/public_test")

print("🚀 Khởi chạy quá trình Two-Stage Training (Flow 2 - ResTran)...")
os.chdir("/kaggle/working")
# Đảm bảo cấu hình config.py trỏ về "restran"
fix_config_cmd = """
with open('configs/config.py', 'r') as f:
    c = f.read()
c = c.replace('MODEL_TYPE: str = "sr_vit"', 'MODEL_TYPE: str = "restran"')
with open('configs/config.py', 'w') as f:
    f.write(c)
"""
subprocess.run(["python", "-c", fix_config_cmd])

os.system("pip install albumentations")
os.system("python train_decoupled.py --model restran")
PY_EOF

echo "☁️ Đang push Kaggle Kernel v12..."
yes | ../.venv/bin/kaggle kernels push -p .

echo "🎉 Xong! Kernel Flow 2 đã được đẩy lên Kaggle."
echo "🔗 Link: https://www.kaggle.com/dotrongminhquan/lrlpr-flow-2-joint-restran-v12"
