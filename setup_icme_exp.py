"""Phase 1+2 实验代码隔离"""
import os, shutil, subprocess

# 1. 创建实验目录
exp_dir = "/root/Qwen3-VL/icme_exp"
os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)

# 2. 复制代码文件
files = ["bridge.py", "train_edit.py", "prepare_edit_dataset.py", 
         "prepare_wplus.py", "generate_text_variants.py",
         "test_closed_loop.py", "test_wplus_final.py", "test_real_pipeline.py",
         "invert_real_photo.py", "test_real_demo.py"]

for f in files:
    src = f"/root/Qwen3-VL/{f}"
    if os.path.exists(src):
        shutil.copy2(src, f"{exp_dir}/{f}")
        print(f"  cp {f}")

# 3. 复制权重
for f in os.listdir("/root/Qwen3-VL/checkpoints"):
    if f.startswith("mix_epoch_") or f.startswith("text_only_epoch_"):
        shutil.copy2(f"/root/Qwen3-VL/checkpoints/{f}", f"{exp_dir}/checkpoints/{f}")
        print(f"  cp checkpoints/{f}")

# 4. 复制数据集
data_dir = "/root/icme_exp_data"
if not os.path.exists(data_dir):
    shutil.copytree("/root/edit_data", data_dir, symlinks=True)
    print(f"  cp edit_data → {data_dir}")

print(f"\n实验代码: {exp_dir}")
print(f"实验数据: {data_dir}")
