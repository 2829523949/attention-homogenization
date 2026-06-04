"""验证 Phase 2 数据集"""
import sys, os
sys.path.insert(0, "/root/Qwen3-VL")
from train_edit import EditDataset

d = EditDataset()
i, t, r = d.get_batch(1)
print(f"Image: {i[0]}")
print(f"Text: {t[0]}")
print(f"Target raw_6d: {r[0,:2].tolist()}")
print(f"File exists: {os.path.exists(i[0])}")
print(f"Train pairs: {len(d.train_pairs)}")
print(f"Val pairs: {len(d.val_pairs)}")

# 检查一组不匹配对: 确保输入图和文本不同视角
img_path = d.train_pairs[0]['image']
text = d.train_pairs[0]['text']
target = d.train_pairs[0]['target_raw_6d']
print(f"\n=== 样本复查 ===")
print(f"图片: {img_path}")
print(f"文本: '{text}'")
print(f"目标位姿: yaw={target[0]:+.2f}, pitch={target[1]:+.2f}")

# 从文件名提取图片实际yaw, 确认不匹配
import re
yaw_match = re.search(r'y([+-]\d+\.\d+)', img_path)
if yaw_match:
    img_yaw = float(yaw_match.group(1))
    print(f"图片真实yaw: {img_yaw:.2f}")
    print(f"文本目标yaw: {target[0]:.2f}")
    print(f"是否不匹配: {'✅' if abs(img_yaw - target[0]) > 0.1 else '❌ 相同角度!' }")
