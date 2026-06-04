"""检查数据集样本"""
import sys
sys.path.insert(0, "/root/Qwen3-VL")
from train_edit import EditDataset

ds = EditDataset()
for i in range(0, min(30, len(ds.train_pairs))):
    item = ds.train_pairs[i]
    text = item["text"]
    target = item["target_raw_6d"]
    yaw, pitch = target[0], target[1]
    is_template = True if "平视" in text or "仰视" in text or "俯视" in text else False
    print(f"[{'T' if is_template else 'D'}] {text:30s} yaw={yaw:+.2f} pitch={pitch:+.2f}")
