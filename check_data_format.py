"""检查 EditDataset 数据格式"""
import sys
sys.path.insert(0, "/root/Qwen3-VL")
from train_edit import EditDataset

ds = EditDataset()
print(f"type: {type(ds.train_pairs)}")
print(f"len: {len(ds.train_pairs)}")
first = ds.train_pairs[0]
print(f"first type: {type(first)}")
if isinstance(first, tuple):
    print(f"first len: {len(first)}")
    print(f"first[0]: {first[0][:50]}")
    print(f"first[1].shape: {first[1].shape}")
elif isinstance(first, dict):
    print(f"keys: {first.keys()}")
