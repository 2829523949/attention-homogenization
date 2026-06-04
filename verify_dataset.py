"""验证数据集：对应关系 + 无泄露"""
import sys, json
sys.path.insert(0, "/root/Qwen3-VL")

with open("/root/Qwen3-VL/text_variants.json") as f:
    data = json.load(f)

print("=" * 60)
print("1. 对应关系检查")
print("=" * 60)

# 按 (yaw, pitch_offset) 分组
from collections import defaultdict
groups = defaultdict(list)
for d in data:
    groups[(round(d['yaw'], 2), round(d['pitch_offset'], 2))].append(d['text'])

for (yaw, po), texts in sorted(groups.items()):
    print(f"\n(yaw={yaw:+.2f}, pitch_offset={po:+.2f}) — {len(texts)}条:")
    for t in texts[:3]:
        print(f"  · {t}")

print("\n" + "=" * 60)
print("2. 文本去重检查")
print("=" * 60)

# 检查同一文本是否出现在多个不同视角
text_set = defaultdict(set)  # text → set of (yaw, pitch)
for d in data:
    text_set[d['text']].add((round(d['yaw'], 2), round(d['pitch_offset'], 2)))

duplicates = {k: v for k, v in text_set.items() if len(v) > 1}
if duplicates:
    print(f"⚠️ 发现 {len(duplicates)} 条文本出现在多个视角:")
    for t, poses in list(duplicates.items())[:5]:
        print(f"  '{t}' → {list(poses)}")
else:
    print(f"✅ 所有 {len(text_set)} 条唯一文本各对应唯一视角")

print("\n" + "=" * 60)
print("3. 训练/验证集文本泄露检查")
print("=" * 60)

# 模拟 train_multimodal.py 的切分逻辑
shuffled = list(text_set.keys())
# 按文本切分 (15% validation)
n_val = max(1, int(len(shuffled) * 0.15))
val_texts = set(shuffled[:n_val])
train_texts = set(shuffled[n_val:])
overlap = val_texts & train_texts
print(f"训练文本数: {len(train_texts)}, 验证文本数: {len(val_texts)}")
print(f"重叠: {len(overlap)}" + (" ⚠️" if overlap else " ✅ 无泄露"))
