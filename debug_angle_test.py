"""调试用: 检查 test_angle_accuracy.py 的数据分组"""
import sys, torch
sys.path.insert(0, "/root/Qwen3-VL")
from train_edit import EditDataset
from collections import defaultdict

ds = EditDataset()

# 统计每个目标角度的样本数
by_target = defaultdict(lambda: {"template": [], "diverse": []})

for item in ds.train_pairs[:5000]:
    text = item["text"]
    target = torch.tensor(item["target_raw_6d"])
    key = (round(target[0].item(), 2), round(target[1].item(), 2))
    
    is_template = ("平视" in text or "仰视" in text or "俯视" in text) and \
                  ("度" in text or "正面" in text or "左侧" in text or "右侧" in text)
    if is_template:
        by_target[key]["template"].append(text)
    else:
        by_target[key]["diverse"].append(text)

print(f"目标角度种类: {len(by_target)}")
print(f"\n{'目标(yaw,pitch)':>20s} {'模板数':>8s} {'多样化数':>10s}")
print("-" * 50)

for key in sorted(by_target.keys())[:20]:
    t_count = len(by_target[key]["template"])
    d_count = len(by_target[key]["diverse"])
    print(f"{key[0]:+.2f}, {key[1]:+.2f}  {t_count:8d} {d_count:10d}")

# 检查是否有目标角度没有模板文本
no_template = [k for k, v in by_target.items() if len(v["template"]) == 0]
print(f"\n无模板的目标角度: {len(no_template)} 个")
if no_template:
    print("示例:", no_template[:5])
