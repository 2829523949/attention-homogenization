"""全面检查 Qwen3-VL 模型结构"""
from transformers import AutoModelForImageTextToText
import torch

model = AutoModelForImageTextToText.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", device_map="auto"
)

print("=== model 顶层 ===")
for name, mod in model.named_children():
    print(f"  {name}: {type(mod).__name__}")

print("\n=== model.model ===")
mm = model.model
for name, mod in mm.named_children():
    if isinstance(mod, (list, tuple)):
        print(f"  {name}: list of {len(mod)}")
    else:
        has_layers = hasattr(mod, 'layers') or hasattr(mod, 'h') or hasattr(mod, 'block')
        layer_count = len(mod.layers) if hasattr(mod, 'layers') else 0
        print(f"  {name}: {type(mod).__name__}  | layers={layer_count if has_layers else '?'}")
        if has_layers:
            for i, layer in enumerate(mod.layers[:2]):
                print(f"    layers[{i}]: {type(layer).__name__}")
            if len(mod.layers) > 2:
                print(f"    ... total {len(mod.layers)} layers")

print("\n=== 检查 vision_tower / vision_model 相关属性 ===")
for attr in dir(mm):
    if 'vision' in attr.lower() or 'visual' in attr.lower():
        try:
            obj = getattr(mm, attr)
            print(f"  {attr}: {type(obj).__name__}")
            if hasattr(obj, 'named_children'):
                for n, m in obj.named_children():
                    print(f"    {n}: {type(m).__name__}")
        except:
            pass

print("\n=== 检查 hidden_size 相关 ===")
for attr in dir(mm):
    if 'hidden' in attr.lower() or 'config' in attr.lower():
        continue
