"""检查 Qwen3-VL 模型结构"""
from transformers import AutoModelForImageTextToText
model = AutoModelForImageTextToText.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", device_map="auto"
)

print("=== 顶层 ===")
for name, mod in model.named_children():
    print(f"  {name}: {type(mod).__name__}")

print("\n=== model.vision_model ===")
vm = model.model.vision_model
for name, mod in vm.named_children():
    if isinstance(mod, (list, tuple)):
        print(f"  {name}: list of {len(mod)} items")
    elif hasattr(mod, '__len__') and not isinstance(mod, torch.nn.Module):
        print(f"  {name}: len={len(mod)}")
    else:
        print(f"  {name}: {type(mod).__name__}")

print("\n=== vision_model.visual ===")
vis = vm.visual  # if exists
for name, mod in vis.named_children():
    print(f"  {name}: {type(mod).__name__}  |  {getattr(mod, 'hidden_size', '?')}")

print("\n=== model.text_model ===")
tm = model.model.text_model
for name, mod in tm.named_children():
    print(f"  {name}: {type(mod).__name__}")

print("\n=== text_model.layers ===")
for i, layer in enumerate(tm.layers[:3]):
    print(f"  layer[{i}]: {type(layer).__name__}")
print(f"  ... total {len(tm.layers)} layers")
