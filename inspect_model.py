"""查看 Qwen3-VL-2B 实际结构"""
import sys, torch
sys.path.insert(0, "/root/Qwen3-VL")
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

# 探索 LLM 子模块
llm = model.model
print("=== LLM内部属性 ===")
for name in dir(llm):
    if not name.startswith('_'):
        obj = getattr(llm, name)
        if hasattr(obj, '__len__') and not isinstance(obj, (str, dict)):
            try:
                print(f"  {name}: {type(obj).__name__} len={len(obj)}")
            except:
                print(f"  {name}: {type(obj).__name__}")
        elif hasattr(obj, 'shape'):
            print(f"  {name}: {obj.shape}")
        elif isinstance(obj, torch.nn.Module):
            children = list(obj.children())
            if children:
                print(f"  {name}: {type(obj).__name__} ({len(children)} children)")

# Vision模块
vis = model.model.visual
print(f"\n=== Vision: {type(vis).__name__} ===")
for name in dir(vis):
    if not name.startswith('_'):
        obj = getattr(vis, name)
        if isinstance(obj, torch.nn.Module) and type(obj).__name__ not in ['ModuleDict', 'ModuleList']:
            sub = list(obj.children())
            print(f"  {name}: {type(obj).__name__}", f"({len(sub)} sub)" if sub else "")
        elif hasattr(obj, '__len__'):
            try:
                print(f"  {name}: len={len(obj)}")
            except:
                pass

# 查看 deepstack 输出
import PIL.Image
img = PIL.Image.new('RGB', (224, 224))
processor = AutoProcessor.from_pretrained("/root/Qwen3-VL-2B-Instruct")
inputs = processor(text=["test"], images=[img], return_tensors="pt", padding=True).to(model.device)
if 'pixel_values' in inputs:
    with torch.no_grad():
        out = vis(hidden_states=inputs['pixel_values'], grid_thw=inputs['image_grid_thw'])
    print(f"\n=== Vision输出 ===")
    for attr in dir(out):
        if not attr.startswith('_'):
            v = getattr(out, attr)
            if isinstance(v, (list, tuple)):
                print(f"  {attr}: list[{len(v)}] x {v[0].shape}")
            elif hasattr(v, 'shape'):
                print(f"  {attr}: {v.shape}")
            else:
                print(f"  {attr}: {type(v).__name__}")
