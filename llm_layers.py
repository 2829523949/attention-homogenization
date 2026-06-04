"""查看 Qwen3-VL-2B LLM 层数 + deepstack 索引"""
import sys, torch
sys.path.insert(0, "/root/Qwen3-VL")
from transformers import AutoModelForImageTextToText
import PIL.Image

model = AutoModelForImageTextToText.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

# LLM 层数
lm = model.model.language_model
print("=== Language Model ===")
for name in dir(lm):
    if not name.startswith('_'):
        obj = getattr(lm, name)
        if isinstance(obj, torch.nn.Module):
            children = list(obj.children())
            print(f"  {name}: {type(obj).__name__} ({len(children)} children)" if children else f"  {name}: {type(obj).__name__}")
    if name == 'model':
        for n2 in dir(obj):
            if 'layer' in n2.lower() or 'block' in n2.lower():
                v = getattr(obj, n2)
                if hasattr(v, '__len__'):
                    print(f"    {n2}: {len(v)}")

# deepstack 索引
vis = model.model.visual
print(f"\ndeepstack_visual_indexes: {vis.deepstack_visual_indexes}")
print(f"deepstack_merger_list: {len(vis.deepstack_merger_list)}")
print(f"第一个 deepstack 输出: [64, 2048]")
print(f"pooler_output: [64, 2048] (64 patches × 2048 dim)")
print(f"last_hidden_state: [256, 1024]")

# 推测 deepstack 的层来源
idxs = vis.deepstack_visual_indexes
print(f"\n=== DeepStack 层索引分析 ===")
print(f"Qwen3-VL ViT 共 24 层, deepstack 从层 {idxs} 提取")
print(f"范围 {idxs[0]}/24 ~ {idxs[-1]}/24")
print(f"按比例: {idxs[0]/24*100:.0f}% ~ {idxs[-1]/24*100:.0f}%")
