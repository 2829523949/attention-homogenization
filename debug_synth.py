"""检查 loaded EG3D 的合成网络结构"""
import sys, torch
sys.path.insert(0, "/root/eg3d/eg3d")
import legacy, dnnlib, inspect

with dnnlib.util.open_url("/root/eg3d/networks/nvidia_ffhqrebalanced512-128.pkl") as f:
    G = legacy.load_network_pkl(f)["G_ema"].to("cuda")

print(f"G type: {type(G).__name__}")
print(f"G.synthesis type: {type(G.synthesis).__name__}")

# 查看 G 的所有属性
for name in dir(G):
    if not name.startswith('_'):
        obj = getattr(G, name)
        if hasattr(obj, '__call__') or isinstance(obj, torch.nn.Module):
            if not name.startswith('forward'):
                sub = list(obj.children()) if hasattr(obj, 'children') else []
                print(f"  {name}: {type(obj).__name__} ({len(sub)} children)" if sub else f"  {name}: {type(obj).__name__}")

# 尝试直接访问 synthesis 模块
if hasattr(G, 'synthesis'):
    synth = G.synthesis
    if hasattr(synth, 'block_resolutions'):
        print(f"\nblock_resolutions: {synth.block_resolutions}")
    # 遍历
    for name2 in dir(synth):
        if 'block' in name2.lower() and not name2.startswith('_'):
            print(f"  {name2}")
