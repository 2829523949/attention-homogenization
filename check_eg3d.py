"""检查 EG3D Synthesis 结构"""
import sys, torch
sys.path.insert(0, "/root/eg3d/eg3d")
import legacy, dnnlib
with dnnlib.util.open_url("/root/eg3d/networks/nvidia_ffhqrebalanced512-128.pkl") as f:
    G = legacy.load_network_pkl(f)["G_ema"].to("cuda")
synth = G.synthesis

# synthesis 可能是 nn.Module 或函数
print(f"type: {type(synth).__name__}")
if hasattr(synth, 'img_resolution'):
    print(f"img_resolution: {synth.img_resolution}")
else:
    print("无 img_resolution 属性")
    
if hasattr(synth, 'block_resolutions'):
    print(f"block_resolutions: {synth.block_resolutions}")
else:
    # 检查 synthesis 内部
    children = list(synth.named_children()) if hasattr(synth, 'named_children') else []
    for name, mod in children[:5]:
        if hasattr(mod, 'block_resolutions'):
            print(f"{name}: block_resolutions={mod.block_resolutions}")
        print(f"  {name}: {type(mod).__name__}", end="")
        if hasattr(mod, 'img_resolution'):
            print(f" res={mod.img_resolution}", end="")
        print()
