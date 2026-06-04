"""检查 backbone 内的 synthesis 结构"""
import sys, torch
sys.path.insert(0, "/root/eg3d/eg3d")
import legacy, dnnlib

with dnnlib.util.open_url("/root/eg3d/networks/nvidia_ffhqrebalanced512-128.pkl") as f:
    G = legacy.load_network_pkl(f)["G_ema"].to("cuda")

bb = G.backbone
synth = bb.synthesis
print(f"backbone: {type(bb).__name__}")
print(f"synth type: {type(synth).__name__}")

if hasattr(synth, 'block_resolutions'):
    print(f"block_resolutions: {synth.block_resolutions}")
    
for res in synth.block_resolutions:
    block = getattr(synth, f'b{res}')
    print(f"  b{res}: conv1.out_channels={block.conv1.out_channels}")
