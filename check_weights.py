"""检查 pose 权重加载是否正常"""
import sys, torch
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge

b = QwenEG3DBridge()

# 加载前
print("加载前 pose 参数:")
for name, p in b.pose.named_parameters():
    print(f"  {name}: shape={p.shape}, mean={p.mean().item():.4f}")
    break

# 加载
ckpt = torch.load("/root/Qwen3-VL/checkpoints/mix_epoch_045.pt", map_location="cuda")
print(f"\ncheckpoint 结构: {list(ckpt.keys())}")
b.pose.load_state_dict(ckpt["pose"])

# 加载后
print("\n加载后 pose 参数:")
for name, p in b.pose.named_parameters():
    print(f"  {name}: shape={p.shape}, mean={p.mean().item():.4f}")
    break

# 简单测试
b.eval()
text = "左侧45度平视"
with torch.no_grad():
    out = b(text, return_raw_6d=True)
raw = out[2][0]
print(f"\n'{text}': yaw={raw[0]:+.2f} pitch={raw[1]:+.2f}")
print(f"目标: yaw=-0.79(左侧45°) pitch=+1.57(平视)")
