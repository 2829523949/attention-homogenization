"""测试 ±45° 范围内的精度"""
import sys, torch, torchvision, os, math
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, get_llm_hidden, pose_6d_to_camera_params

bridge = QwenEG3DBridge()
ckpt = torch.load("/root/Qwen3-VL/checkpoints/mix_epoch_045.pt", map_location='cuda')
bridge.pose.load_state_dict(ckpt['pose'])
print("加载 mix_epoch_045.pt\n")

# ±45°范围内的视角
tests = [
    # (text, yaw_gt, pitch_gt)
    ("左侧45度俯视", -0.79, 1.22),
    ("左侧45度平视", -0.79, 1.57),
    ("左侧45度仰视", -0.79, 1.92),
    ("正面俯视", 0.00, 1.22),
    ("正面平视", 0.00, 1.57),
    ("正面仰视", 0.00, 1.92),
    ("右侧45度俯视", 0.79, 1.22),
    ("右侧45度平视", 0.79, 1.57),
    ("右侧45度仰视", 0.79, 1.92),
]

print(f"{'文本':16s} | {'pred yaw':>8s} {'pred pitch':>10s} | {'Δy':>4s} {'Δp':>4s}")
print("-" * 50)

total_dy = total_dp = 0
for text, gt_y, gt_p in tests:
    with torch.no_grad():
        h = get_llm_hidden(bridge.qwen, bridge.processor, text)
        raw = bridge.pose(h.unsqueeze(0))
    py, pp = raw[0,0].item(), raw[0,1].item()
    dy, dp = abs(py - gt_y), abs(pp - gt_p)
    total_dy += dy; total_dp += dp
    print(f"{text:16s} | {py:>+8.2f} {pp:>+10.2f} | {dy:.2f}  {dp:.2f}")

n = len(tests)
print(f"\n平均: Δy={total_dy/n:.3f} ({total_dy/n*180/math.pi:.1f}°)  Δp={total_dp/n:.3f} ({total_dp/n*180/math.pi:.1f}°)")
