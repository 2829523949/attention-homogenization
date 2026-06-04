"""验证 Phase 1: LLM → 6D → EG3D"""
import sys, math
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge

b = QwenEG3DBridge()

tests = [
    ("左侧60度平视", -1.05),
    ("左侧30度平视", -0.52),
    ("正面平视",      0.00),
    ("右侧30度仰视",  0.52),
    ("右侧60度俯视",  1.05),
]

print(f"{'文本':>12s}  {'预测yaw':>8s}  {'目标yaw':>8s}  {'Δyaw°':>6s}")
print("-" * 45)
for text, gt_yaw in tests:
    _, _, raw = b(text, return_raw_6d=True)
    yaw = raw[0, 0].item()
    ye = abs(yaw - gt_yaw) * 180 / math.pi
    print(f"{text:>12s}  {yaw:+8.3f}  {gt_yaw:+8.3f}  {ye:5.2f}°")
