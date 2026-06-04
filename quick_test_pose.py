"""快速检查 pose 预测是否正常"""
import sys, math
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge

b = QwenEG3DBridge()

tests = [
    (-0.79, "左侧45度平视"),
    (-0.52, "左侧30度平视"),
    (-0.26, "左侧15度平视"),
    ( 0.00, "正面平视"),
    ( 0.26, "右侧15度平视"),
    ( 0.52, "右侧30度平视"),
    ( 0.79, "右侧45度平视"),
]
gt_pitch = math.pi / 2  # 平视

print(f"{'文本':>12s}  {'预测yaw':>8s}  {'目标yaw':>8s}  {'Δyaw°':>6s}  {'预测pitch':>8s}  {'Δpitch°':>6s}")
print("-" * 65)
for gt_yaw, text in tests:
    _, _, raw = b(text, return_raw_6d=True)
    yaw = raw[0, 0].item()
    pitch = raw[0, 1].item()
    ye = abs(yaw - gt_yaw) * 180 / math.pi
    pe = abs(pitch - gt_pitch) * 180 / math.pi
    print(f"{text:>12s}  {yaw:+8.3f}  {gt_yaw:+8.3f}  {ye:5.2f}°  {pitch:+8.3f}  {pe:5.2f}°")

print("-" * 65)
print("正常期望: yaw平均误差 ≈ 4~7°, pitch平均误差 ≈ 5~10°")
