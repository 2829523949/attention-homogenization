"""测试最终 checkpoint：纯文本 → 对应角度人脸"""
import sys, torch, torchvision
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge

bridge = QwenEG3DBridge()

# 加载最新 checkpoint
import os
ckpt_dir = "/root/Qwen3-VL/checkpoints"
ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('text_only_') and f.endswith('.pt')],
               key=lambda x: int(x.split('_')[-1].split('.')[0]))
latest = os.path.join(ckpt_dir, ckpts[-1])
print(f"加载: {latest}")

ckpt = torch.load(latest, map_location='cuda')
bridge.pose.load_state_dict(ckpt['pose'])
if 'fusion' in ckpt:
    bridge.fusion.load_state_dict(ckpt['fusion'])
print(f"加载完成 (epoch {ckpt.get('epoch', 0)+1})!\n")

# 测试视角
test_views = [
    "正面平视", "正面仰视", "正面俯视",
    "左侧45度平视", "左侧45度俯视", "左侧45度仰视",
    "右侧45度平视", "右侧45度俯视", "右侧45度仰视",
    "左侧90度平视", "右侧90度平视",
    "背面"
]

os.makedirs("/root/output_final", exist_ok=True)

print("=" * 60)
print("纯文本推理测试")
print("=" * 60)

with torch.no_grad():
    for text in test_views:
        # 纯文本推理（无图片输入）
        img, cam, _, raw_6d = bridge(text, return_gates=True, return_raw_6d=True)
        
        yaw, pitch, radius_raw, fov_raw, _, _ = raw_6d[0].cpu().numpy()
        
        # 解码 radius 和 fov（没有 sigmoid）
        radius = max(radius_raw, 1.0)
        fov = max(fov_raw, 5.0)
        
        print(f"{text:12s} → yaw={yaw:+.2f} pitch={pitch:+.2f} r={radius:.2f} fov={fov:.1f}°")
        
        # 保存图像
        img_out = img[0] * 0.5 + 0.5  # [-1,1] → [0,1]
        torchvision.utils.save_image(img_out.clamp(0,1), f"/root/output_final/{text}.png")

print(f"\n图像已保存到 /root/output_final/")
print("=" * 60)
