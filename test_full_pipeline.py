"""全流程测试: 输入参考图 + 文本 → 同一人脸新角度"""
import sys, torch, torchvision, os, re
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge

b = QwenEG3DBridge()

# 加载 pose
mix = sorted([f for f in os.listdir("/root/Qwen3-VL/checkpoints") if f.startswith('mix_epoch_') and f.endswith('.pt')],
             key=lambda x: int(x.split('_')[-1].split('.')[0]))
if mix:
    b.pose.load_state_dict(torch.load(os.path.join("/root/Qwen3-VL/checkpoints", mix[-1]), map_location='cuda')['pose'])

b.eval()
os.makedirs("/root/output_full_pipeline", exist_ok=True)

# 测试样本: 用验证集的不同人
test_cases = [
    ("z000_y+0.00_p+0.00.png", ["左侧45度平视", "右侧45度俯视", "正面仰视"]),
    ("z001_y-0.52_p+0.35.png", ["正面平视", "右侧45度平视"]),
    ("z003_y+0.79_p+0.00.png", ["左侧45度平视"]),
]

for img_name, texts in test_cases:
    zid = int(re.search(r'z(\d+)', img_name).group(1))
    img_path = f"/root/edit_data/images/{img_name}"
    if not os.path.exists(img_path):
        continue

    # 1. 显示参考图
    ref = torchvision.io.read_image(img_path).float() / 255.0
    torchvision.utils.save_image(ref, f"/root/output_full_pipeline/ref_z{zid}.png")

    for text in texts:
        # 2. 用 z_seed 固定身份 + 文本控制角度
        with torch.no_grad():
            out = b(text, z_seed=zid, return_raw_6d=True)
        img, raw_6d = out[0], out[2]
        y, p = raw_6d[0,0].item(), raw_6d[0,1].item()
        fname = f"z{zid}_{text}.png"
        torchvision.utils.save_image((img[0]*0.5+0.5).clamp(0,1), f"/root/output_full_pipeline/{fname}")
        print(f"参考z{zid} | '{text:16s}' → yaw={y:+.2f} pitch={p:+.2f}")

print("\n保存到 /root/output_full_pipeline/")
print("ref_z*.png = 参考图")
print("z*_*.png  = 同人+编辑后")
