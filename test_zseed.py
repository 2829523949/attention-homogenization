"""测试 Phase2: z_seed 固定身份, 文本控角度"""
import sys, torch, torchvision, os
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge

b = QwenEG3DBridge()

ckpt_dir = "/root/Qwen3-VL/checkpoints"
mix = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('mix_epoch_') and f.endswith('.pt')],
             key=lambda x: int(x.split('_')[-1].split('.')[0]))
if mix:
    b.pose.load_state_dict(torch.load(os.path.join(ckpt_dir, mix[-1]), map_location='cuda')['pose'])

b.eval()
os.makedirs("/root/output_zseed", exist_ok=True)

zid = 0
tests = ["正面平视", "正面仰视", "正面俯视",
         "左侧45度平视", "右侧45度平视"]

for text in tests:
    with torch.no_grad():
        out = b(text, z_seed=zid, return_raw_6d=True)
    img, cam, raw_6d = out[0], out[1], out[2]
    y, p = raw_6d[0,0].item(), raw_6d[0,1].item()
    print(f"'{text:16s}' → yaw={y:+.2f} pitch={p:+.2f}")
    torchvision.utils.save_image((img[0]*0.5+0.5).clamp(0,1),
                                 f"/root/output_zseed/{text}.png")

print("\n保存到 /root/output_zseed/")
print("z_seed=0 固定身份, 所有角度应为同一人")
