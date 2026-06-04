"""测试 w+ 反演 Phase2: 同一身份, 文本控角度"""
import sys, torch, torchvision, os
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge

b = QwenEG3DBridge()

# 加载预训练 pose 和 w+
ckpt_dir = "/root/Qwen3-VL/checkpoints"
mix = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('mix_epoch_') and f.endswith('.pt')],
             key=lambda x: int(x.split('_')[-1].split('.')[0]))
if mix:
    b.pose.load_state_dict(torch.load(os.path.join(ckpt_dir, mix[-1]), map_location='cuda')['pose'])

identity_w = torch.load("/root/edit_data/identity_w.pt", map_location='cuda')
print(f"加载 {len(identity_w)} 个身份的 w+")

b.eval()
os.makedirs("/root/output_wplus", exist_ok=True)

# 测试: 取身份0的 w+, 用不同文本改变角度
w = identity_w[0]

tests = ["正面平视", "正面仰视", "正面俯视",
         "左侧45度平视", "右侧45度平视",
         "左侧45度仰视", "右侧45度俯视"]

for text in tests:
    with torch.no_grad():
        out = b(text, w_plus=w, return_raw_6d=True)
    img, cam, raw_6d = out[0], out[1], out[2]
    y, p = raw_6d[0,0].item(), raw_6d[0,1].item()
    print(f"'{text:16s}' → yaw={y:+.2f} pitch={p:+.2f}")
    torchvision.utils.save_image((img[0]*0.5+0.5).clamp(0,1),
                                 f"/root/output_wplus/{text}.png")

print("\n保存到 /root/output_wplus/")
print("所有输出应为同一个人 (身份0) 的不同角度")
