"""Phase2 验证: 固定 w+ (身份不变) + 文本控角度"""
import sys, torch, torchvision, os, re
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge

b = QwenEG3DBridge()
mix = sorted([f for f in os.listdir("/root/Qwen3-VL/checkpoints") if f.startswith('mix_epoch_') and f.endswith('.pt')],
             key=lambda x: int(x.split('_')[-1].split('.')[0]))
if mix:
    b.pose.load_state_dict(torch.load(os.path.join("/root/Qwen3-VL/checkpoints", mix[-1]), map_location='cuda')['pose'])

# 加载预计算的 w+
identity_w = torch.load("/root/edit_data/identity_w.pt", map_location='cuda')
print(f"加载 {len(identity_w)} 个身份的 w+")

b.eval()
os.makedirs("/root/output_wplus_final", exist_ok=True)

# 测试: 每个身份, 多种文本
for zid in [0, 1, 2]:
    w = identity_w[zid]
    texts = ["正面平视", "正面仰视", "左侧45度平视", "右侧45度俯视"]
    for text in texts:
        with torch.no_grad():
            out = b(text, w_plus=w, return_raw_6d=True)
        img, raw_6d = out[0], out[2]
        y, p = raw_6d[0,0].item(), raw_6d[0,1].item()
        fname = f"id{zid}_{text}.png"
        torchvision.utils.save_image((img[0]*0.5+0.5).clamp(0,1), f"/root/output_wplus_final/{fname}")
        print(f"id{zid} | '{text:16s}' → yaw={y:+.2f} pitch={p:+.2f}")

print("\n保存到 /root/output_wplus_final/")
print("同一数字(id) = 同一身份, 不同角度")
