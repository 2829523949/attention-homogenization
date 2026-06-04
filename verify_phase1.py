import sys, torch, torchvision
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, pose_6d_to_camera_params

# 1. 加载
b = QwenEG3DBridge()
ckpt = torch.load("/root/Qwen3-VL/checkpoints/mix_v2_epoch_035.pt", map_location="cuda")
b.pose.load_state_dict(ckpt["pose"])
b.eval()

# 2. 随机人脸 + 正面 cam
G = b.eg3d
z = torch.randn(1, G.z_dim, device='cuda')
cam_f = torch.zeros(1, 25, device='cuda')
cam_f[:, 2:6] = torch.tensor([[2.7, 18.8, 0.0, 0.2]])

with torch.no_grad():
    ws = G.mapping(z, cam_f, truncation_psi=0.7)
    # 人脸正面
    img_f = G.synthesis(ws, cam_f, neural_rendering_resolution=128)['image']

# 3. 文本 → 位姿 (调试版)
from bridge import get_llm_hidden
for text in ["正面平视", "左侧60度平视", "右侧30度仰视"]:
    h = get_llm_hidden(b.qwen, b.processor, text)
    print(f"'{text}' hidden mean={h.mean():.4f} std={h.std():.4f}")
    raw = b.pose(h.unsqueeze(0))
    print(f"  → raw_6d: {raw[0].detach().cpu().numpy()}")
    with torch.no_grad():
        img = b(text)[0]
    torchvision.utils.save_image((img[0]*0.5+0.5).clamp(0,1), f"/root/output_pti/{text}.png")
    print(f"  → yaw={raw[0,0]:+.1f}°")

# 4. 保存正面
torchvision.utils.save_image((img_f[0]*0.5+0.5).clamp(0,1), "/root/output_pti/正面平视.png")
print("\n全部在 /root/output_pti/")
