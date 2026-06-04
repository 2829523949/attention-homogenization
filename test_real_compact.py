import sys, torch, torchvision, os
sys.path.insert(0, "/root/Qwen3-VL"); sys.path.insert(0, "/root/eg3d/eg3d")
import legacy, dnnlib, torch.nn.functional as F

# 1. 用数据集图作为"真实输入"
real_img = "/root/edit_data/images/z005_y+0.00_p+0.00.png"
gt = torchvision.io.read_image(real_img).float() / 127.5 - 1
gt = F.interpolate(gt.unsqueeze(0).cuda(), (128, 128))

# 2. GAN 反演优化 z
with dnnlib.util.open_url("/root/eg3d/networks/nvidia_ffhqrebalanced512-128.pkl") as f:
    G = legacy.load_network_pkl(f)["G_ema"].to("cuda")
G.eval()
z = torch.randn(1, G.z_dim, device="cuda", requires_grad=True)
opt = torch.optim.Adam([z], lr=0.05)
cam_base = torch.tensor([[1,0,0,0,0,1,0,0,0,0,1,2.7,0,0,0,1,1,0,0.5,0,1,0.5,0,0,1]], device="cuda")

print("GAN 反演...")
for step in range(100):
    opt.zero_grad()
    ws = G.mapping(z, cam_base, truncation_psi=0.7)
    pred = G.synthesis(ws, cam_base, neural_rendering_resolution=128)["image_raw"]
    F.l1_loss(pred, gt).backward()
    opt.step()
print(f"loss={F.l1_loss(G.synthesis(G.mapping(z, cam_base, truncation_psi=0.7), cam_base, neural_rendering_resolution=128)['image_raw'], gt).item():.4f}")

# 3. 用 bridge 做角度编辑
from bridge import QwenEG3DBridge
b = QwenEG3DBridge()
b.pose.load_state_dict(torch.load("/root/Qwen3-VL/checkpoints/mix_epoch_045.pt", map_location="cuda")["pose"])
b.eval()
os.makedirs("/root/output_real", exist_ok=True)
ws_final = G.mapping(z, cam_base, truncation_psi=0.7)

for t in ["正面平视", "正面仰视", "左侧45度平视", "右侧45度平视"]:
    out = b(t, w_plus=ws_final, return_raw_6d=True)
    img, raw = out[0], out[2]
    print(f"{t}: yaw={raw[0,0]:+.2f} pitch={raw[0,1]:+.2f}")
    torchvision.utils.save_image((img[0]*0.5+0.5).clamp(0,1), f"/root/output_real/{t}.png")
print("完成")
