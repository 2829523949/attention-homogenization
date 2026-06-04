"""真实照片反演 → 文本控角度"""
import sys, torch, torchvision, os, math
from PIL import Image

# 1. 读取 GIF 第一帧
gif_path = "/mnt/c/Users/张俊豪/Downloads/712ce4ad9d1c42bc84d050ce0bbb47bd.gif"
img = Image.open(gif_path).convert('RGB')
img.save("/root/real_input/real_photo.png")
print(f"输入图: {img.size}")

# 2. 加载 EG3D 和 bridge
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, pose_6d_to_camera_params

b = QwenEG3DBridge()
b.pose.load_state_dict(torch.load(
    "/root/Qwen3-VL/checkpoints/mix_epoch_045.pt", map_location="cuda")["pose"])

# 3. GAN 反演优化 z
gt = torchvision.io.read_image("/root/real_input/real_photo.png").float() / 127.5 - 1
if gt.shape[0] == 4: gt = gt[:3]
# resize 到 EG3D 支持的尺寸
H, W = gt.shape[1], gt.shape[2]
s = min(H, W)
gt = torchvision.transforms.functional.center_crop(gt, (s, s))
gt = torch.nn.functional.interpolate(gt.unsqueeze(0).cuda(), (512, 512), mode='bilinear')
gt_128 = torch.nn.functional.interpolate(gt, (128, 128), mode='bilinear')

z = torch.randn(1, b.eg3d.z_dim, device='cuda', requires_grad=True)
opt = torch.optim.Adam([z], lr=0.05)

cam_front = pose_6d_to_camera_params(
    torch.tensor([[0.0, math.pi/2, 2.7, 18.8, 0.0, 0.2]], device="cuda"))

os.makedirs("/root/output_real_edit", exist_ok=True)

print("GAN 反演中 (200步)...")
for step in range(200):
    opt.zero_grad()
    ws = b.eg3d.mapping(z, cam_front, truncation_psi=0.7)
    out = b.eg3d.synthesis(ws, cam_front, neural_rendering_resolution=128)
    loss = torch.nn.functional.l1_loss(out['image_raw'], gt_128)
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(f"  step {step}: loss={loss.item():.4f}")

# 4. 用 w+ 做角度编辑
with torch.no_grad():
    ws_final = b.eg3d.mapping(z, cam_front, truncation_psi=0.7)
    recon = b.eg3d.synthesis(ws_final, cam_front)['image']
torchvision.utils.save_image((recon[0]*0.5+0.5).clamp(0,1),
                              "/root/output_real_edit/recon_input.png")
print(f"\n重建完成, 保存到 /root/output_real_edit/recon_input.png")

b.eval()
for text in ["正面平视", "左侧45度平视", "右侧45度仰视", "正面仰视"]:
    with torch.no_grad():
        out = b(text, w_plus=ws_final, return_raw_6d=True)
    img, raw = out[0], out[2]
    torchvision.utils.save_image((img[0]*0.5+0.5).clamp(0,1),
                                  f"/root/output_real_edit/{text}.png")
    print(f"'{text:12s}' → yaw={raw[0,0]:+.2f}")

print("\n全部结果在 /root/output_real_edit/")
