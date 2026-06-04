"""封闭测试: seed固定 + cam固定 → 图 → w+ → 重建 → 编辑角度"""
import sys, torch, torchvision, math, os
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, pose_6d_to_camera_params

b = QwenEG3DBridge()

# 1. 用已知 seed 在正面平视下生成"输入图"
zid = 0
torch.manual_seed(zid)
z = torch.randn(1, b.eg3d.z_dim, device='cuda')

cam_front = pose_6d_to_camera_params(
    torch.tensor([[0.0, math.pi/2, 2.7, 18.8, 0.0, 0.2]], device="cuda"))

with torch.no_grad():
    ws_ref = b.eg3d.mapping(z, cam_front, truncation_psi=0.7)
    img_ref = b.eg3d.synthesis(ws_ref, cam_front)['image']

os.makedirs("/root/output_closed", exist_ok=True)
torchvision.utils.save_image((img_ref[0]*0.5+0.5).clamp(0,1),
                              "/root/output_closed/01_input_seed0.png")
print("1. 生成输入图 (seed=0, 正面平视) ✅")

# 2. w+ = mapping(seed=0, cam=正面平视) — 这就是 ws_ref
# 用 ws_ref 和 cam_front 重建
with torch.no_grad():
    img_recon = b.eg3d.synthesis(ws_ref, cam_front)['image']
    mse = torch.nn.functional.mse_loss(img_recon, img_ref).item()

torchvision.utils.save_image((img_recon[0]*0.5+0.5).clamp(0,1),
                              f"/root/output_closed/02_recon_mse{mse:.4f}.png")
print(f"2. w+ 重建: MSE={mse:.4f} {'✅ 完全一致' if mse < 1e-6 else '⚠️ 有差异'}")

# 3. 加载 pose
b.pose.load_state_dict(torch.load(
    "/root/Qwen3-VL/checkpoints/mix_epoch_045.pt", map_location="cuda")["pose"])
b.eval()

# 4. 用固定 w+ + 文本换角度 (不重新 mapping)
for text in ["正面平视", "左侧45度平视", "右侧45度仰视", "正面俯视"]:
    with torch.no_grad():
        out = b(text, w_plus=ws_ref, return_raw_6d=True)
    img, raw = out[0], out[2]
    y, p = raw[0,0].item(), raw[0,1].item()
    torchvision.utils.save_image((img[0]*0.5+0.5).clamp(0,1),
                                  f"/root/output_closed/03_{text}.png")
    print(f"3. '{text:12s}' → yaw={y:+.2f} pitch={p:+.2f}")

print("\n结果在 /root/output_closed/")
print("01_input = 输入 | 02_recon = 同一w+重建 | 03_* = 文本编辑")
