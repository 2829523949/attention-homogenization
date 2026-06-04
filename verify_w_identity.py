"""验证 w+ 是否真的对应输入图像的身份"""
import sys, torch, torchvision, os, math
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, pose_6d_to_camera_params

b = QwenEG3DBridge()
b.pose.load_state_dict(torch.load("/root/Qwen3-VL/checkpoints/mix_epoch_045.pt", map_location="cuda")["pose"])
b.eval()

identity_w = torch.load("/root/edit_data/identity_w.pt", map_location="cuda")

os.makedirs("/root/output_verify_w", exist_ok=True)

for zid in [10, 25]:
    img_path = f"/root/edit_data/images/z{zid:03d}_y+0.00_p+0.00.png"
    ref = torchvision.io.read_image(img_path).float() / 255.0
    torchvision.utils.save_image(ref, f"/root/output_verify_w/input_z{zid}.png")

    # 生成 w+ 时的原始 cam (正面平视)
    w = identity_w[zid]
    
    # 用 w+ 和正面 cam 渲染
    cam_front = pose_6d_to_camera_params(
        torch.tensor([[0.0, math.pi/2, 2.7, 18.8, 0.0, 0.2]], device="cuda"))
    
    with torch.no_grad():
        ws = w.to('cuda')
        recon = b.eg3d.synthesis(ws, cam_front)['image']
    torchvision.utils.save_image((recon[0]*0.5+0.5).clamp(0,1),
                                 f"/root/output_verify_w/recon_z{zid}.png")
    
    # 计算相似度 (MSE)
    recon_flat = (recon[0]*0.5+0.5).clamp(0,1)
    ref_flat = ref.cuda()
    mse = torch.nn.functional.mse_loss(recon_flat, ref_flat).item()
    print(f"z{zid}: 重建 MSE = {mse:.4f}, w+ vs 原图 {'✅ 同人' if mse < 0.05 else '❌ 不同人'}")

print("\n检查 /root/output_verify_w/ 中 input vs recon 是否一致")
