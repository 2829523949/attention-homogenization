"""
GAN 反演 (PTI 风格): 真实人脸 → W+ → EG3D 角度编辑
====================================================
参考: PTI (Pivotal Tuning Inversion, CVPR 2022)

全部图像读取改用 PIL，避免格式兼容问题
"""
import os, sys, math, torch, torch.nn.functional as F
import torchvision
from PIL import Image

sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, pose_6d_to_camera_params
import lpips


def load_image_pil(img_path, size=512):
    """PIL 读取图像，转换为 [-1,1] 的 tensor [1,3,H,W]"""
    img = Image.open(img_path).convert('RGB')
    tensor = torchvision.transforms.ToTensor()(img)  # [3,H,W] [0,1]
    tensor = tensor.unsqueeze(0) * 2 - 1  # [1,3,H,W] [-1,1]
    
    # 居中裁剪为正方形
    _, _, H, W = tensor.shape
    s = min(H, W)
    tensor = torchvision.transforms.functional.center_crop(tensor, (s, s))
    
    # 缩放到目标尺寸
    if s != size:
        tensor = F.interpolate(tensor, (size, size), mode='bilinear')
    
    return tensor.cuda()


def invert_face_wplus(bridge, img_path, steps=1000, lr=0.01, save_dir="/root/output_inversion"):
    """PTI Phase 1: W+ 空间优化"""
    os.makedirs(save_dir, exist_ok=True)
    
    perceptual = lpips.LPIPS(net='alex').cuda().eval()
    
    print(f"[反演] 加载: {img_path}")
    gt = load_image_pil(img_path, 512)
    torchvision.utils.save_image((gt[0]*0.5+0.5).clamp(0,1), f"{save_dir}/input.png")
    print(f"  尺寸: {gt.shape}")

    cam_front = pose_6d_to_camera_params(
        torch.tensor([[0.0, math.pi/2, 2.7, 18.8, 0.0, 0.2]], device="cuda"))
    
    # 初始化 W+
    with torch.no_grad():
        z = torch.randn(1, bridge.eg3d.z_dim, device='cuda')
        w_plus = bridge.eg3d.mapping(z, cam_front)  # [1, num_ws, 512]
        num_ws = w_plus.shape[1]
    w_plus = w_plus.detach().clone().requires_grad_(True)
    print(f"[反演] num_ws={num_ws}, 优化 {steps} 步...")
    
    optimizer = torch.optim.Adam([w_plus], lr=lr, betas=(0.9, 0.999))
    best_wp, best_loss = None, float('inf')
    
    for step in range(steps):
        optimizer.zero_grad()
        out = bridge.eg3d.synthesis(w_plus, cam_front)['image']
        
        loss_l1 = F.l1_loss(out, gt)
        loss_perc = perceptual(out, gt).mean()
        loss = loss_l1 + loss_perc
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w_plus], 1.0)
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_wp = w_plus.detach().clone()
        
        if (step + 1) % 100 == 0:
            print(f"  step {step+1:4d}: L1={loss_l1.item():.4f} perc={loss_perc.item():.4f}")
    
    with torch.no_grad():
        out = bridge.eg3d.synthesis(best_wp, cam_front)['image']
    torchvision.utils.save_image((out[0]*0.5+0.5).clamp(0,1), f"{save_dir}/recon_wplus.png")
    print(f"[反演] 完成: loss={best_loss:.4f}")
    
    return best_wp.cpu()


if __name__ == "__main__":
    bridge = QwenEG3DBridge()
    bridge.pose.load_state_dict(torch.load(
        "/root/Qwen3-VL/checkpoints/mix_v2_epoch_030.pt", map_location="cuda")["pose"])
    bridge.eval()
    
    # 测试数据集图片
    invert_face_wplus(bridge, "/root/edit_data/images/z005_y+0.00_p+0.00.png",
                      steps=500, save_dir="/root/output_inversion/z005")
