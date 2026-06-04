"""
PTI 反演: 真实图 → w+ (函数接口)
用法: python invert_pti.py                    # 独立运行
      from invert_pti import run_pti; run_pti(G, "photo.jpg")  # 作为模块
"""
import os
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.8'
os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')
os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'
os.environ['CXX'] = '/usr/bin/g++-12'
os.environ['CC'] = '/usr/bin/gcc-12'
import sys, torch, torchvision, math
sys.path.insert(0, "/root/Qwen3-VL")
sys.path.insert(0, "/root/next3d")
import legacy, dnnlib
from PIL import Image

NUM_STEPS = 300


def run_pti(G, image_path, device='cuda'):
    """PTI 反演: 将真实图反演到 EG3D 的 w+ 空间"""
    gt = torchvision.io.read_image(image_path).float() / 127.5 - 1
    if gt.shape[0] == 4:
        gt = gt[:3]
    gt = torch.nn.functional.interpolate(gt.unsqueeze(0).to(device), (512, 512), mode='bilinear')
    gt_128 = torch.nn.functional.interpolate(gt, (128, 128), mode='bilinear')

    # 固定正面 cam
    cam = torch.zeros(1, 25, device=device)
    cam[:, 0] = 0; cam[:, 1] = 0; cam[:, 2] = 2.7
    cam[:, 3] = 18.8; cam[:, 4] = 0.0; cam[:, 5] = 0.2

    z = torch.randn(1, G.z_dim, device=device)
    with torch.no_grad():
        w_init = G.mapping(z, cam, truncation_psi=1.0)

    # PTI: 只优化 w, G 的参数量保持冻结
    w = w_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=0.01, betas=(0.9, 0.999))
    # 确保 synthesis 可以回传梯度 (参数不更新, 但 w 的梯度要流过)
    G.synthesis.requires_grad_(True)

    for step in range(NUM_STEPS):
        opt.zero_grad()
        out = G.synthesis(w, cam, neural_rendering_resolution=128)
        pred = out['image_raw']
        loss = torch.nn.functional.l1_loss(pred, gt_128)
        reg = 0.005 * torch.nn.functional.mse_loss(w, w_init.detach())
        (loss + reg).backward()
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"  PTI step {step+1}: L1={loss.item():.4f} reg={reg.item():.4f}")

    G.synthesis.requires_grad_(False)
    return w.detach()


# ===== 独立运行 =====
if __name__ == "__main__":
    import torch.nn.functional as F
    from bridge import QwenEG3DBridge, pose_6d_to_camera_params

    target_path = "/root/real_input/real_photo.png"
    os.makedirs("/root/output_pti", exist_ok=True)
    os.makedirs("/root/real_input", exist_ok=True)

    if not os.path.exists(target_path):
        print("请放一张图到 /root/real_input/real_photo.png")
        exit()

    # 加载 EG3D
    print("加载 EG3D...")
    with dnnlib.util.open_url("/root/eg3d/networks/nvidia_ffhqrebalanced512-128.pkl") as f:
        G = legacy.load_network_pkl(f)['G_ema'].to('cuda')
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)

    # PTI 反演
    w_final = run_pti(G, target_path)
    torch.save({'w': w_final}, "/root/output_pti/w_plus.pt")

    # 最终渲染 (512×512)
    cam = torch.zeros(1, 25, device='cuda')
    cam[:, 2:6] = torch.tensor([[2.7, 18.8, 0.0, 0.2]])
    with torch.no_grad():
        recon_full = G.synthesis(w_final, cam)['image']
    torchvision.utils.save_image((recon_full[0]*0.5+0.5).clamp(0,1),
                                  "/root/output_pti/recon.png")
    print("重建保存到 /root/output_pti/recon.png")

    # 文本控角度
    print("\n加载 Bridge...")
    b = QwenEG3DBridge()
    ckpt = torch.load("/root/Qwen3-VL/checkpoints/mix_v2_epoch_030.pt",
                       map_location="cuda")
    b.pose.load_state_dict(ckpt["pose"])
    b.eval()

    for text in ["正面平视", "左侧45度平视", "右侧45度仰视", "正面仰视"]:
        with torch.no_grad():
            out = b(text, w_plus=w_final, return_raw_6d=True)
        img, raw = out[0], out[2]
        torchvision.utils.save_image((img[0]*0.5+0.5).clamp(0,1),
                                      f"/root/output_pti/{text}.png")
        print(f"'{text:12s}' → yaw={raw[0,0]:+.2f}")

    print("\n全部结果在 /root/output_pti/")
