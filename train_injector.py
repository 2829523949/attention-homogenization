"""
训练 Injector: 64×64 低分辨率渲染 Loss (可控显存)
"""
import os, sys, torch, torch.nn.functional as F, torchvision
from tqdm import tqdm

sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, get_multimodal_features, pose_6d_to_camera_params

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

bridge = QwenEG3DBridge()
import glob
ckpts = sorted(glob.glob("/root/Qwen3-VL/checkpoints/mix_epoch_*.pt"),
               key=lambda x: int(x.split('_')[-1].split('.')[0]))
if ckpts:
    bridge.pose.load_state_dict(torch.load(ckpts[-1], map_location='cuda')['pose'])

for p in bridge.parameters(): p.requires_grad_(False)
for p in bridge.injector.parameters(): p.requires_grad = True

data = torch.load("/root/edit_data/dataset.pt")
train_samples = [s for s in data['samples'] if s['zid'] not in {0,1,2,3}]

params = list(bridge.injector.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)

for step in tqdm(range(500), desc="Injector"):
    optimizer.zero_grad()
    s = train_samples[step % len(train_samples)]
    img_path = f"/root/edit_data/images/{s['image']}"
    text = s['text']

    # GT 图
    gt_img = torchvision.io.read_image(img_path).float() / 127.5 - 1  # [3,H,W] [-1,1]
    gt_img = gt_img.unsqueeze(0).cuda()

    # 前向 (可选是否 full pipeline)
    with torch.no_grad():
        ds_spatial, llm_hidden = get_multimodal_features(bridge.qwen, bridge.processor, text, img_path,
            dino=bridge.dino, dino_processor=bridge.dino_processor)
        ds_proj = [bridge.dino_proj(f) for f in ds_spatial]  # list[Tensor], fusion 内部会 pool
        fused, gates = bridge.fusion(llm_hidden, ds_proj)
        raw_6d = bridge.pose(fused.unsqueeze(0))
        cam = pose_6d_to_camera_params(raw_6d)
        z = torch.randn(1, bridge.eg3d.z_dim, device='cuda')
        ws = bridge.eg3d.mapping(z, cam, truncation_psi=0.7)

    # 注入 planes (有梯度)
    pred_planes = bridge.injector.inject_planes(ws, ds_spatial, gates)
    bridge.eg3d._last_planes = pred_planes

    # 低分辨率渲染 (保留梯度通过 injector)
    out = bridge.eg3d.synthesis(ws, cam, neural_rendering_resolution=64, use_cached_backbone=True)
    pred_img = out['image_raw']  # [1, 3, 64, 64]

    gt_64 = F.interpolate(gt_img, size=(64,64), mode='bilinear', align_corners=False)

    loss = F.l1_loss(pred_img, gt_64)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 5.0)
    optimizer.step()

    if (step+1) % 100 == 0:
        vram = torch.cuda.memory_allocated() / 1024**3
        print(f"  step {step+1}: loss={loss.item():.4f} gates={gates.data.cpu().numpy().round(2)} VRAM={vram:.2f}GB")

torch.save({'injector': bridge.injector.state_dict()},
           "/root/Qwen3-VL/checkpoints/injector_trained.pt")
print("完成!")
