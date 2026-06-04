"""
Injector v3（最终版）: 重建任务训练
架构:
  文本 → LLM → pose → cam (纯文本, 与注入无关)
  图 → DINOv2 → injector → EG3D tri-plane (风格/纹理)
  两条路径完全独立 (Phase 1 + Phase 2)

训练策略:
  身份固定 zid → 同一人不同角度
  GT camera 渲染 → 像素对齐 L1 loss
  injector 不依赖 ws/cam, 只学 DINOv2→tri-plane 映射
"""
import os, sys, torch, torch.nn as nn, torch.nn.functional as F, torchvision

sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, get_multimodal_features, pose_6d_to_camera_params

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class LLMGuidedInjector(nn.Module):
    """DINOv2 多尺度特征 → 每个 block 接收全部 3 层"""
    def __init__(self, eg3d, dino_dim=1024, num_layers=3):
        super().__init__()
        self.eg3d = eg3d
        synth = eg3d.backbone.synthesis
        self.block_resolutions = synth.block_resolutions

        # 每个 block: 每层 DINOv2 一个独立 3×3 Conv, 再加 1×1 融合
        self.inject_3x3 = nn.ModuleDict()  # dino_dim → dino_dim//2, 3×3
        self.inject_1x1 = nn.ModuleDict()  # dino_dim//2 * num_layers → ch, 1×1
        for res in self.block_resolutions:
            block = getattr(synth, f'b{res}')
            ch = block.conv1.out_channels
            # 3 个独立的 3×3 卷积 (每层 DINOv2 一个)
            self.inject_3x3[str(res)] = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dino_dim, dino_dim // 2, 3, padding=1),
                    nn.LeakyReLU(0.2),
                ) for _ in range(num_layers)
            ])
            # 1×1 融合: 3×512 → ch
            self.inject_1x1[str(res)] = nn.Conv2d(dino_dim // 2 * num_layers, ch, 1)
            nn.init.zeros_(self.inject_1x1[str(res)].weight)
            nn.init.zeros_(self.inject_1x1[str(res)].bias)
            self.add_module(f'merge_{res}', self.inject_1x1[str(res)])

    def forward(self, ws, deepstack_spatial):
        ds_maps = []
        for f in deepstack_spatial:
            ds_maps.append(f.permute(1, 0).reshape(1, -1, 16, 16))

        synth = self.eg3d.backbone.synthesis
        x = img = None
        w_idx = 0
        for res in self.block_resolutions:
            block = getattr(synth, f'b{res}')
            ws_block = ws[:, w_idx:w_idx + block.num_conv + block.num_torgb]
            w_idx += block.num_conv
            x, img = block(x, img, ws_block)

            # 所有 3 层 DINOv2 各自过 3×3, 拼起来再 1×1 融合
            feats = []
            for i, conv3 in enumerate(self.inject_3x3[str(res)]):
                feat = ds_maps[i]
                if feat.shape[2:] != x.shape[2:]:
                    feat = F.interpolate(feat, size=x.shape[2:], mode='bilinear', align_corners=False)
                feats.append(conv3(feat))
            merged = self.inject_1x1[str(res)](torch.cat(feats, dim=1))
            x = x + merged

        return img


# ===== 训练 =====
bridge = QwenEG3DBridge()
for p in bridge.parameters():
    p.requires_grad_(False)

bridge.injector = LLMGuidedInjector(bridge.eg3d).to('cuda')
for p in bridge.injector.parameters():
    p.requires_grad_(True)

params = list(bridge.injector.parameters())
opt = torch.optim.Adam(params, lr=5e-4, betas=(0.9, 0.99))

data = torch.load("/root/edit_data/dataset.pt")
samples = data['samples']
print(f"samples: {len(samples)}, params: {sum(p.numel() for p in params):,}")

os.makedirs("/root/output_injector", exist_ok=True)

# checkpoint 恢复
ckpt_path = "/root/Qwen3-VL/checkpoints/injector_final.pt"
start_step = 0
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cuda')
    bridge.injector.load_state_dict(ckpt['injector'])
    opt.load_state_dict(ckpt['optimizer'])
    start_step = ckpt['step']
    print(f"恢复训练: step {start_step}")

for step in range(start_step, 5000):
    opt.zero_grad()

    s = samples[step % len(samples)]
    img_path = f"/root/edit_data/images/{s['image']}"
    text, zid = s['text'], s['zid']

    gt = torchvision.io.read_image(img_path).float() / 127.5 - 1
    gt = gt.unsqueeze(0).cuda()

    with torch.no_grad():
        ds, h = get_multimodal_features(bridge.qwen, bridge.processor, text, img_path,
                                        dino=bridge.dino, dino_processor=bridge.dino_processor)
        raw_6d = torch.tensor(s['raw_6d'], dtype=torch.float32, device='cuda').unsqueeze(0)
        cam = pose_6d_to_camera_params(raw_6d)
        torch.manual_seed(zid)
        ws = bridge.eg3d.mapping(torch.randn(1, bridge.eg3d.z_dim, device='cuda'), cam, truncation_psi=0.7)

    bridge.eg3d._last_planes = bridge.injector(ws, ds)
    pred = bridge.eg3d.synthesis(ws, cam, neural_rendering_resolution=128, use_cached_backbone=True)['image_raw']

    loss = F.l1_loss(pred, F.interpolate(gt, size=(128, 128), mode='bilinear', align_corners=False))
    loss.backward()
    nn.utils.clip_grad_norm_(params, 5.0)
    opt.step()

    # 每 500 步保存 checkpoint + 可视化
    if (step + 1) % 500 == 0:
        vram = torch.cuda.memory_allocated() / 1024**3
        torch.save({
            'injector': bridge.injector.state_dict(),
            'optimizer': opt.state_dict(),
            'step': step + 1,
        }, ckpt_path)
        with torch.no_grad():
            full = bridge.eg3d.synthesis(ws, cam, use_cached_backbone=True)['image']
        torchvision.utils.save_image((full[0]*0.5+0.5).clamp(0,1), f"/root/output_injector/step_{step+1}.png")
        print(f"step {step+1}: L1={loss.item():.5f} VRAM={vram:.2f}GB", flush=True)

print("Done!")
