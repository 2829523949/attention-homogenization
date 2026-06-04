"""
Injector v2 — 论文用复现
架构: 
  Image → DINOv2 → injector → EG3D tri-plane (视觉路径)
  Text → LLM → pose → camera (Phase 1 文本路径, 仅用于获取 ws/cam)
两条路径独立。
输出: loss CSV + 可视化 + 结构化的论文用结果
"""

import os, sys, json, csv, math
from datetime import datetime

sys.path.insert(0, "/root/Qwen3-VL")
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

# ===== 1. Bridge (Phase 1, 仅用于 eg3d + qwen + pose) =====
print("[1/5] Loading Phase 1 bridge...")
from bridge import QwenEG3DBridge, pose_6d_to_camera_params, get_llm_hidden

bridge = QwenEG3DBridge()
for p in bridge.parameters():
    p.requires_grad_(False)
bridge.eval()
print("  Bridge loaded")

# ===== 2. DINOv2 =====
print("[2/5] Loading DINOv2-large...")
from transformers import AutoImageProcessor, AutoModel

dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large",
    local_files_only=True)
dino = AutoModel.from_pretrained("facebook/dinov2-large",
    local_files_only=True).cuda().eval()
for p in dino.parameters():
    p.requires_grad_(False)
print(f"  DINOv2 params: {sum(p.numel() for p in dino.parameters())/1e6:.0f}M")

# DINOv2-large: patch14, 224x224 → 16x16 patches, hidden=1024
# We want 3 intermediate layers (deepstack)
DINO_LAYERS = [6, 12, 18]  # 3 layers from the 24-layer ViT

def get_dino_features(img_path):
    """Extract DINOv2 deep features from image.
    Returns list of 3 spatial feature maps, each [256, 1024] = [16*16, 1024]
    """
    from PIL import Image
    pil = Image.open(img_path).convert("RGB")
    inputs = dino_processor(pil, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = dino(**inputs, output_hidden_states=True)
        # outputs.hidden_states is tuple of (patch_embed, layer1, ..., layer24)
        # Each: [1, 257, 1024] where 257 = 1 cls + 256 patches
        feats = []
        for layer_idx in DINO_LAYERS:
            h = outputs.hidden_states[layer_idx]  # [1, 257, 1024]
            h = h[:, 1:, :]  # remove CLS → [1, 256, 1024]
            h = h.squeeze(0)  # [256, 1024]
            feats.append(h)
    return feats  # list of 3 tensors, each [256, 1024]


# ===== 3. Injector (LLMGuidedInjector from v2) =====
print("[3/5] Building LLMGuidedInjector...")

class LLMGuidedInjector(nn.Module):
    """DINOv2 多尺度特征 → 每个 block 注入全部 3 层"""
    def __init__(self, eg3d, dino_dim=1024, num_layers=3):
        super().__init__()
        self.eg3d = eg3d
        synth = eg3d.backbone.synthesis
        self.block_resolutions = synth.block_resolutions
        
        self.inject_3x3 = nn.ModuleDict()
        self.inject_1x1 = nn.ModuleDict()
        for res in self.block_resolutions:
            block = getattr(synth, f'b{res}')
            ch = block.conv1.out_channels
            self.inject_3x3[str(res)] = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dino_dim, dino_dim // 2, 3, padding=1),
                    nn.LeakyReLU(0.2),
                ) for _ in range(num_layers)
            ])
            self.inject_1x1[str(res)] = nn.Conv2d(dino_dim // 2 * num_layers, ch, 1)
            nn.init.zeros_(self.inject_1x1[str(res)].weight)
            nn.init.zeros_(self.inject_1x1[str(res)].bias)
    
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
            
            feats = []
            for i, conv3 in enumerate(self.inject_3x3[str(res)]):
                feat = ds_maps[i]
                if feat.shape[2:] != x.shape[2:]:
                    feat = F.interpolate(feat, size=x.shape[2:], mode='bilinear', align_corners=False)
                feats.append(conv3(feat))
            merged = self.inject_1x1[str(res)](torch.cat(feats, dim=1))
            x = x + merged
        
        return img


injector = LLMGuidedInjector(bridge.eg3d).cuda()
for p in injector.parameters():
    p.requires_grad_(True)

n_params = sum(p.numel() for p in injector.parameters())
print(f"  Injector params: {n_params:,}")

# ===== 4. Data =====
print("[4/5] Loading data...")
data = torch.load("/root/edit_data/dataset.pt")
samples = data['samples']
zids = set(s['zid'] for s in samples)
print(f"  Total samples: {len(samples)}, unique identities: {len(zids)}")

# Train/val split by identity
all_zids = sorted(zids)
val_zids = set(all_zids[:3])  # 3 identities for val
train_samples = [s for s in samples if s['zid'] not in val_zids]
val_samples = [s for s in samples if s['zid'] in val_zids]
print(f"  Train: {len(train_samples)}, Val: {len(val_samples)} (val_zids={sorted(val_zids)})")
    # Precompute DINOv2 features for all samples
print("  Precomputing DINOv2 features... This may take a while.")
dino_cache = {}
for s in samples:
    img_key = s["image"]
    if img_key not in dino_cache:
        dino_cache[img_key] = get_dino_features(f"/root/edit_data/images/{img_key}")
print(f"  Cached DINOv2 features for {len(dino_cache)} unique images")

os.makedirs("/root/output_injector_paper", exist_ok=True)

# ===== 5. Training =====
print("[5/5] Starting training...")
optimizer = torch.optim.Adam(injector.parameters(), lr=5e-4, betas=(0.9, 0.99))
ckpt_path = "/root/Qwen3-VL/checkpoints/injector_final.pt"
start_step = 0

# Try resume
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cuda')
    injector.load_state_dict(ckpt['injector'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_step = ckpt['step']
    print(f"  Resumed from step {start_step}")

# CSV log
csv_path = "/root/output_injector_paper/training_log.csv"
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['step', 'train_loss', 'val_loss', 'lr', 'vram_gb'])
csv_file.flush()

MAX_STEPS = 5000
LOG_EVERY = 50        # log CSV every 50 steps
VIS_EVERY = 500       # save visualization every 500 steps
CKPT_EVERY = 1000     # save checkpoint every 1000 steps

pbar = tqdm(range(start_step, MAX_STEPS), desc="Injector", initial=start_step, total=MAX_STEPS)

for step in pbar:
    optimizer.zero_grad()
    
    s = train_samples[step % len(train_samples)]
    img_path = f"/root/edit_data/images/{s['image']}"
    text, zid = s['text'], s['zid']
    
    # GT image
    gt = torchvision.io.read_image(img_path).float() / 127.5 - 1
    gt = gt.unsqueeze(0).cuda()
    
    with torch.no_grad():
        # DINOv2 features
        ds = dino_cache[s["image"]]
        # ws from the known GT pose
        raw_6d = torch.tensor(s['raw_6d'], dtype=torch.float32, device='cuda').unsqueeze(0)
        cam = pose_6d_to_camera_params(raw_6d)
        torch.manual_seed(zid)
        ws = bridge.eg3d.mapping(torch.randn(1, bridge.eg3d.z_dim, device='cuda'), cam, truncation_psi=0.7)
    
    # Inject
    bridge.eg3d._last_planes = injector(ws, ds)
    pred = bridge.eg3d.synthesis(ws, cam, neural_rendering_resolution=128,
                                 use_cached_backbone=True)['image_raw']
    
    loss = F.l1_loss(pred, F.interpolate(gt, size=(128, 128), mode='bilinear', align_corners=False))
    loss.backward()
    nn.utils.clip_grad_norm_(injector.parameters(), 5.0)
    optimizer.step()
    
    # --- Logging ---
    log_this_step = (step + 1) % LOG_EVERY == 0 or step == start_step
    
    if log_this_step:
        # Val loss
        with torch.no_grad():
            val_losses = []
            for vs in val_samples[:20]:  # subset for speed
                v_path = f"/root/edit_data/images/{vs['image']}"
                v_gt = torchvision.io.read_image(v_path).float() / 127.5 - 1
                v_gt = v_gt.unsqueeze(0).cuda()
                
                v_ds = dino_cache[vs["image"]]
                v_raw = torch.tensor(vs['raw_6d'], dtype=torch.float32, device='cuda').unsqueeze(0)
                v_cam = pose_6d_to_camera_params(v_raw)
                torch.manual_seed(vs['zid'])
                v_ws = bridge.eg3d.mapping(torch.randn(1, bridge.eg3d.z_dim, device='cuda'), v_cam, truncation_psi=0.7)
                
                bridge.eg3d._last_planes = injector(v_ws, v_ds)
                v_pred = bridge.eg3d.synthesis(v_ws, v_cam, neural_rendering_resolution=128,
                                               use_cached_backbone=True)['image_raw']
                v_l = F.l1_loss(v_pred, F.interpolate(v_gt, size=(128, 128), mode='bilinear'))
                val_losses.append(v_l.item())
            
            val_loss = sum(val_losses) / len(val_losses)
        
        vram = torch.cuda.memory_allocated() / 1024**3
        csv_writer.writerow([step + 1, loss.item(), val_loss, optimizer.param_groups[0]['lr'], vram])
        csv_file.flush()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'val': f'{val_loss:.4f}', 'vram': f'{vram:.1f}GB'})
    
    # --- Visualization ---
    if (step + 1) % VIS_EVERY == 0:
        with torch.no_grad():
            full = bridge.eg3d.synthesis(ws, cam, use_cached_backbone=True)['image']
        torchvision.utils.save_image(
            (full[0]*0.5+0.5).clamp(0,1),
            f"/root/output_injector_paper/step_{step+1}.png"
        )
        # Also save GT comparison
        torchvision.utils.save_image(
            (gt[0]*0.5+0.5).clamp(0,1),
            f"/root/output_injector_paper/gt_{step+1}.png"
        )
    
    # --- Checkpoint ---
    if (step + 1) % CKPT_EVERY == 0:
        torch.save({
            'injector': injector.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step + 1,
        }, ckpt_path)

# Done
csv_file.close()
print(f"\nTraining complete! {MAX_STEPS} steps.")
print(f"Log: {csv_path}")
print(f"Outputs: /root/output_injector_paper/")

# ===== 6. Summary =====
print("\n[Summary] Generating results...")
# Read log and report
with open(csv_path) as f:
    lines = f.readlines()
print(f"  Log entries: {len(lines)-1}")
if len(lines) > 1:
    last = lines[-1].strip().split(',')
    print(f"  Final train loss: {last[1]}")
    print(f"  Final val loss: {last[2]}")

# Save structured results for paper
results = {
    "experiment": "Injector v2 (LLMGuidedInjector)",
    "architecture": "DINOv2-large 3-layer deepstack → multi-res injection → EG3D",
    "dataset": {
        "total_samples": len(samples),
        "unique_identities": len(zids),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
    },
    "training": {
        "steps": MAX_STEPS,
        "optimizer": "Adam lr=5e-4",
        "loss": "L1 @ 128x128",
        "injector_params": n_params,
    },
    "status": "completed",
    "conclusion": "Injector trained. Whether visual fusion succeeds depends on qualitative inspection and downstream pose accuracy.",
}
with open("/root/output_injector_paper/results.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results: /root/output_injector_paper/results.json")
print("Done!")
