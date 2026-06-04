"""
English version: Template(×5) + diverse text → Pose
All text in English for fair comparison across encoders (CLIP/BERT/VLM/XLM).
"""
import os, sys, math, random, json, torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, get_llm_hidden, pose_6d_to_camera_params


# ============================================================
# Loss
# ============================================================
def circular_mse(pred, target):
    diff = (pred - target) % (2 * math.pi)
    diff = torch.min(diff, 2 * math.pi - diff)
    return (diff ** 2).mean()


# ============================================================
# English diverse text generator
# ============================================================
def generate_diverse_text_en(yaw, pitch_offset):
    """Generate English descriptions for any (yaw, pitch_offset)"""
    abs_yaw = abs(yaw)
    
    # Horizontal
    if abs_yaw < 0.08:
        dir_phrases = [
            "front view", "facing forward", "looking straight ahead",
            "centered face", "direct front", "straight on view"
        ]
    elif abs_yaw < 0.35:
        side = "left" if yaw > 0 else "right"
        dir_phrases = [
            f"{side} 15 degrees", f"{side} slightly turned", 
            f"slight {side} angle", f"mild {side} profile",
            f"a bit to the {side}", f"{side} quarter profile"
        ]
    elif abs_yaw < 0.65:
        side = "left" if yaw > 0 else "right"
        dir_phrases = [
            f"{side} 30 degrees", f"{side} medium angle",
            f"moderate {side} turn", f"{side} half profile",
            f"thirty degrees to the {side}"
        ]
    else:
        side = "left" if yaw > 0 else "right"
        dir_phrases = [
            f"{side} 45 degrees", f"{side} strong profile",
            f"steep {side} angle", f"three-quarter {side} view",
            f"forty-five degrees {side}"
        ]
    
    # Vertical
    if pitch_offset < -0.3:
        pit_phrases = [
            "looking down", "from above", "top angle", 
            "camera looking down", "high angle shot", "overlooking"
        ]
    elif pitch_offset > 0.3:
        pit_phrases = [
            "looking up", "from below", "low angle",
            "camera looking up", "upward view", "view from below"
        ]
    else:
        pit_phrases = [
            "eye level", "straight ahead", "horizontal view",
            "normal height", "at eye level", "forward gaze"
        ]
    
    dir_p = random.choice(dir_phrases)
    pit_p = random.choice(pit_phrases)
    
    templates = [f"{dir_p}, {pit_p}"]
    if random.random() > 0.5:
        templates.append(f"{dir_p} {pit_p}")
    if random.random() > 0.7:
        templates.append(f"face seen from {dir_p}, {pit_p}")
    
    return random.choice(templates)


# ============================================================
# English Templates (21 = 7 yaw × 3 pitch)
# ============================================================
EN_TEMPLATES_RAW = [
    # 0° front × 3 pitch
    ( 0.00, -0.35, "front view, looking down"),
    ( 0.00,  0.0,  "front view, eye level"),
    ( 0.00,  0.35, "front view, looking up"),
    # left 15°
    (-0.26, -0.35, "left 15 degrees, looking down"),
    (-0.26,  0.0,  "left 15 degrees, eye level"),
    (-0.26,  0.35, "left 15 degrees, looking up"),
    # left 30°
    (-0.52, -0.35, "left 30 degrees, looking down"),
    (-0.52,  0.0,  "left 30 degrees, eye level"),
    (-0.52,  0.35, "left 30 degrees, looking up"),
    # left 45°
    (-0.79, -0.35, "left 45 degrees, looking down"),
    (-0.79,  0.0,  "left 45 degrees, eye level"),
    (-0.79,  0.35, "left 45 degrees, looking up"),
    # right 15°
    ( 0.26, -0.35, "right 15 degrees, looking down"),
    ( 0.26,  0.0,  "right 15 degrees, eye level"),
    ( 0.26,  0.35, "right 15 degrees, looking up"),
    # right 30°
    ( 0.52, -0.35, "right 30 degrees, looking down"),
    ( 0.52,  0.0,  "right 30 degrees, eye level"),
    ( 0.52,  0.35, "right 30 degrees, looking up"),
    # right 45°
    ( 0.79, -0.35, "right 45 degrees, looking down"),
    ( 0.79,  0.0,  "right 45 degrees, eye level"),
    ( 0.79,  0.35, "right 45 degrees, looking up"),
]
EN_TEMPLATE_TEXTS = {t[2] for t in EN_TEMPLATES_RAW}


# ============================================================
# English Dataset
# ============================================================
class MixedPoseDatasetEn:
    """
    English-only mixed dataset.
    Templates + real-time generated diverse text (no JSON file needed).
    Text-based train/val split (no leakage).
    """
    def __init__(self, val_ratio=0.15, template_weight=5, extra_diverse_per_epoch=500):
        self.template_weight = template_weight
        self.extra_diverse_per_epoch = extra_diverse_per_epoch
        self.templates = list(EN_TEMPLATES_RAW)
        
        # Generate a fixed set of diverse texts for deterministic val split
        random.seed(42)
        diverse_pool = []
        for _ in range(500):
            yaw = random.uniform(-0.79, 0.79)
            pitch_off = random.uniform(-0.6, 0.6)
            text = generate_diverse_text_en(yaw, pitch_off)
            diverse_pool.append({
                'text': text, 'yaw': round(yaw, 2),
                'pitch_offset': round(pitch_off, 2)
            })
        random.seed(None)  # reset seed
        
        # Deduplicate by text
        seen_texts = set()
        unique = []
        for d in diverse_pool:
            if d['text'] not in seen_texts and d['text'] not in EN_TEMPLATE_TEXTS:
                seen_texts.add(d['text'])
                unique.append(d)
        
        # Train/val split by text
        unique_texts = list({d['text'] for d in unique})
        random.shuffle(unique_texts)
        n_val = max(1, int(len(unique_texts) * val_ratio))
        val_texts = set(unique_texts[:n_val])
        train_texts = set(unique_texts[n_val:])
        
        self.diverse_train = [d for d in unique if d['text'] in train_texts]
        self.diverse_val = [d for d in unique if d['text'] in val_texts]
        
        print(f"[EN Dataset] Templates: {len(self.templates)}")
        print(f"[EN Dataset] Diverse train: {len(self.diverse_train)}, val: {len(self.diverse_val)}")
        
        # Build val sets
        self.val_data = []
        self.val_templates = []
        
        for item in self.diverse_val:
            pitch = math.pi / 2 + item['pitch_offset']
            raw_6d = torch.tensor([[item['yaw'], pitch, 2.7, 18.8, 0.0, 0.2]], dtype=torch.float32)
            self.val_data.append({
                'text': item['text'],
                'raw_6d_gt': raw_6d,
                'cam_gt': pose_6d_to_camera_params(raw_6d),
                'yaw_gt': item['yaw'],
                'pitch_gt': pitch,
                'pitch_offset': item['pitch_offset'],
            })
        
        for yaw, pitch_off, text in self.templates:
            pitch = math.pi / 2 + pitch_off
            raw_6d = torch.tensor([[yaw, pitch, 2.7, 18.8, 0.0, 0.2]], dtype=torch.float32)
            self.val_templates.append({
                'text': text,
                'raw_6d_gt': raw_6d,
                'cam_gt': pose_6d_to_camera_params(raw_6d),
                'yaw_gt': yaw,
                'pitch_gt': pitch,
                'pitch_offset': pitch_off,
            })
        
        print(f"[EN Dataset] Val: {len(self.val_data)} diverse + {len(self.val_templates)} templates")
        self._rebuild_train_pool()
    
    def _rebuild_train_pool(self):
        self.pool = []
        for _ in range(self.template_weight):
            for yaw, pitch_off, text in self.templates:
                self.pool.append({
                    'text': text, 'yaw': yaw,
                    'pitch_offset': pitch_off, 'is_template': True,
                })
        for d in self.diverse_train:
            self.pool.append({
                'text': d['text'], 'yaw': d['yaw'],
                'pitch_offset': d['pitch_offset'], 'is_template': False,
            })
        for _ in range(self.extra_diverse_per_epoch):
            yaw = random.uniform(-0.79, 0.79)
            pitch_offset = random.uniform(-0.6, 0.6)
            text = generate_diverse_text_en(yaw, pitch_offset)
            self.pool.append({
                'text': text, 'yaw': round(yaw, 2),
                'pitch_offset': round(pitch_offset, 2), 'is_template': False,
            })
        random.shuffle(self.pool)
        self.num_epoch_samples = len(self.pool)
        print(f"[EN Dataset] Train pool: {len(self.pool)} samples")
    
    def get_train_batch(self, n=1):
        items = random.choices(self.pool, k=n)
        texts, r6s, cams = [], [], []
        for item in items:
            yaw = item['yaw'] + random.uniform(-0.06, 0.06)
            pitch_offset = item['pitch_offset'] + random.uniform(-0.06, 0.06)
            pitch = math.pi / 2 + pitch_offset
            radius = random.uniform(2.5, 3.0)
            fov = random.uniform(16, 22)
            lx = random.uniform(-0.05, 0.05)
            lz = random.uniform(0.15, 0.25)
            raw_6d = torch.tensor([[yaw, pitch, radius, fov, lx, lz]], dtype=torch.float32)
            texts.append(item['text'])
            r6s.append(raw_6d)
            cams.append(pose_6d_to_camera_params(raw_6d))
        return texts, torch.cat(r6s).cuda(), torch.cat(cams).cuda()


# ============================================================
# VLM Training (English) — reuse same function but with EN dataset
# ============================================================
def train_vlm():
    print("=" * 60)
    print("VLM Training (English Dataset)")
    print("=" * 60)
    
    bridge = QwenEG3DBridge()
    ckpt_dir = "/root/Qwen3-VL/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    for p in bridge.qwen.parameters(): p.requires_grad = False
    for p in bridge.pose.parameters(): p.requires_grad = True
    params = list(bridge.pose.parameters())
    n_params = sum(p.numel() for p in params)
    print(f"Trainable: {n_params:,}")
    
    optimizer = torch.optim.AdamW(params, lr=3e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6)
    num_epochs = 80
    
    for epoch in range(num_epochs):
        bridge.train()
        dataset = MixedPoseDatasetEn(template_weight=5, extra_diverse_per_epoch=500)
        total_loss = 0.0
        smooth_loss = 0.0
        
        pbar = tqdm(range(dataset.num_epoch_samples), desc=f"Epoch {epoch+1}/{num_epochs}")
        for step in pbar:
            optimizer.zero_grad()
            texts, r6, cam = dataset.get_train_batch(n=1)
            with torch.no_grad():
                hidden = get_llm_hidden(bridge.qwen, bridge.processor, texts[0])
            pred_6d = bridge.pose(hidden.unsqueeze(0))
            pred_cam = pose_6d_to_camera_params(pred_6d)
            
            loss_yaw = circular_mse(pred_6d[:, 0], r6[:, 0])
            loss_other = F.mse_loss(pred_6d[:, 1:], r6[:, 1:])
            loss_6d = loss_yaw + loss_other
            loss_cam = F.mse_loss(pred_cam, cam)
            loss = loss_6d + loss_cam
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            smooth_loss = 0.95 * smooth_loss + 0.05 * loss.item() if smooth_loss else loss.item()
            pbar.set_postfix({'loss': f"{smooth_loss:.4f}"})
        
        scheduler.step()
        avg_loss = total_loss / dataset.num_epoch_samples
        
        if (epoch + 1) % 5 == 0:
            bridge.eval()
            y_errs, p_errs = [], []
            with torch.no_grad():
                for v in dataset.val_data:
                    hidden = get_llm_hidden(bridge.qwen, bridge.processor, v['text'])
                    p6d = bridge.pose(hidden.unsqueeze(0))
                    ye = min(abs(p6d[0, 0].item() - v['yaw_gt']), 2 * math.pi - abs(p6d[0, 0].item() - v['yaw_gt']))
                    y_errs.append(ye)
                    p_errs.append(abs(p6d[0, 1].item() - v['pitch_gt']))
            
            path = os.path.join(ckpt_dir, f"vlm_en_epoch_{epoch+1:03d}.pt")
            torch.save({'epoch': epoch + 1, 'pose': bridge.pose.state_dict(), 'optimizer': optimizer.state_dict()}, path)
            
            avg_ye = np.mean(y_errs) * 180 / math.pi
            avg_pe = np.mean(p_errs) * 180 / math.pi
            print(f"[VLM EN Epoch {epoch+1}] loss={avg_loss:.4f}  Δyaw={avg_ye:.2f}°  Δpitch={avg_pe:.2f}°  saved={path}")
            bridge.train()
    
    print("VLM EN training done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', choices=['vlm'], default='vlm')
    args = parser.parse_args()
    if args.train == 'vlm':
        train_vlm()
