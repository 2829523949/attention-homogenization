"""
混合训练: 固定模板(×5权重) + 多样化文本 → PoseMappingLayer → 6D
策略: 50%模板提供高精度锚点, 50%变体提供泛化能力
"""
import os, sys, math, random, json
import torch, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, get_llm_hidden, pose_6d_to_camera_params


class TextPoseDataset:
    """混合数据集: 固定模板(高权重) + 多样化文本(泛化)"""

    # 22个基础视角模板
    VIEWPOINTS = [
        (-3.14,  0.0,   "背面"),
        (-2.36, -0.35,  "左后方俯视"),  (-2.36, 0.0, "左后方平视"),  (-2.36, 0.35, "左后方仰视"),
        (-1.57, -0.4,   "左侧90度俯视"), (-1.57, 0.0, "左侧90度平视"), (-1.57, 0.4, "左侧90度仰视"),
        (-0.79, -0.35,  "左侧45度俯视"), (-0.79, 0.0, "左侧45度平视"), (-0.79, 0.35, "左侧45度仰视"),
        ( 0.00, -0.35,  "正面俯视"),      ( 0.00, 0.0, "正面平视"),      ( 0.00, 0.35, "正面仰视"),
        ( 0.79, -0.35,  "右侧45度俯视"), ( 0.79, 0.0, "右侧45度平视"), ( 0.79, 0.35, "右侧45度仰视"),
        ( 1.57, -0.4,   "右侧90度俯视"), ( 1.57, 0.0, "右侧90度平视"), ( 1.57, 0.4, "右侧90度仰视"),
        ( 2.36, -0.35,  "右后方俯视"),   ( 2.36, 0.0, "右后方平视"),   ( 2.36, 0.35, "右后方仰视"),
    ]

    def __init__(self, variant_path="/root/Qwen3-VL/text_variants.json", val_ratio=0.15):
        # 加载多样化文本
        with open(variant_path) as f:
            all_variants = json.load(f)
        print(f"加载 {len(all_variants)} 条多样化文本")

        # 训练/验证切分: 多样化文本按文本去重后切分
        base_template_names = {v[2] for v in self.VIEWPOINTS}
        filtered = [d for d in all_variants if d['text'] not in base_template_names]
        seen = set()
        unique = []
        for d in filtered:
            key = (d['text'], round(d['yaw'],2), round(d['pitch_offset'],2))
            if key not in seen:
                seen.add(key); unique.append(d)

        # 按文本切分
        text_map = {}
        for d in unique:
            text_map.setdefault(d['text'], []).append(d)
        all_texts = list(text_map.keys())
        random.shuffle(all_texts)
        n_val = max(1, int(len(all_texts) * val_ratio))
        val_set = set(all_texts[:n_val])
        train_set = set(all_texts[n_val:])

        diverse_train = [d for d in unique if d['text'] in train_set]
        diverse_val   = [d for d in unique if d['text'] in val_set]
        print(f"多样化: 训练 {len(diverse_train)}条, 验证 {len(diverse_val)}条")

        # 构建混合池: 固定模板(重复5x) + 多样化文本
        self.pool = []

        # 模板 × 5 (高权重锚点)
        for _ in range(5):
            for yaw, pitch_off, text in self.VIEWPOINTS:
                self.pool.append({'text': text, 'yaw': yaw, 'pitch_offset': pitch_off, 'is_template': True})

        # 多样化文本 (训练集)
        for d in diverse_train:
            self.pool.append({'text': d['text'], 'yaw': d['yaw'],
                              'pitch_offset': d['pitch_offset'], 'is_template': False})

        print(f"混合池: {len(self.pool)} 条 (模板{5*22}条×5 + 多样化{len(diverse_train)}条)")

        # 验证集 (多样化文本)
        self.val_data = []
        for item in diverse_val:
            pitch = math.pi / 2 + item['pitch_offset']
            raw_6d = torch.tensor([[item['yaw'], pitch, 2.7, 18.8, 0.0, 0.2]], dtype=torch.float32)
            self.val_data.append({
                'text': item['text'], 'raw_6d_gt': raw_6d,
                'cam_gt': pose_6d_to_camera_params(raw_6d),
                'yaw_gt': item['yaw'], 'pitch_gt': pitch,
            })

        # 验证集 (22模板, 用于监控模板精度不下降)
        self.val_templates = []
        for yaw, pitch_off, text in self.VIEWPOINTS:
            pitch = math.pi / 2 + pitch_off
            raw_6d = torch.tensor([[yaw, pitch, 2.7, 18.8, 0.0, 0.2]], dtype=torch.float32)
            self.val_templates.append({
                'text': text, 'raw_6d_gt': raw_6d,
                'cam_gt': pose_6d_to_camera_params(raw_6d),
                'yaw_gt': yaw, 'pitch_gt': pitch,
            })

        self.num_epoch_samples = len(diverse_train) + 5 * len(self.VIEWPOINTS)
        print(f"每轮 {self.num_epoch_samples} 步")
        print(f"验证: 多样化{len(self.val_data)}条 + 模板{len(self.val_templates)}条")

    def get_train_batch(self, n=1):
        items = random.choices(self.pool, k=n)
        texts, r6s, cams = [], [], []
        for item in items:
            yaw = item['yaw'] + random.uniform(-0.26, 0.26)
            pitch_offset = item['pitch_offset'] + random.uniform(-0.17, 0.17)
            pitch = math.pi / 2 + pitch_offset
            radius = random.uniform(2.5, 3.0)
            fov = random.uniform(16, 22)
            raw_6d = torch.tensor([[yaw, pitch, radius, fov, 0.0, 0.2]], dtype=torch.float32)
            texts.append(item['text'])
            r6s.append(raw_6d)
            cams.append(pose_6d_to_camera_params(raw_6d))
        return texts, torch.cat(r6s).cuda(), torch.cat(cams).cuda()

    def get_positive_pair(self):
        """返回同一视角的2种不同说法 (用于对比学习)"""
        vp = random.choice(self.VIEWPOINTS)
        yaw, pitch_off = vp[0], vp[1]
        pool_items = [d for d in self.pool if abs(d['yaw'] - yaw) < 0.01
                      and abs(d['pitch_offset'] - pitch_off) < 0.01]
        if len(pool_items) >= 2:
            return random.sample(pool_items, 2)
        return None, None


def train():
    print("=" * 60)
    print("混合训练: 模板(×5) + 多样化文本 → PoseMappingLayer")
    print("=" * 60)

    bridge = QwenEG3DBridge()

    # 加载预训练 PoseMappingLayer
    ckpt_dir = "/root/Qwen3-VL/checkpoints"
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('text_only_') and f.endswith('.pt')],
                   key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if ckpts:
        ckpt = torch.load(os.path.join(ckpt_dir, ckpts[-1]), map_location='cuda')
        bridge.pose.load_state_dict(ckpt['pose'])
        print(f"已加载 pose: {ckpts[-1]}")

    # 冻结 fusion 和 adapter, 训练 pose
    for p in bridge.fusion.parameters(): p.requires_grad = False
    for p in bridge.adapter.parameters(): p.requires_grad = False
    for p in bridge.pose.parameters(): p.requires_grad = True

    dataset = TextPoseDataset()
    params = list(bridge.pose.parameters())
    optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)
    num_epochs = 60
    print(f"可训练参数: {sum(p.numel() for p in params):,} | lr={5e-5}→{1e-5}")

    for epoch in range(num_epochs):
        total_loss = total_contra = 0.0
        pbar = tqdm(range(dataset.num_epoch_samples), desc=f"Epoch {epoch+1}/{num_epochs}")
        for step in pbar:
            optimizer.zero_grad()

            # 主任务: pose 预测
            texts, r6, cam = dataset.get_train_batch(n=1)
            with torch.no_grad():
                hidden = get_llm_hidden(bridge.qwen, bridge.processor, texts[0])
            pred_6d = bridge.pose(hidden.unsqueeze(0))
            pred_cam = pose_6d_to_camera_params(pred_6d)
            loss_pose = F.mse_loss(pred_6d, r6) + F.mse_loss(pred_cam, cam)
            total_loss += loss_pose.item()

            # 每3步: 对比正则化 (拉近同视角不同文本的 hidden)
            loss_info = f"pose={loss_pose.item():.4f}"
            if step % 3 == 0:
                t1, t2 = dataset.get_positive_pair()
                if t1 and t2:
                    with torch.no_grad():
                        h1 = get_llm_hidden(bridge.qwen, bridge.processor, t1['text'])
                        h2 = get_llm_hidden(bridge.qwen, bridge.processor, t2['text'])
                    loss_contra = 0.1 * (1 - F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)))
                    total_contra += loss_contra.item()
                    loss_info += f" contra={loss_contra.item():.4f}"
                    loss_pose = loss_pose + loss_contra  # 合并到总 loss

            loss_pose.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            pbar.set_postfix({'loss': loss_info})

        scheduler.step()
        avg_loss = total_loss / dataset.num_epoch_samples

        if (epoch + 1) % 5 == 0:
            path = f"/root/Qwen3-VL/checkpoints/mix_epoch_{epoch+1:03d}.pt"
            torch.save({'epoch': epoch+1, 'pose': bridge.pose.state_dict(),
                        'optimizer': optimizer.state_dict()}, path)

            bridge.eval()
            # 验证多样化文本
            y_errs, p_errs = [], []
            with torch.no_grad():
                for v in dataset.val_data:
                    hidden = get_llm_hidden(bridge.qwen, bridge.processor, v['text'])
                    p6d = bridge.pose(hidden.unsqueeze(0))
                    y_errs.append(abs(p6d[0, 0].item() - v['yaw_gt']))
                    p_errs.append(abs(p6d[0, 1].item() - v['pitch_gt']))
            # 验证模板精度
            ty_errs, tp_errs = [], []
            with torch.no_grad():
                for v in dataset.val_templates:
                    hidden = get_llm_hidden(bridge.qwen, bridge.processor, v['text'])
                    p6d = bridge.pose(hidden.unsqueeze(0))
                    ty_errs.append(abs(p6d[0, 0].item() - v['yaw_gt']))
                    tp_errs.append(abs(p6d[0, 1].item() - v['pitch_gt']))
            bridge.train()
            print(f"  val 多样: Δy={np.mean(y_errs):.3f} Δp={np.mean(p_errs):.3f}")
            print(f"  val 模板: Δy={np.mean(ty_errs):.3f} Δp={np.mean(tp_errs):.3f}")
            for i, v in enumerate(dataset.val_data[:3]):
                hidden = get_llm_hidden(bridge.qwen, bridge.processor, v['text'])
                p6d = bridge.pose(hidden.unsqueeze(0))
                print(f"  [{i+1}] '{v['text']}' → ({p6d[0,0]:+.2f},{p6d[0,1]:+.2f}) gt=({v['yaw_gt']:+.2f},{v['pitch_gt']:+.2f})")
            print(f"  saved: {path} | loss: {avg_loss:.4f}\n")

    print("混合训练完成!")


if __name__ == "__main__":
    train()
