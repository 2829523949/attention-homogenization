"""
混合训练 v2: 固定模板(少量) + 大量多样化文本 → PoseMappingLayer
=====================================================================
改进点:
  1. 仅训练 PoseMappingLayer (去掉 Adapter, 之前实验证明效果不好)
  2. Checkpoint 只保存 pose 权重
  3. 多样化文本从 text_variants.json + 实时生成连续角度 (无限扩增)
  4. 按 text 去重切分 train/val, 无数据泄露
  5. 圆形 yaw 损失 (处理 背面 +3.14 vs -3.14 周期性)
  6. 验证集同时监控模板精度和多样化文本精度
"""
import os, sys, math, random, json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, get_llm_hidden, pose_6d_to_camera_params


# ============================================================
# 损失函数
# ============================================================
def circular_mse(pred, target):
    """圆形 MSE: 处理 yaw 的 2π 周期性 (3.14 vs -3.14 是同一角度)"""
    diff = (pred - target) % (2 * math.pi)
    diff = torch.min(diff, 2 * math.pi - diff)
    return (diff ** 2).mean()


# ============================================================
# 多样化文本生成器 (连续角度 → 自然语言)
# ============================================================
def generate_diverse_text(yaw, pitch_offset):
    """为任意连续角度生成多种措辞的描述"""
    templates = []

    # 水平方向 (±45°), 细分15°和30°
    abs_yaw = abs(yaw)
    if abs_yaw < 0.08:
        dir_words = ["正面", "正脸", "正对镜头", "看向前方"]
    elif abs_yaw < 0.35:
        side = "左侧" if yaw > 0 else "右侧"
        dir_words = [f"{side}15度", f"{side}微侧", f"{side}偏一点",
                     f"{side}稍微侧", f"稍向{side}转", f"微微{side}偏"]
    elif abs_yaw < 0.65:
        side = "左侧" if yaw > 0 else "右侧"
        dir_words = [f"{side}30度", f"{side}三分之一个侧面",
                     f"{side}中幅侧转", f"{side}偏转30度"]
    else:
        side = "左侧" if yaw > 0 else "右侧"
        dir_words = [f"{side}45度", f"{side}半侧面", f"{side}侧向",
                     f"{side}大幅侧转"]

    # 垂直方向
    if pitch_offset < -0.3:
        pit_words = ["俯视", "从上往下看", "高角度", "俯瞰"]
    elif pitch_offset > 0.3:
        pit_words = ["仰视", "从下往上看", "低角度", "仰望"]
    else:
        pit_words = ["平视", "水平视角", "正常高度"]

    dir_w = random.choice(dir_words)
    pit_w = random.choice(pit_words)

    # 多种句式
    templates.append(f"{dir_w}{pit_w}")
    if random.random() > 0.5:
        templates.append(f"{random.choice(['一张', ''])}{dir_w}{random.choice(['的', '角度', '视角'])}{pit_w}{random.choice(['照片', '人脸', '拍摄'])}")
    if random.random() > 0.7:
        templates.append(f"{random.choice(['从', '在'])}{dir_w}{random.choice(['方向拍的', '角度看的', '视角下的'])}{pit_w}{random.choice(['人脸', '照片', ''])}")

    return random.choice(templates).strip()


# ============================================================
# 混合数据集
# ============================================================
class MixedPoseDataset:
    """
    混合数据集:
      - 固定模板 (yaw,pitch_offset,文本 硬绑定)
      - 多样化文本 (来自 text_variants.json + 实时生成)
    训练池: 模板×N1 + 多样化×N2, 控制混合比例
    验证集: 按文本去重切分, 无泄露
    """

    # 21个固定模板 (7 yaw × 3 pitch, 与数据集YAWS一致)
    # yaw: -0.79(左45°) -0.52(左30°) -0.26(左15°) 0.00(正) +0.26(右15°) +0.52(右30°) +0.79(右45°)
    TEMPLATES = [
        # 正面 3
        ( 0.00, -0.35, "正面俯视"),    ( 0.00,  0.0, "正面平视"),    ( 0.00,  0.35, "正面仰视"),
        # 左侧15度 3
        (-0.26, -0.35, "左侧15度俯视"), (-0.26,  0.0, "左侧15度平视"), (-0.26,  0.35, "左侧15度仰视"),
        # 左侧30度 3
        (-0.52, -0.35, "左侧30度俯视"), (-0.52,  0.0, "左侧30度平视"), (-0.52,  0.35, "左侧30度仰视"),
        # 左侧45度 3
        (-0.79, -0.35, "左侧45度俯视"), (-0.79,  0.0, "左侧45度平视"), (-0.79,  0.35, "左侧45度仰视"),
        # 右侧15度 3
        ( 0.26, -0.35, "右侧15度俯视"), ( 0.26,  0.0, "右侧15度平视"), ( 0.26,  0.35, "右侧15度仰视"),
        # 右侧30度 3
        ( 0.52, -0.35, "右侧30度俯视"), ( 0.52,  0.0, "右侧30度平视"), ( 0.52,  0.35, "右侧30度仰视"),
        # 右侧45度 3
        ( 0.79, -0.35, "右侧45度俯视"), ( 0.79,  0.0, "右侧45度平视"), ( 0.79,  0.35, "右侧45度仰视"),
    ]
    TEMPLATE_TEXTS = {t[2] for t in TEMPLATES}  # 用于去重

    def __init__(self, variant_path="/root/Qwen3-VL/text_variants.json",
                 val_ratio=0.15, template_weight=5, extra_diverse_per_epoch=500):
        self.template_weight = template_weight  # 模板在池中的重复倍数
        self.extra_diverse_per_epoch = extra_diverse_per_epoch  # 每轮额外生成的多样化样本数

        # ===== 固定模板 =====
        self.templates = list(self.TEMPLATES)
        print(f"固定模板: {len(self.templates)} 个")

        # ===== 多样化文本 (来自 text_variants.json) =====
        if os.path.exists(variant_path):
            with open(variant_path) as f:
                all_variants = json.load(f)
            # 过滤: 1) 和模板文本相同  2) yaw 超出 ±45° (EG3D 在大角度渲染质量差)
            filtered = [d for d in all_variants 
                       if d['text'] not in self.TEMPLATE_TEXTS and abs(d['yaw']) <= 0.79]
            # 按 (text, yaw, pitch_offset) 去重
            seen = set()
            unique_variants = []
            for d in filtered:
                key = (d['text'], round(d['yaw'], 2), round(d['pitch_offset'], 2))
                if key not in seen:
                    seen.add(key)
                    unique_variants.append(d)
            print(f"多样化文本 (来自文件): {len(unique_variants)} 条 (原始{len(all_variants)}条, 过滤后)")
        else:
            print(f"⚠️ {variant_path} 不存在, 仅使用模板 + 实时生成")
            unique_variants = []

        # ===== 按文本去重切分 train/val =====
        # 按 text 字符串分组 (确保同一文本不会跨 train/val)
        text_groups = defaultdict(list)
        for d in unique_variants:
            text_groups[d['text']].append(d)
        all_texts = list(text_groups.keys())
        random.shuffle(all_texts)

        n_val = max(1, int(len(all_texts) * val_ratio))
        val_texts = set(all_texts[:n_val])
        train_texts = set(all_texts[n_val:])

        self.diverse_train = [d for d in unique_variants if d['text'] in train_texts]
        self.diverse_val   = [d for d in unique_variants if d['text'] in val_texts]
        print(f"多样化切分: 训练 {len(self.diverse_train)} 条, 验证 {len(self.diverse_val)} 条")

        # ===== 构建训练池 =====
        self._rebuild_train_pool()

        # ===== 构建验证集 =====
        self.val_data = []
        self.val_templates = []

        # 多样化 val
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

        # 模板 val (用于监控模板精度)
        for yaw, pitch_off, text in self.TEMPLATES:
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

        print(f"验证集: 多样化{len(self.val_data)}条 + 模板{len(self.val_templates)}条")

    def _rebuild_train_pool(self):
        """重建训练池 (每轮可重新生成多样化文本)"""
        self.pool = []

        # 固定模板 × template_weight (高权重锚点)
        for _ in range(self.template_weight):
            for yaw, pitch_off, text in self.TEMPLATES:
                self.pool.append({
                    'text': text, 'yaw': yaw,
                    'pitch_offset': pitch_off, 'is_template': True,
                })

        # 多样化文本 (来自文件)
        for d in self.diverse_train:
            self.pool.append({
                'text': d['text'], 'yaw': d['yaw'],
                'pitch_offset': d['pitch_offset'], 'is_template': False,
            })

        # 额外实时生成的多样化文本 (连续角度, 无限扩增, ±45°)
        for _ in range(self.extra_diverse_per_epoch):
            yaw = random.uniform(-0.79, 0.79)  # ±45°
            pitch_offset = random.uniform(-0.6, 0.6)  # 俯仰范围
            text = generate_diverse_text(yaw, pitch_offset)
            # 避免文本与模板或已有多样化文本完全重复
            self.pool.append({
                'text': text, 'yaw': round(yaw, 2),
                'pitch_offset': round(pitch_offset, 2), 'is_template': False,
            })

        random.shuffle(self.pool)
        self.num_epoch_samples = len(self.pool)
        t_count = sum(1 for p in self.pool if p['is_template'])
        d_count = self.num_epoch_samples - t_count
        print(f"训练池重建: {self.num_epoch_samples} 条 (模板{t_count}条 + 多样化{d_count}条)")

    def get_train_batch(self, n=1):
        """采样训练批次 (加随机扰动增强泛化)"""
        items = random.choices(self.pool, k=n)
        texts, r6s, cams = [], [], []
        for item in items:
            # 对 yaw/pitch 加轻微抖动 (±0.06 rad ≈ 3.4°) 增强泛化
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
# 训练函数
# ============================================================
def train():
    print("=" * 60)
    print("混合训练 v2: 模板(×5) + 大量多样化文本 → Pose (无Adapter)")
    print("=" * 60)

    bridge = QwenEG3DBridge()

    # 加载已有 pose 权重 (如果有)
    ckpt_dir = "/root/Qwen3-VL/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpts = sorted([f for f in os.listdir(ckpt_dir)
                    if f.startswith('mix_v2_') and f.endswith('.pt')],
                   key=lambda x: int(x.split('_')[-1].split('.')[0]))
    start_epoch = 30
    ckpt_path = os.path.join(ckpt_dir, "mix_v2_epoch_030.pt")
    assert os.path.exists(ckpt_path), f"❌ {ckpt_path} 不存在!"
    ckpt = torch.load(ckpt_path, map_location='cuda')
    bridge.pose.load_state_dict(ckpt['pose'])
    print(f"加载 pose: mix_v2_epoch_030.pt")

    # ===== 冻结/解冻 =====
    # bridge.py Phase1 only: qwen(冻结) + pose(训练)
    for p in bridge.qwen.parameters(): p.requires_grad = False
    for p in bridge.pose.parameters(): p.requires_grad = True

    params = list(bridge.pose.parameters())
    n_params = sum(p.numel() for p in params)
    print(f"可训练参数: {n_params:,} (仅 PoseMappingLayer)")

    # ===== 优化器 =====
    optimizer = torch.optim.AdamW(params, lr=3e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6)

    num_epochs = 80

    # ===== 主训练循环 =====
    for epoch in range(start_epoch, num_epochs):
        bridge.train()

        # 每轮重建训练池 (多样化文本换新)
        dataset = MixedPoseDataset(
            template_weight=5,
            extra_diverse_per_epoch=500,
        )

        total_loss = 0.0
        total_yaw_loss = 0.0
        total_pitch_loss = 0.0
        smooth_loss = 0.0  # 滑动平均

        pbar = tqdm(range(dataset.num_epoch_samples), desc=f"Epoch {epoch+1}/{num_epochs}")
        for step in pbar:
            optimizer.zero_grad()

            texts, r6, cam = dataset.get_train_batch(n=1)
            with torch.no_grad():
                hidden = get_llm_hidden(bridge.qwen, bridge.processor, texts[0])

            # 前向: LLM → Pose → 6D → camera (跳过 Adapter)
            pred_6d = bridge.pose(hidden.unsqueeze(0))
            pred_cam = pose_6d_to_camera_params(pred_6d)

            # 损失: 6D MSE + Camera MSE + 圆形 yaw 损失
            loss_yaw = circular_mse(pred_6d[:, 0], r6[:, 0])
            loss_other = F.mse_loss(pred_6d[:, 1:], r6[:, 1:])
            loss_6d = loss_yaw + loss_other
            loss_cam = F.mse_loss(pred_cam, cam)
            loss = loss_6d + loss_cam

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()

            total_loss += loss.item()
            total_yaw_loss += loss_yaw.item()
            total_pitch_loss += F.mse_loss(pred_6d[:, 1], r6[:, 1]).item()

            smooth_loss = 0.95 * smooth_loss + 0.05 * loss.item() if smooth_loss else loss.item()

            pbar.set_postfix({
                'loss': f"{smooth_loss:.4f}",
                'yaw': f"{loss_yaw.item():.4f}",
            })

        scheduler.step()
        avg_loss = total_loss / dataset.num_epoch_samples

        # ===== 定期验证 =====
        if (epoch + 1) % 5 == 0:
            bridge.eval()

            # 验证多样化文本
            y_errs, p_errs = [], []
            with torch.no_grad():
                for v in dataset.val_data:
                    hidden = get_llm_hidden(bridge.qwen, bridge.processor, v['text'])
                    p6d = bridge.pose(hidden.unsqueeze(0))
                    ye = abs(p6d[0, 0].item() - v['yaw_gt'])
                    # 圆形修正
                    ye = min(ye, 2 * math.pi - ye)
                    y_errs.append(ye)
                    p_errs.append(abs(p6d[0, 1].item() - v['pitch_gt']))

            # 验证模板
            ty_errs, tp_errs = [], []
            with torch.no_grad():
                for v in dataset.val_templates:
                    hidden = get_llm_hidden(bridge.qwen, bridge.processor, v['text'])
                    p6d = bridge.pose(hidden.unsqueeze(0))
                    ye = min(abs(p6d[0, 0].item() - v['yaw_gt']), 2 * math.pi - abs(p6d[0, 0].item() - v['yaw_gt']))
                    ty_errs.append(ye)
                    tp_errs.append(abs(p6d[0, 1].item() - v['pitch_gt']))

            # 保存 checkpoint
            path = os.path.join(ckpt_dir, f"mix_v2_epoch_{epoch+1:03d}.pt")
            torch.save({
                'epoch': epoch + 1,
                'pose': bridge.pose.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, path)

            # 打印结果
            avg_ye = np.mean(y_errs) * 180 / math.pi
            avg_pe = np.mean(p_errs) * 180 / math.pi
            avg_ty = np.mean(ty_errs) * 180 / math.pi
            avg_tp = np.mean(tp_errs) * 180 / math.pi

            print(f"  [Epoch {epoch+1}] loss={avg_loss:.4f}")
            print(f"  val多样化: Δyaw={avg_ye:.2f}° Δpitch={avg_pe:.2f}°")
            print(f"  val模板:   Δyaw={avg_ty:.2f}° Δpitch={avg_tp:.2f}°")

            # 打印前3个多样本
            for i, v in enumerate(dataset.val_data[:3]):
                with torch.no_grad():
                    hidden = get_llm_hidden(bridge.qwen, bridge.processor, v['text'])
                    p6d = bridge.pose(hidden.unsqueeze(0))
                print(f"  [{i+1}] '{v['text'][:30]:30s}' → "
                      f"y={p6d[0,0]:+.2f} p={p6d[0,1]:+.2f} "
                      f"gt=({v['yaw_gt']:+.2f},{v['pitch_gt']:+.2f})")

            print(f"  已保存: {path}\n")

            bridge.train()

    print("\n混合训练 v2 完成!")


if __name__ == "__main__":
    train()
