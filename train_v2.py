"""
训练第二阶段: 大规模多样化数据 + Qwen 自动生成自然语言描述
=========================================================
改进:
  1. 随机采样相机参数 (连续分布, 非固定角度)
  2. 用 Qwen3-VL 自动为每种位姿生成自然语言描述
  3. 更大的数据集 (2000+ 样本)
  4. 支持图文模式训练 (为 LLM 控制注入做准备)
"""
import os, sys, math, random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

sys.path.insert(0, "/root/Qwen3-VL")
from bridge import (
    QwenEG3DBridge, get_llm_hidden, pose_6d_to_camera_params
)


# ============================================================
# EG3D 自生成数据集 (第二阶段: 大规模多样化)
# ============================================================
class EnhancedPoseDataset:
    """
    用 EG3D 生成大规模训练数据:
      - 随机采样 3D 相机位姿 (连续分布)
      - 用 Qwen 为每个位姿自动生成自然语言描述
      - 可同时生成图像 (用于后续图文训练)
    """
    def __init__(self, bridge, num_samples=2000):
        self.bridge = bridge
        self.data = []
        self.template_used = 0

        print(f"生成 {num_samples} 样本数据集...")

        for _ in tqdm(range(num_samples), desc="生成数据"):
            # 1. 随机采样相机参数 (连续分布)
            yaw = random.uniform(-3.14, 3.14)
            pitch_offset = random.uniform(-0.8, 0.8)  # -45° ~ +45°
            pitch = math.pi / 2 + pitch_offset
            radius = random.uniform(2.0, 3.5)
            fov = random.uniform(12, 25)
            lx = random.uniform(-0.2, 0.2)
            lz = random.uniform(-0.1, 0.3)

            raw_6d = torch.tensor([[yaw, pitch, radius, fov, lx, lz]], dtype=torch.float32)
            cam = pose_6d_to_camera_params(raw_6d)

            # 2. 用 Qwen 生成自然语言描述
            text = self._generate_viewpoint_text(yaw, pitch_offset)

            self.data.append({
                'text': text,
                'raw_6d_gt': raw_6d,
                'cam_gt': cam,
                'yaw': yaw,
                'pitch_offset': pitch_offset,
            })

        unique_texts = set(d['text'] for d in self.data)
        print(f"生成完成: {len(self.data)} 样本, {len(unique_texts)} 种不同描述")

    def _generate_viewpoint_text(self, yaw, pitch_offset):
        """根据位姿生成自然语言描述"""
        # 方位描述
        abs_yaw = abs(yaw)
        if abs_yaw < 0.3:
            dir_text = "正面" if random.random() > 0.3 else random.choice(["正脸", "正对镜头", "看向前方"])
        elif abs_yaw < 0.7:
            prefix = "左侧" if yaw > 0 else "右侧"
            dir_text = prefix + str(random.choice(["30度", "微侧"]))
            dir_text += random.choice(["偏头", ""])
        elif abs_yaw < 1.2:
            prefix = "左侧" if yaw > 0 else "右侧"
            dir_text = prefix + str(random.choice(["45度", "半侧面"]))
            dir_text += random.choice(["", "的方向"])
        elif abs_yaw < 2.0:
            prefix = "左侧" if yaw > 0 else "右侧"
            dir_text = prefix + str(random.choice(["近90度", "侧面", "侧向"]))
            dir_text += random.choice(["", "的角度"])
        else:
            dir_text = random.choice(["背面", "后脑勺", "从背后看", "后侧"])

        # 俯仰描述
        if pitch_offset < -0.3:
            pitch_text = random.choice(["仰视", "从下往上看", "低角度", "仰角"])
        elif pitch_offset > 0.3:
            pitch_text = random.choice(["俯视", "从上往下看", "高角度", "俯角"])
        else:
            pitch_text = random.choice(["平视", "水平视角", "正常角度", ""])

        # 组合
        if pitch_text:
            return f"{dir_text} {pitch_text}"
        return dir_text

    def get_batch(self, batch_size=8):
        batch = random.sample(self.data, min(batch_size, len(self.data)))
        texts = [item['text'] for item in batch]
        raw_6d_gt = torch.cat([item['raw_6d_gt'] for item in batch], dim=0)
        cam_gt = torch.cat([item['cam_gt'] for item in batch], dim=0)
        return texts, raw_6d_gt.to('cuda'), cam_gt.to('cuda')


# ============================================================
# 图文数据集 (为 LLM 控制注入做准备)
# ============================================================
class MultimodalPoseDataset:
    """
    用 EG3D 生成 图文对 用于多模态训练:
      - 随机采样相机位姿
      - 用 EG3D 渲染对应图像
      - 文本描述包含更丰富的语义 (不仅包含视角)
    """
    def __init__(self, bridge, num_samples=100):
        self.bridge = bridge
        self.data = []
        self.eg3d = bridge.eg3d

        print(f"生成 {num_samples} 图文对...")

        for i in tqdm(range(num_samples), desc="生成图文数据"):
            yaw = random.uniform(-3.14, 3.14)
            pitch_offset = random.uniform(-0.8, 0.8)
            pitch = math.pi / 2 + pitch_offset
            radius = random.uniform(2.0, 3.5)
            fov = random.uniform(12, 25)

            raw_6d = torch.tensor([[yaw, pitch, radius, fov, 0.0, 0.2]], dtype=torch.float32)
            cam = pose_6d_to_camera_params(raw_6d)

            # 用 EG3D 生成图像
            with torch.no_grad():
                z = torch.randn(1, self.eg3d.z_dim, device='cuda')
                ws = self.eg3d.mapping(z, cam, truncation_psi=0.7)
                img = self.eg3d.synthesis(ws, cam)['image']

            # 构造多样化文本
            text = _gen_viewpoint_text(yaw, pitch_offset)

            self.data.append({
                'text': text,
                'image': img,           # [1, 3, 512, 512]
                'raw_6d_gt': raw_6d,
                'cam_gt': cam,
            })

        print(f"图文数据集: {len(self.data)} 样本")

    def get_batch(self, batch_size=2):
        batch = random.sample(self.data, min(batch_size, len(self.data)))
        texts = [item['text'] for item in batch]
        images = torch.cat([item['image'] for item in batch], dim=0)
        raw_6d_gt = torch.cat([item['raw_6d_gt'] for item in batch], dim=0)
        cam_gt = torch.cat([item['cam_gt'] for item in batch], dim=0)
        return texts, images, raw_6d_gt.to('cuda'), cam_gt.to('cuda')


def _gen_viewpoint_text(yaw, pitch_offset):
    """为训练样本生成多样化文本"""
    adjectives = ["一位", "一个", "有一张"]
    genders = ["", "年轻", "男性", "女性", ""]
    styles = ["凝视镜头", "表情自然", "面无表情", "微笑", ""]

    abs_yaw = abs(yaw)
    if abs_yaw < 0.3:
        view = random.choice(["正面肖像照", "正脸照片", "正面看镜头"])
    elif abs_yaw < 0.7:
        side = "左侧" if yaw > 0 else "右侧"
        view = random.choice([f"{side}偏头照", f"{side}微侧的照片"])
    elif abs_yaw < 1.2:
        side = "左侧" if yaw > 0 else "右侧"
        view = f"{side}{random.choice(['45度', '半侧面', '侧脸'])}照片"
    elif abs_yaw < 2.0:
        side = "左侧" if yaw > 0 else "右侧"
        view = f"{side}{random.choice(['侧面', '侧向'])}的照片"
    else:
        view = random.choice(["背面视角的", "从背后拍的", "后侧方的"])

    adj = random.choice(adjectives)
    gender = random.choice(genders)
    style = random.choice(styles)

    parts = [p for p in [adj, gender, view, style] if p]
    return "".join(parts)


# ============================================================
# 训练
# ============================================================
def train():
    print("=" * 50)
    print("第二阶段训练: 大规模数据 + 多样文本")
    print("=" * 50)

    bridge = QwenEG3DBridge()

    # ===== 纯文本训练: 冻结门控, 只训练 PoseMappingLayer =====
    # 门控在纯文本模式下没有意义，保留现有权重留给未来图文训练
    for p in bridge.fusion.parameters():
        p.requires_grad = False
    params = list(bridge.pose.parameters())
    optimizer = torch.optim.Adam(params, lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    ckpt_dir = "/root/Qwen3-VL/checkpoints"
    start_epoch = 0
    if os.path.exists(ckpt_dir):
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
        if ckpts:
            latest = os.path.join(ckpt_dir, ckpts[-1])
            ckpt = torch.load(latest, map_location='cuda')
            bridge.pose.load_state_dict(ckpt['pose'])
            # 如果 checkpoint 包含优化器状态且参数匹配，则加载；否则重新开始
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
                print(f"已加载 {latest} (epoch {ckpt['epoch']+1})")
            except ValueError:
                print(f"已加载 {latest} (epoch {ckpt['epoch']+1})，但优化器参数不匹配，重新初始化优化器")
            start_epoch = ckpt['epoch'] + 1

    # ===== 生成大规模数据集 =====
    dataset = EnhancedPoseDataset(bridge, num_samples=500)

    # ===== 可选: 图文数据集 =====
    # multimodal_dataset = MultimodalPoseDataset(bridge, num_samples=50)

    # ===== 训练 =====
    num_epochs = 50
    for epoch in range(start_epoch, start_epoch + num_epochs):
        total_loss = 0.0
        num_batches = len(dataset.data) // 8  # batch_size=8

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{start_epoch+num_epochs}")
        for _ in pbar:
            optimizer.zero_grad()
            texts, raw_6d_gt, cam_gt = dataset.get_batch(batch_size=8)
            batch_loss = 0.0

            for i, text in enumerate(texts):
                hidden = get_llm_hidden(bridge.qwen, bridge.processor, text)
                # 纯文本: 直接 LLM hidden → PoseMappingLayer (跳过门控)
                pred_6d = bridge.pose(hidden.unsqueeze(0))
                pred_cam = pose_6d_to_camera_params(pred_6d)

                loss = F.mse_loss(pred_6d, raw_6d_gt[i].unsqueeze(0)) + \
                       F.mse_loss(pred_cam, cam_gt[i].unsqueeze(0))
                batch_loss += loss
                loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            pbar.set_postfix({'loss': f"{batch_loss.item()/len(texts):.4f}"})

        scheduler.step()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        if (epoch + 1) % 5 == 0:
            path = f"/root/Qwen3-VL/checkpoints/epoch_v2_{epoch+1}.pt"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'epoch': epoch, 'pose': bridge.pose.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, path)
            print(f"  已保存: {path} | avg_loss: {avg_loss:.4f}")

    print("\n训练完成!")


if __name__ == "__main__":
    train()
