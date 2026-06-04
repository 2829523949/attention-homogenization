"""
Phase 2 训练: 图文角度编辑
输入: 人脸图片(任意角度) + 文本(目标角度) → 输出: 目标6D位姿
训练数据: 图像在V1 + 文本描述V2 → MSE到V2
"""
import os, sys, math, random
import torch, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, get_multimodal_features, pose_6d_to_camera_params


class EditDataset:
    """图文角度编辑数据集 (不匹配对)"""

    def __init__(self, data_path="/root/edit_data/dataset.pt"):
        data = torch.load(data_path)
        self.train_pairs = data['train_pairs']
        self.val_pairs = data['val_pairs']
        self.train_pairs = [p for p in self.train_pairs if os.path.exists(f"/root/edit_data/images/{p['image']}")]
        self.val_pairs = [p for p in self.val_pairs if os.path.exists(f"/root/edit_data/images/{p['image']}")]
        print(f"训练对: {len(self.train_pairs)} 验证对: {len(self.val_pairs)}")

    def get_batch(self, n=1):
        items = random.sample(self.train_pairs, min(n, len(self.train_pairs)))
        images = [f"/root/edit_data/images/{p['image']}" for p in items]
        texts = [p['text'] for p in items]
        targets = [torch.tensor(p['target_raw_6d'], dtype=torch.float32) for p in items]
        return images, texts, torch.stack(targets).cuda()


def train():
    print("=" * 60)
    print("Phase 2: 图文角度编辑 (不匹配对训练)")
    print("=" * 60)

    bridge = QwenEG3DBridge()

    # 加载预训练权重 (混合训练的pose)
    ckpt_dir = "/root/Qwen3-VL/checkpoints"
    mix_ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('mix_epoch_') and f.endswith('.pt')],
                       key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if mix_ckpts:
        ckpt = torch.load(os.path.join(ckpt_dir, mix_ckpts[-1]), map_location='cuda')
        bridge.pose.load_state_dict(ckpt['pose'])
        print(f"加载 pose: {mix_ckpts[-1]}")
    else:
        text_ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('text_only_') and f.endswith('.pt')],
                            key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if text_ckpts:
            bridge.pose.load_state_dict(torch.load(os.path.join(ckpt_dir, text_ckpts[-1]), map_location='cuda')['pose'])
            print(f"加载 pose: {text_ckpts[-1]}")

    # 冻结Qwen, 训练fusion + pose
    # 重置门控网络 (打破 [1,0,0] 锁死)
    for layer in bridge.fusion.gate_net:
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    print("已重置 gate_net")

    # 重置 fusion 的 text_proj 以避免锁死
    bridge.fusion.text_proj.reset_parameters()

    for p in bridge.qwen.parameters(): p.requires_grad = False
    for p in bridge.fusion.parameters(): p.requires_grad = True
    for p in bridge.pose.parameters(): p.requires_grad = True
    for p in bridge.adapter.parameters(): p.requires_grad = False

    dataset = EditDataset()
    params = list(bridge.fusion.parameters()) + list(bridge.pose.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    num_epochs = 30
    print(f"可训练参数: {sum(p.numel() for p in params):,} (fusion + pose)")

    for epoch in range(num_epochs):
        total_loss = 0.0
        n_batches = 500  # 每轮随机采样500步 (≈2分钟)
        pbar = tqdm(range(n_batches), desc=f"Edit Epoch {epoch+1}/{num_epochs}")

        for _ in pbar:
            optimizer.zero_grad()
            images, texts, target_6d = dataset.get_batch(n=1)

            with torch.no_grad():
                deep_features, llm_hidden = get_multimodal_features(
                    bridge.qwen, bridge.processor, texts[0], images[0]
                )
            fused, gates = bridge.fusion(llm_hidden, deep_features)
            pred_6d = bridge.pose(fused.unsqueeze(0))
            pred_cam = pose_6d_to_camera_params(pred_6d)
            target_cam = pose_6d_to_camera_params(target_6d)
            loss = F.mse_loss(pred_6d, target_6d) + F.mse_loss(pred_cam, target_cam)
            # 门控熵正则: 鼓励门控分化, 避免坍缩为 [1,0,0]
            ent = -(gates * torch.log(gates + 1e-8)).sum()
            loss = loss - 0.1 * ent  # 熵越大 → loss越小
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'gates': f"{gates.data.cpu().numpy().round(2)}"})

        scheduler.step()
        avg_loss = total_loss / n_batches

        # 每5轮验证: 用验证身份的不匹配对
        if (epoch + 1) % 5 == 0:
            path = f"/root/Qwen3-VL/checkpoints/edit_epoch_{epoch+1:03d}.pt"
            torch.save({'epoch': epoch+1, 'fusion': bridge.fusion.state_dict(),
                        'pose': bridge.pose.state_dict(), 'optimizer': optimizer.state_dict()}, path)
            bridge.eval()
            y_errs = []
            with torch.no_grad():
                for v in dataset.val_pairs[:50]:  # 50个验证对
                    deep_f, llm_h = get_multimodal_features(bridge.qwen, bridge.processor, v['text'], f"/root/edit_data/images/{v['image']}")
                    fused, gates = bridge.fusion(llm_h, deep_f)
                    p6d = bridge.pose(fused.unsqueeze(0))
                    target = torch.tensor(v['target_raw_6d'], device='cuda').unsqueeze(0)
                    y_errs.append(abs(p6d[0,0].item() - target[0,0].item()))
            bridge.train()
            print(f"  val: Δy={np.mean(y_errs):.3f} | saved: {path} | loss: {avg_loss:.4f}\n")

    print("Phase 2 训练完成!")


if __name__ == "__main__":
    train()
