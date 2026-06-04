"""
训练: 文本 → LLM控制注入 → 6D Pose → EG3D
=============================================
训练两部分:
  1. LLMControlledFeatureFusion (13.1M) - LLM门控控制视觉特征注入
  2. PoseMappingLayer (2.6M) - 融合特征→6D pose
总计: 15.7M 可训练参数
"""
import os, sys, math, random, torch, torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, "/root/Qwen3-VL")
from bridge import (
    QwenEG3DBridge, get_llm_hidden, pose_6d_to_camera_params
)


# 预定义视角 (yaw, pitch_offset, 文本描述)
VIEWPOINTS = [
    (-1.57, -0.3, "右侧近90度仰视"), (-1.57,  0.0, "右侧近90度平视"), (-1.57,  0.3, "右侧近90度俯视"),
    (-0.79, -0.3, "右侧45度仰视"),  (-0.79,  0.0, "右侧45度平视"),  (-0.79,  0.3, "右侧45度俯视"),
    ( 0.00, -0.3, "正面仰视"),      ( 0.00,  0.0, "正面平视"),      ( 0.00,  0.3, "正面俯视"),
    ( 0.79, -0.3, "左侧45度仰视"),  ( 0.79,  0.0, "左侧45度平视"),  ( 0.79,  0.3, "左侧45度俯视"),
    ( 1.57, -0.3, "左侧近90度仰视"),( 1.57,  0.0, "左侧近90度平视"),( 1.57,  0.3, "左侧近90度俯视"),
    ( 3.14,  0.0, "背面"),
]

def make_ground_truth(yaw, pitch_offset, radius=2.7, fov=18.837):
    pitch = math.pi / 2 + pitch_offset
    return torch.tensor([[yaw, pitch, radius, fov, 0.0, 0.0]], dtype=torch.float32)


def train():
    print("=" * 50)
    print("训练: LLM控制注入 + 6D Pose 映射")
    print("=" * 50)

    bridge = QwenEG3DBridge()

    # 训练参数
    params = list(bridge.fusion.parameters()) + list(bridge.pose.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    start_epoch = 0
    # 自动找最新的 checkpoint 继续训练
    ckpt_dir = "/root/Qwen3-VL/checkpoints"
    if os.path.exists(ckpt_dir):
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
        if ckpts:
            latest = os.path.join(ckpt_dir, ckpts[-1])
            ckpt = torch.load(latest, map_location='cuda')
            bridge.fusion.load_state_dict(ckpt['fusion'])
            bridge.pose.load_state_dict(ckpt['pose'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch'] + 1
            print(f"已加载 {latest} (epoch {ckpt['epoch']+1}, loss={ckpt['loss']:.4f})")

    num_epochs = 30
    for epoch in range(start_epoch, num_epochs):
        random.shuffle(VIEWPOINTS)
        total_loss = 0.0

        pbar = tqdm(VIEWPOINTS, desc=f"Epoch {epoch+1}/{num_epochs}")
        for yaw, pitch_offset, text in pbar:
            optimizer.zero_grad()

            # GT
            gt_6d = make_ground_truth(yaw, pitch_offset)
            gt_cam = pose_6d_to_camera_params(gt_6d)

            # 前向: 文本 → LLM hidden → 门控融合 → 6D → camera
            hidden = get_llm_hidden(bridge.qwen, bridge.processor, text)
            fused, gates = bridge.fusion(hidden)
            pred_6d = bridge.pose(fused.unsqueeze(0))
            pred_cam = pose_6d_to_camera_params(pred_6d)

            # 损失
            loss_6d = F.mse_loss(pred_6d, gt_6d.to(pred_6d.device))
            loss_cam = F.mse_loss(pred_cam, gt_cam.to(pred_cam.device))
            # 门控多样性正则化 (鼓励不要坍缩到均匀分布)
            gate_entropy = -(gates * torch.log(gates + 1e-8)).sum()
            loss = loss_6d + loss_cam - 0.05 * gate_entropy  # 减熵 = 鼓励分化

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'gates': f"{gates.data.cpu().numpy().round(2)}",
            })

        scheduler.step()
        avg_loss = total_loss / len(VIEWPOINTS)
        print(f"  Epoch {epoch+1} | loss: {avg_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % 5 == 0:
            path = f"/root/Qwen3-VL/checkpoints/epoch_{epoch+1}.pt"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                'epoch': epoch, 'fusion': bridge.fusion.state_dict(),
                'pose': bridge.pose.state_dict(), 'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, path)
            print(f"  已保存: {path}")

    print("\n训练完成!")


if __name__ == "__main__":
    train()
