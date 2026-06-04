"""
轻量编码器训练: CLIP/BERT → Projection → PoseMappingLayer → 6D → Camera
与 train_mixed_v2.py 相同的数据/损失/验证逻辑, 仅编码器不同.
"""
import os, sys, math, random, json, argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, "/root/Qwen3-VL")
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from lightweight_bridge import LightweightBridge, get_embedding_for_training
from bridge import pose_6d_to_camera_params

# ===== 复用 train_mixed_v2 的数据集和损失函数 =====
from train_mixed_v2 import circular_mse, MixedPoseDataset


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', choices=['clip', 'bert'], required=True)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"轻量编码器训练: {args.encoder.upper()}")
    print(f"  epochs={args.epochs} lr={args.lr}")
    print("=" * 60)
    
    bridge = LightweightBridge(encoder_name=args.encoder)
    
    # ===== 冻结/解冻 =====
    for p in bridge.text_model.parameters(): p.requires_grad = False
    for p in bridge.projection.parameters(): p.requires_grad = True
    for p in bridge.pose.parameters(): p.requires_grad = True
    
    params = list(bridge.projection.parameters()) + list(bridge.pose.parameters())
    n_params = sum(p.numel() for p in params)
    print(f"可训练参数: {n_params:,} (Projection + PoseMappingLayer)")
    
    # ===== 加载已有权重 =====
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cuda")
        bridge.projection.load_state_dict(ckpt["projection"])
        bridge.pose.load_state_dict(ckpt["pose"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"恢复训练: {args.resume} (epoch {start_epoch})")
    
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    ckpt_dir = "/root/Qwen3-VL/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    for epoch in range(start_epoch, args.epochs):
        bridge.train()
        
        dataset = MixedPoseDataset(template_weight=5, extra_diverse_per_epoch=500)
        total_loss = 0.0
        smooth_loss = 0.0
        
        pbar = tqdm(range(dataset.num_epoch_samples), desc=f"Epoch {epoch+1}/{args.epochs}")
        for step in pbar:
            optimizer.zero_grad()
            
            texts, r6, cam = dataset.get_train_batch(n=1)
            emb = get_embedding_for_training(bridge, texts[0])
            h = bridge.projection(emb)
            pred_6d = bridge.pose(h)
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
        
        # ===== 验证 =====
        if (epoch + 1) % 5 == 0:
            bridge.eval()
            y_errs, p_errs = [], []
            
            with torch.no_grad():
                for v in dataset.val_data:
                    emb = get_embedding_for_training(bridge, v['text'])
                    h = bridge.projection(emb)
                    p6d = bridge.pose(h)
                    ye = abs(p6d[0, 0].item() - v['yaw_gt'])
                    ye = min(ye, 2 * math.pi - ye)
                    y_errs.append(ye)
                    p_errs.append(abs(p6d[0, 1].item() - v['pitch_gt']))
            
            # 模板精度
            ty_errs, tp_errs = [], []
            with torch.no_grad():
                for v in dataset.val_templates:
                    emb = get_embedding_for_training(bridge, v['text'])
                    h = bridge.projection(emb)
                    p6d = bridge.pose(h)
                    ye = min(abs(p6d[0, 0].item() - v['yaw_gt']), 
                             2 * math.pi - abs(p6d[0, 0].item() - v['yaw_gt']))
                    ty_errs.append(ye)
                    tp_errs.append(abs(p6d[0, 1].item() - v['pitch_gt']))
            
            ckpt_path = os.path.join(ckpt_dir, f"{args.encoder}_epoch_{epoch+1:03d}.pt")
            torch.save({
                'epoch': epoch + 1,
                'projection': bridge.projection.state_dict(),
                'pose': bridge.pose.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            
            avg_ye = np.mean(y_errs) * 180 / math.pi
            avg_pe = np.mean(p_errs) * 180 / math.pi
            avg_ty = np.mean(ty_errs) * 180 / math.pi
            avg_tp = np.mean(tp_errs) * 180 / math.pi
            
            print(f"\n  [{args.encoder.upper()} Epoch {epoch+1}] loss={avg_loss:.4f}")
            print(f"  多样化: Δyaw={avg_ye:.2f}° Δpitch={avg_pe:.2f}°")
            print(f"  模板:   Δyaw={avg_ty:.2f}° Δpitch={avg_tp:.2f}°")
            print(f"  已保存: {ckpt_path}\n")
            
            bridge.train()
    
    print(f"\n{args.encoder.upper()} 训练完成!")


if __name__ == "__main__":
    train()
