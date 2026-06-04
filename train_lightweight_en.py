"""
Lightweight encoder training (English dataset)
CLIP/BERT/XLM → Projection → PoseMappingLayer → 6D → Camera
"""
import os, sys, math, argparse, torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/root/Qwen3-VL")
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from lightweight_bridge import LightweightBridge, get_embedding_for_training
from bridge import pose_6d_to_camera_params
from train_mixed_v2_en import circular_mse, MixedPoseDatasetEn


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', choices=['clip', 'bert', 'xlm'], required=True)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Lightweight EN: {args.encoder.upper()}")
    print(f"  epochs={args.epochs} lr={args.lr}")
    print("=" * 60)
    
    bridge = LightweightBridge(encoder_name=args.encoder)
    
    for p in bridge.text_model.parameters(): p.requires_grad = False
    for p in bridge.projection.parameters(): p.requires_grad = True
    for p in bridge.pose.parameters(): p.requires_grad = True
    params = list(bridge.projection.parameters()) + list(bridge.pose.parameters())
    
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    ckpt_dir = "/root/Qwen3-VL/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        bridge.train()
        dataset = MixedPoseDatasetEn(template_weight=5, extra_diverse_per_epoch=500)
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
        
        if (epoch + 1) % 5 == 0:
            bridge.eval()
            y_errs, p_errs = [], []
            with torch.no_grad():
                for v in dataset.val_data:
                    emb = get_embedding_for_training(bridge, v['text'])
                    h = bridge.projection(emb)
                    p6d = bridge.pose(h)
                    ye = min(abs(p6d[0, 0].item() - v['yaw_gt']), 2 * math.pi - abs(p6d[0, 0].item() - v['yaw_gt']))
                    y_errs.append(ye)
                    p_errs.append(abs(p6d[0, 1].item() - v['pitch_gt']))
            
            ckpt_path = os.path.join(ckpt_dir, f"{args.encoder}_en_epoch_{epoch+1:03d}.pt")
            torch.save({
                'epoch': epoch + 1,
                'projection': bridge.projection.state_dict(),
                'pose': bridge.pose.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            
            avg_ye = np.mean(y_errs) * 180 / math.pi
            avg_pe = np.mean(p_errs) * 180 / math.pi
            print(f"[{args.encoder.upper()} EN Epoch {epoch+1}] loss={avg_loss:.4f}  Δyaw={avg_ye:.2f}°  Δpitch={avg_pe:.2f}°")
            bridge.train()
    
    print(f"{args.encoder.upper()} EN training done!")


if __name__ == "__main__":
    train()
