"""检查训练后的 pitch 精度 (支持 VLM 和轻量编码器)"""
import sys, math, torch, os
sys.path.insert(0, "/root/Qwen3-VL")
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--encoder', choices=['vlm','clip','bert'], default='vlm')
args = parser.parse_args()

if args.encoder == 'vlm':
    from bridge import QwenEG3DBridge, get_llm_hidden
    bridge = QwenEG3DBridge()
    ckpt_path = "/root/Qwen3-VL/checkpoints/mix_v2_epoch_045.pt"
    assert os.path.exists(ckpt_path), f"{ckpt_path} not found"
    bridge.pose.load_state_dict(torch.load(ckpt_path, map_location="cuda")["pose"])
    bridge.eval()
    def get_pose_fn(text):
        hidden = get_llm_hidden(bridge.qwen, bridge.processor, text)
        return bridge.pose(hidden.unsqueeze(0))
else:
    from lightweight_bridge import LightweightBridge, get_embedding_for_training
    bridge = LightweightBridge(encoder_name=args.encoder)
    # 尝试加载训练好的权重
    ckpt_path = f"/root/Qwen3-VL/checkpoints/{args.encoder}_epoch_latest.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cuda")
        bridge.projection.load_state_dict(ckpt["projection"])
        bridge.pose.load_state_dict(ckpt["pose"])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"No checkpoint found at {ckpt_path}, using untrained weights")
    bridge.eval()
    def get_pose_fn(text):
        emb = get_embedding_for_training(bridge, text)
        h = bridge.projection(emb)
        return bridge.pose(h)

# 加载数据集
from train_mixed_v2 import MixedPoseDataset
ds = MixedPoseDataset()

yaw_errs, pit_errs = [], []
with torch.no_grad():
    for v in ds.val_data:
        pred_6d = get_pose_fn(v['text'])
        ye = abs(pred_6d[0, 0].item() - v['yaw_gt'])
        ye = min(ye, 2 * math.pi - ye)
        yaw_errs.append(ye)
        pit_errs.append(abs(pred_6d[0, 1].item() - v['pitch_gt']))

avg_y = sum(yaw_errs)/len(yaw_errs)
avg_p = sum(pit_errs)/len(pit_errs)
print(f"\n=== {args.encoder.upper()} Baseline ===")
print(f"样本数: {len(yaw_errs)}")
print(f"yaw 误差: {avg_y:.4f}rad ({avg_y*180/math.pi:.2f}°)")
print(f"pitch 误差: {avg_p:.4f}rad ({avg_p*180/math.pi:.2f}°)")
print(f"yaw 中位数: {sorted(yaw_errs)[len(yaw_errs)//2]*180/math.pi:.2f}°")
print(f"pitch 中位数: {sorted(pit_errs)[len(pit_errs)//2]*180/math.pi:.2f}°")
