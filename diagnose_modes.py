"""诊断: 对比图文模式 vs 纯文本模式的位姿精度"""
import sys, torch, os, math
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, get_llm_hidden, get_multimodal_features, pose_6d_to_camera_params

bridge = QwenEG3DBridge()

import glob
ckpt_dir = "/root/Qwen3-VL/checkpoints"
ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('mm_') and f.endswith('.pt')],
               key=lambda x: int(x.split('_')[-1].split('.')[0]))
latest = os.path.join(ckpt_dir, ckpts[-1])
ckpt = torch.load(latest)
bridge.fusion.load_state_dict(ckpt['fusion'])
bridge.pose.load_state_dict(ckpt['pose'])
print(f"加载: {latest}")

# 用训练数据的标准角度做测试（无抖动）
test_cases = [
    ( 0.00, 0.0,  "正面平视"),
    ( 0.00, 0.35, "正面俯视"),
    ( 0.00, -0.35,"正面仰视"),
    (-0.79, 0.0,  "左侧45度平视"),
    ( 1.57, 0.0,  "右侧90度平视"),
    ( 3.14, 0.0,  "背面"),
]

print(f"\n{'='*70}")
print(f"{'文本':12s} | {'模式':6s} | {'pred yaw':>8s} {'pred pitch':>10s} | {'gt yaw':>7s} {'gt pitch':>10s}")
print(f"{'-'*70}")

# 先用EG3D生成标准位姿的图像（模拟训练数据）
with torch.no_grad():
    z = torch.randn(1, bridge.eg3d.z_dim, device='cuda')
    import tempfile, torchvision

    for gt_yaw, pitch_offset, text in test_cases:
        gt_pitch = math.pi / 2 + pitch_offset
        
        # 生成标准位姿的图像
        raw = torch.tensor([[gt_yaw, gt_pitch, 2.7, 18.837, 0.0, 0.2]])
        cam_gt = pose_6d_to_camera_params(raw).cuda()
        ws = bridge.eg3d.mapping(z, cam_gt, truncation_psi=0.7)
        img = bridge.eg3d.synthesis(ws, cam_gt)['image']
        
        # 保存临时文件
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        torchvision.utils.save_image(img[0]*0.5+0.5, tmp.name)
        
        # === 图文模式 ===
        with torch.no_grad():
            deep_f, llm_h = get_multimodal_features(bridge.qwen, bridge.processor, text, tmp.name)
            fused, gates = bridge.fusion(llm_h, deep_f)
            pred_6d = bridge.pose(fused.unsqueeze(0))
        pred_yaw, pred_pitch = pred_6d[0, 0].item(), pred_6d[0, 1].item()
        yaw_err = abs(pred_yaw - gt_yaw)
        pitch_err = abs(pred_pitch - gt_pitch)
        print(f"{text:12s} | {'图文':6s} | {pred_yaw:>+8.2f} {pred_pitch:>+10.2f} | {gt_yaw:>+7.2f} {gt_pitch:>+10.2f} | Δy={yaw_err:.2f} Δp={pitch_err:.2f}")
        
        # === 纯文本模式 ===
        with torch.no_grad():
            llm_h = get_llm_hidden(bridge.qwen, bridge.processor, text)
            pred_6d = bridge.pose(llm_h.unsqueeze(0))
        pred_yaw_t, pred_pitch_t = pred_6d[0, 0].item(), pred_6d[0, 1].item()
        yaw_err_t = abs(pred_yaw_t - gt_yaw)
        pitch_err_t = abs(pred_pitch_t - gt_pitch)
        print(f"{text:12s} | {'纯文本':6s} | {pred_yaw_t:>+8.2f} {pred_pitch_t:>+10.2f} | {gt_yaw:>+7.2f} {gt_pitch:>+10.2f} | Δy={yaw_err_t:.2f} Δp={pitch_err_t:.2f}")
        print()
        
        os.unlink(tmp.name)

print(f"{'='*70}")
print("结论: 如果图文模式误差小，纯文本误差大 → 训练正确，需要纯文本混合训练")
print("如果两者误差都大 → 训练本身有问题")
