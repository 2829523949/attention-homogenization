"""Phase 2 验证: 输入图+文本 → 同一人脸+目标角度"""
import sys, torch, torchvision, os, re, math
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, get_multimodal_features, pose_6d_to_camera_params

bridge = QwenEG3DBridge()

# 加载最新 edit checkpoint
ckpt_dir = "/root/Qwen3-VL/checkpoints"
ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('edit_epoch_') and f.endswith('.pt')],
               key=lambda x: int(x.split('_')[-1].split('.')[0]))
if ckpts:
    ckpt = torch.load(os.path.join(ckpt_dir, ckpts[-1]), map_location='cuda')
    bridge.fusion.load_state_dict(ckpt['fusion'])
    bridge.pose.load_state_dict(ckpt['pose'])
    print(f"加载: {ckpts[-1]}")

os.makedirs("/root/output_edit_test", exist_ok=True)

test_cases = [
    ("z000_y+0.00_p+0.00.png", "左侧45度平视"),
    ("z000_y+0.26_p-0.35.png", "正面平视"),
    ("z001_y-0.52_p+0.35.png", "右侧45度仰视"),
    ("z001_y+0.79_p+0.00.png", "左侧45度俯视"),
    ("z002_y+0.00_p+0.00.png", "右侧45度平视"),
]

for img_name, text in test_cases:
    img_path = f"/root/edit_data/images/{img_name}"
    if not os.path.exists(img_path): continue

    # 从文件名提取 zid
    zid = int(re.search(r'z(\d+)', img_name).group(1))
    # 用相同 zid 的随机种子恢复身份
    torch.manual_seed(zid)
    z = torch.randn(1, bridge.eg3d.z_dim, device='cuda')
    
    # 输入图的原位姿
    orig_yaw = float(re.search(r'y([+-]\d+\.\d+)', img_name).group(1))
    orig_po = float(re.search(r'p([+-]\d+\.\d+)', img_name).group(1))
    orig_pitch = math.pi / 2 + orig_po
    raw_orig = torch.tensor([[orig_yaw, orig_pitch, 2.7, 18.8, 0.0, 0.2]], dtype=torch.float32).cuda()
    cam_orig = pose_6d_to_camera_params(raw_orig)
    ws_orig = bridge.eg3d.mapping(z, cam_orig, truncation_psi=0.7)
    img_orig = bridge.eg3d.synthesis(ws_orig, cam_orig)['image']
    
    # 编辑: 图文输入 → 预测目标位姿
    with torch.no_grad():
        deep_f, llm_h = get_multimodal_features(bridge.qwen, bridge.processor, text, img_path)
        fused, gates = bridge.fusion(llm_h, deep_f)
        pred_6d = bridge.pose(fused.unsqueeze(0))
        pred_cam = pose_6d_to_camera_params(pred_6d)
        
    # 用相同 z (同一身份) + 预测位姿 渲染输出
    ws_new = bridge.eg3d.mapping(z, pred_cam, truncation_psi=0.7)
    img_new = bridge.eg3d.synthesis(ws_new, pred_cam)['image']
    
    y, p = pred_6d[0,0].item(), pred_6d[0,1].item()
    print(f"\n输入: {img_name} (zid={zid})")
    print(f"文本: '{text}'")
    print(f"输出: yaw={y:+.2f} pitch={p:+.2f}")
    print(f"门控: {gates.data.cpu().numpy().round(2)}")
    
    # 保存输入和输出对比
    torchvision.utils.save_image((img_orig[0]*0.5+0.5).clamp(0,1),
                                 f"/root/output_edit_test/in_{text}.png")
    torchvision.utils.save_image((img_new[0]*0.5+0.5).clamp(0,1),
                                 f"/root/output_edit_test/out_{text}.png")
    # 并排对比
    grid = torchvision.utils.make_grid([
        (img_orig[0]*0.5+0.5).clamp(0,1),
        (img_new[0]*0.5+0.5).clamp(0,1)
    ], nrow=2)
    torchvision.utils.save_image(grid, f"/root/output_edit_test/compare_{text}.png")

print(f"\n对比图已保存到 /root/output_edit_test/")
print("in_*.png = 输入图  |  out_*.png = 输出图 (同一身份+编辑后角度)")
print("compare_*.png = 左右并排对比")
