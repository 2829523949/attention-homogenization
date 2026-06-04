"""生成图文角度编辑数据集: 同一身份多视角图 + 不匹配文本"""
import sys, os, math, random, torch, torchvision, json
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, pose_6d_to_camera_params

bridge = QwenEG3DBridge()
G = bridge.eg3d

# 密集视角: 9个基准 + 中间层 (共17个yaw × 3个pitch = 51个角度)
YAWS = [-0.79, -0.52, -0.26, 0.0, 0.26, 0.52, 0.79]  # 7个yaw间隔~15°
PITCH_OFFSETS = [-0.35, 0.0, 0.35]  # 俯/平/仰

NUM_IDENTITIES = 50  # 50个身份
os.makedirs("/root/edit_data/images", exist_ok=True)

all_samples = []  # 每个样本: id, yaw, pitch, text, image_path, raw_6d
idx = 0

for zid in range(NUM_IDENTITIES):
    z = torch.randn(1, G.z_dim, device='cuda')
    # 固定随机种子确保可复现
    torch.manual_seed(zid)
    for yaw in YAWS:
        for po in PITCH_OFFSETS:
            pitch = math.pi / 2 + po
            raw_6d = torch.tensor([[yaw, pitch, 2.7, 18.8, 0.0, 0.2]], dtype=torch.float32).cuda()
            cam = pose_6d_to_camera_params(raw_6d)
            with torch.no_grad():
                ws = G.mapping(z, cam, truncation_psi=0.7)
                img = G.synthesis(ws, cam)['image']
            
            # 确定文本
            if abs(yaw) < 0.1: dir_name = "正面"
            elif yaw < -0.1: dir_name = "左侧"
            else: dir_name = "右侧"
            if abs(yaw) < 0.3: deg = ""
            elif abs(yaw) < 0.4: deg = "30度"
            else: deg = "45度"
            
            pitch_name = ["俯视", "平视", "仰视"][PITCH_OFFSETS.index(po)]
            text = f"{dir_name}{deg}{pitch_name}"
            
            fname = f"z{zid:03d}_y{round(yaw,2):+.2f}_p{po:+.2f}.png"
            torchvision.utils.save_image((img[0]*0.5+0.5).clamp(0,1), f"/root/edit_data/images/{fname}")
            
            all_samples.append({
                'zid': zid, 'yaw': round(yaw,2), 'pitch_offset': round(po,2),
                'text': text, 'image': fname,
                'raw_6d': [yaw, pitch, 2.7, 18.8, 0.0, 0.2],
            })
            idx += 1

print(f"生成 {len(all_samples)} 张图 ({NUM_IDENTITIES}身份 × {len(YAWS)}yaw × {len(PITCH_OFFSETS)}pitch)")

# 构建不匹配图文对 (训练数据)
# 每对: image_z_y1 + text_y2 → target = y2 的位姿
train_pairs = []
val_pairs = []

# 按zid分配: 45个身份训练, 5个验证
random.shuffle(all_samples)
by_zid = {}
for s in all_samples:
    by_zid.setdefault(s['zid'], []).append(s)
all_zids = list(by_zid.keys())
random.shuffle(all_zids)
val_zids = set(all_zids[:5])
train_zids = set(all_zids[5:])

# 为每个身份构建不匹配对
for zid, samples in by_zid.items():
    is_val = zid in val_zids
    for src in samples:
        for dst in samples:
            if src['yaw'] == dst['yaw'] and src['pitch_offset'] == dst['pitch_offset']:
                continue  # 同一角度跳过
            pair = {
                'image': src['image'],
                'text': dst['text'],
                'target_raw_6d': dst['raw_6d'],
                'zid': zid,
            }
            if is_val:
                val_pairs.append(pair)
            else:
                train_pairs.append(pair)

print(f"训练对: {len(train_pairs)} (45身份 × {len(train_pairs)//45}对/身份)")
print(f"验证对: {len(val_pairs)} (5身份)")

# 保存
torch.save({
    'samples': all_samples,
    'train_pairs': train_pairs,
    'val_pairs': val_pairs,
    'yaws': YAWS,
    'pitch_offsets': PITCH_OFFSETS,
}, "/root/edit_data/dataset.pt")

print(f"保存到 /root/edit_data/dataset.pt")
print(f"\n总图数: {len(all_samples)}")
print(f"训练对: {len(train_pairs)}, 验证对: {len(val_pairs)}")
