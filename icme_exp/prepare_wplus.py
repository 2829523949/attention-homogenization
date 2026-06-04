"""预计算每个身份的 w+ 编码 (独立加载 EG3D, 不需要 DINO)"""
import sys, torch, math
sys.path.insert(0, "/root/eg3d/eg3d")
import legacy, dnnlib

print("加载 EG3D...")
with dnnlib.util.open_url("/root/eg3d/networks/nvidia_ffhqrebalanced512-128.pkl") as f:
    G = legacy.load_network_pkl(f)['G_ema'].to('cuda')
G.eval()

# 6D → cam 参数
def pose_6d_to_camera_params(raw_6d, device='cuda'):
    import torch.nn.functional as F
    B = raw_6d.shape[0]
    yaw, pitch = raw_6d[:, 0], raw_6d[:, 1]
    radius = raw_6d[:, 2].clamp(min=1.0)
    fov = raw_6d[:, 3].clamp(min=5.0)
    lx, lz = torch.tanh(raw_6d[:, 4]) * 0.3, torch.tanh(raw_6d[:, 5]) * 0.3
    lookat = torch.stack([lx, torch.zeros(B, device=device), lz], dim=1)
    origin = torch.zeros((B, 3), device=device)
    origin[:, 0] = radius * torch.sin(pitch) * torch.cos(math.pi / 2 - yaw)
    origin[:, 1] = radius * torch.cos(pitch)
    origin[:, 2] = radius * torch.sin(pitch) * torch.sin(math.pi / 2 - yaw)
    forward = F.normalize(lookat - origin, dim=1)
    up = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).expand(B, 3)
    right = F.normalize(-torch.cross(up, forward, dim=1), dim=1)
    up = F.normalize(torch.cross(forward, right, dim=1), dim=1)
    rot = torch.eye(4, device=device).unsqueeze(0).expand(B, 4, 4).clone()
    rot[:, :3, :3] = torch.stack([right, up, forward], dim=-1)
    trans = torch.eye(4, device=device).unsqueeze(0).expand(B, 4, 4).clone()
    trans[:, :3, 3] = origin
    cam2world = trans @ rot
    focal = 1.0 / (torch.tan(fov * math.pi / 360) * 1.414)
    intrinsics = torch.zeros((B, 3, 3), device=device)
    intrinsics[:, 0, 0] = focal; intrinsics[:, 1, 1] = focal
    intrinsics[:, 0, 2] = 0.5; intrinsics[:, 1, 2] = 0.5; intrinsics[:, 2, 2] = 1.0
    return torch.cat([cam2world.reshape(B, 16), intrinsics.reshape(B, 9)], dim=1)

# 加载数据集
data = torch.load("/root/edit_data/dataset.pt")

# 每个身份: zid → w+ (14×512)
identity_w = {}
for s in data['samples']:
    zid = s['zid']
    if zid in identity_w:
        continue
    torch.manual_seed(zid)
    z = torch.randn(1, G.z_dim, device='cuda')
    yaw, po = s['yaw'], s['pitch_offset']
    cam = pose_6d_to_camera_params(torch.tensor([[yaw, math.pi/2+po, 2.7, 18.8, 0.0, 0.2]], device='cuda'))
    with torch.no_grad():
        ws = G.mapping(z, cam, truncation_psi=0.7)  # [1, 14, 512]
    identity_w[zid] = ws.cpu()
    print(f"  zid={zid:3d}: w+ computed")

torch.save(identity_w, "/root/edit_data/identity_w.pt")
print(f"\n保存 {len(identity_w)} 个身份的 w+ 到 /root/edit_data/identity_w.pt")
