"""测试: 仅 PoseMappingLayer (跳过 adapter)"""
import sys, torch, torchvision, os
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, get_llm_hidden, pose_6d_to_camera_params

bridge = QwenEG3DBridge()
ckpt = torch.load("/root/Qwen3-VL/checkpoints/text_only_epoch_060.pt", map_location='cuda')
bridge.pose.load_state_dict(ckpt['pose'])
print("加载 text_only_epoch_060.pt")

tests = ["正面平视","正面仰视","正面俯视",
         "左侧45度平视","右侧90度平视","背面"]
os.makedirs("/root/output_pose", exist_ok=True)

for t in tests:
    with torch.no_grad():
        h = get_llm_hidden(bridge.qwen, bridge.processor, t)
        raw = bridge.pose(h.unsqueeze(0))
        cam = pose_6d_to_camera_params(raw)
        ws = bridge.eg3d.mapping(
            torch.randn(1, bridge.eg3d.z_dim, device='cuda'), cam, truncation_psi=0.7)
        img = bridge.eg3d.synthesis(ws, cam)['image']
        y, p = raw[0,0].item(), raw[0,1].item()
        print(f"'{t:12s}' → yaw={y:+.2f} pitch={p:+.2f}")
        torchvision.utils.save_image(
            (img[0]*0.5+0.5).clamp(0,1), f"/root/output_pose/{t}.png")

print("保存到 /root/output_pose/")
