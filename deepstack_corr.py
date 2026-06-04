"""检查 deepstack 三层之间的相关性"""
import sys, torch, tempfile, torchvision
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, get_multimodal_features, pose_6d_to_camera_params

bridge = QwenEG3DBridge()

# 多角度测试
results = []
for yaw in [-0.79, 0.0, 0.79]:
    for po in [-0.35, 0.0, 0.35]:
        cam = pose_6d_to_camera_params(torch.tensor([[yaw, 1.57+po, 2.7, 18.8, 0.0, 0.2]]).cuda())
        ws = bridge.eg3d.mapping(torch.randn(1, bridge.eg3d.z_dim, device='cuda'), cam, truncation_psi=0.7)
        img = bridge.eg3d.synthesis(ws, cam)['image']
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        torchvision.utils.save_image(img[0]*0.5+0.5, tmp.name)
        feats, _ = get_multimodal_features(bridge.qwen, bridge.processor, "test", tmp.name)
        
        cos01 = torch.cosine_similarity(feats[0].mean(0).unsqueeze(0), feats[1].mean(0).unsqueeze(0)).item()
        cos02 = torch.cosine_similarity(feats[0].mean(0).unsqueeze(0), feats[2].mean(0).unsqueeze(0)).item()
        cos12 = torch.cosine_similarity(feats[1].mean(0).unsqueeze(0), feats[2].mean(0).unsqueeze(0)).item()
        results.append((yaw, po, cos01, cos02, cos12))
        import os; os.unlink(tmp.name)

print("yaw  po    | cos[0,1]  cos[0,2]  cos[1,2]")
print("-" * 45)
for y, p, c01, c02, c12 in results:
    print(f"{y:+.2f} {p:+.2f} | {c01:.4f}   {c02:.4f}   {c12:.4f}")

avg01 = sum(r[2] for r in results)/len(results)
avg02 = sum(r[3] for r in results)/len(results)
avg12 = sum(r[4] for r in results)/len(results)
print(f"\n平均: {avg01:.4f}   {avg02:.4f}   {avg12:.4f}")
