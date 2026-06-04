"""查看 Qwen3-VL deepstack 各层是什么"""
import sys, torch, tempfile, torchvision
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, get_multimodal_features, get_llm_hidden, pose_6d_to_camera_params

bridge = QwenEG3DBridge()

# 生成一张测试图
z = torch.randn(1, bridge.eg3d.z_dim, device='cuda')
cam = pose_6d_to_camera_params(torch.tensor([[0.0, 1.57, 2.7, 18.8, 0.0, 0.2]]).cuda())
ws = bridge.eg3d.mapping(z, cam, truncation_psi=0.7)
img = bridge.eg3d.synthesis(ws, cam)['image']

tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
torchvision.utils.save_image(img[0]*0.5+0.5, tmp.name)

feats, llm_hidden = get_multimodal_features(bridge.qwen, bridge.processor, "正面平视", tmp.name)

print("\n=== DeepStack 特征分析 ===")
for i, f in enumerate(feats):
    print(f"\ndeepstack[{i}]: shape={f.shape}")
    print(f"  mean={f.mean().item():.4f}, std={f.std().item():.4f}")
    print(f"  dim 0..5: {f[0,:5].tolist()}")

print(f"\n=== 与 llm_hidden 的相似度 ===")
for i, f in enumerate(feats):
    pooled = f.mean(dim=0)
    cos = torch.cosine_similarity(llm_hidden.unsqueeze(0), pooled.unsqueeze(0))
    print(f"  deepstack[{i}] pool → llm_hidden: cos={cos.item():.4f}")

# 不同图同角度的 deepstack 对比
print("\n=== 不同身份同角度的 deepstack 相似度 ===")
z2 = torch.randn(1, bridge.eg3d.z_dim, device='cuda')
ws2 = bridge.eg3d.mapping(z2, cam, truncation_psi=0.7)
img2 = bridge.eg3d.synthesis(ws2, cam)['image']
tmp2 = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
torchvision.utils.save_image(img2[0]*0.5+0.5, tmp2.name)
feats2, _ = get_multimodal_features(bridge.qwen, bridge.processor, "正面平视", tmp2.name)

for i in range(3):
    p1, p2 = feats[i].mean(dim=0), feats2[i].mean(dim=0)
    cos = torch.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0))
    print(f"  deepstack[{i}] 身份1 vs 身份2: cos={cos.item():.4f}")

import os; os.unlink(tmp.name); os.unlink(tmp2.name)
