"""对比 Qwen ViT deepstack vs DINOv2 (简化, 用已有数据集图片)"""
import sys, torch
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, get_multimodal_features

b = QwenEG3DBridge()

# 从数据集中取两张不同人同一角度的图
data = torch.load("/root/edit_data/dataset.pt")
# 身份0, 正面平视
s1 = [s for s in data['samples'] if s['zid']==0 and s['yaw']==0.0 and s['pitch_offset']==0.0][0]
# 身份1, 正面平视
s2 = [s for s in data['samples'] if s['zid']==1 and s['yaw']==0.0 and s['pitch_offset']==0.0][0]

p1 = f"/root/edit_data/images/{s1['image']}"
p2 = f"/root/edit_data/images/{s2['image']}"

for label, path in [("身份0", p1), ("身份1", p2)]:
    d_feats, _ = get_multimodal_features(b.qwen, b.processor, "test", path, b.dino, b.dino_processor)
    if label == "身份0":
        d1 = d_feats
    else:
        d2 = d_feats

print(f"DINOv2 特征形状: {[f.shape for f in d1]}")
for i in range(3):
    cos_pool = torch.cosine_similarity(d1[i].mean(0).unsqueeze(0), d2[i].mean(0).unsqueeze(0)).item()
    cos_p0 = torch.cosine_similarity(d1[i][0].unsqueeze(0), d2[i][0].unsqueeze(0)).item()
    cos_all = torch.cosine_similarity(d1[i], d2[i]).mean().item()
    print(f"DINOv2[{i}]: pool={cos_pool:.4f} patch[0]={cos_p0:.4f} avg_patch={cos_all:.4f}")
