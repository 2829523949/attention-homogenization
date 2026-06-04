"""对比 Qwen ViT vs DINOv2 per-patch 身份区分度"""
import sys, torch
sys.path.insert(0, "/root/Qwen3-VL")
from bridge import QwenEG3DBridge, get_multimodal_features

b = QwenEG3DBridge()
data = torch.load("/root/edit_data/dataset.pt")

s1 = [s for s in data['samples'] if s['zid']==0 and s['yaw']==0.0 and s['pitch_offset']==0.0][0]
s2 = [s for s in data['samples'] if s['zid']==1 and s['yaw']==0.0 and s['pitch_offset']==0.0][0]

p1 = f"/root/edit_data/images/{s1['image']}"
p2 = f"/root/edit_data/images/{s2['image']}"

# DINOv2 (current)
for label, path in [("id0", p1), ("id1", p2)]:
    dino_ds, _ = get_multimodal_features(b.qwen, b.processor, "test", path, b.dino, b.dino_processor)
    if label == "id0": d1 = dino_ds
    else: d2 = dino_ds

# Qwen ViT (单独提取)
from PIL import Image
imsgs = [{"role":"user","content":[{"type":"image","image":p1},{"type":"text","text":"test"}]}]
inputs = b.processor(text=[b.processor.apply_chat_template(imsgs, tokenize=False, add_generation_prompt=True)], images=[Image.open(p1).convert('RGB')], return_tensors="pt", padding=True).to('cuda')
with torch.no_grad():
    pv, gt = inputs.get('pixel_values'), inputs.get('image_grid_thw')
    if pv is not None:
        vo = b.qwen.model.visual(hidden_states=pv, grid_thw=gt)
        qwen1 = vo.deepstack_features  # list[3] of [N_v, 2048]

imsgs[0]["content"][0]["image"] = p2
inputs2 = b.processor(text=[b.processor.apply_chat_template(imsgs, tokenize=False, add_generation_prompt=True)], images=[Image.open(p2).convert('RGB')], return_tensors="pt", padding=True).to('cuda')
with torch.no_grad():
    pv2, gt2 = inputs2.get('pixel_values'), inputs2.get('image_grid_thw')
    if pv2 is not None:
        vo2 = b.qwen.model.visual(hidden_states=pv2, grid_thw=gt2)
        qwen2 = vo2.deepstack_features

print(f"\n{'layer':10s} {'Qwen pool':>10s} {'Qwen patch[0]':>14s} {'Qwen avg_patch':>14s} {'DINO pool':>10s} {'DINO patch[0]':>14s} {'DINO avg_patch':>14s}")
print("-" * 90)
for i in range(3):
    q_pool = torch.cosine_similarity(qwen1[i].mean(0).unsqueeze(0), qwen2[i].mean(0).unsqueeze(0)).item()
    q_p0 = torch.cosine_similarity(qwen1[i][0].unsqueeze(0), qwen2[i][0].unsqueeze(0)).item()
    q_avg = torch.cosine_similarity(qwen1[i], qwen2[i]).mean().item()
    d_pool = torch.cosine_similarity(d1[i].mean(0).unsqueeze(0), d2[i].mean(0).unsqueeze(0)).item()
    d_p0 = torch.cosine_similarity(d1[i][0].unsqueeze(0), d2[i][0].unsqueeze(0)).item()
    d_avg = torch.cosine_similarity(d1[i], d2[i]).mean().item()
    print(f"deep[{i}]:   {q_pool:.4f}      {q_p0:.4f}        {q_avg:.4f}       {d_pool:.4f}      {d_p0:.4f}        {d_avg:.4f}")
