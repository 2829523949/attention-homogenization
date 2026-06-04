"""Ablation: Text encoder sensitivity to fine-grained pose differences
Compare how VLM vs CLIP distinguish similar pose descriptions in embedding space.
"""

import sys, os, json, math
sys.path.insert(0, "/root/Qwen3-VL")
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
import torch.nn.functional as F
import numpy as np

# Prompts that differ ONLY in angle
ANGLES = list(range(0, 91, 10))
PROMPTS = [f"A person looking to the right at {a} degrees" for a in ANGLES]
print(f"Testing {len(PROMPTS)} prompts with angles: {ANGLES}")

# 1. VLM hidden states
print("\n[1/3] Extracting VLM hidden states...")
from bridge import get_llm_hidden, load_qwen
qwen, processor = load_qwen()
qwen.eval()

vlm_h = []
for text in PROMPTS:
    h = get_llm_hidden(qwen, processor, text)
    vlm_h.append(h)
vlm_h = torch.stack(vlm_h)
print(f"  VLM: {vlm_h.shape}")
del qwen
torch.cuda.empty_cache()

# 2. CLIP embeddings
print("\n[2/3] Extracting CLIP embeddings...")
from lightweight_bridge import LightweightBridge, get_embedding_for_training
clip_b = LightweightBridge(encoder_name='clip')
ckpt = torch.load("/root/Qwen3-VL/checkpoints/clip_en_epoch_080.pt", map_location="cuda")
clip_b.projection.load_state_dict(ckpt["projection"])
clip_b.pose.load_state_dict(ckpt["pose"])
clip_b.eval()

clip_e = []
for text in PROMPTS:
    emb = get_embedding_for_training(clip_b, text)
    clip_e.append(emb.squeeze(0))
clip_e = torch.stack(clip_e)
print(f"  CLIP: {clip_e.shape}")

# 3. Compute pairwise cosine similarities
def pair_cos(mat):
    norm = F.normalize(mat, dim=1)
    return norm @ norm.T

def off_diag_mean(mat):
    N = mat.shape[0]
    return mat[~torch.eye(N, dtype=bool, device=mat.device)].mean().item()

print("\n[3/3] Computing similarities...")
vlm_s = pair_cos(vlm_h)
clip_s = pair_cos(clip_e)

vlm_m = off_diag_mean(vlm_s)
clip_m = off_diag_mean(clip_s)

adj_v = vlm_s.diagonal(offset=1).mean().item()
adj_c = clip_s.diagonal(offset=1).mean().item()
far_v = vlm_s[torch.triu(torch.ones(10,10), diagonal=4).bool()].mean().item()
far_c = clip_s[torch.triu(torch.ones(10,10), diagonal=4).bool()].mean().item()

# Summary
print(f"\n{'='*60}")
print(f"  {'Metric':30s} {'VLM 2B':>12s} {'CLIP 63M':>12s}")
print(f"  {'─'*56}")
print(f"  {'Mean pairwise similarity':30s} {vlm_m:>12.4f} {clip_m:>12.4f}")
print(f"  {'Adjacent (10°) similarity':30s} {adj_v:>12.4f} {adj_c:>12.4f}")
print(f"  {'Far (40°+) similarity':30s} {far_v:>12.4f} {far_c:>12.4f}")
print(f"  {'Discriminability (adj-far)':30s} {adj_v-far_v:>12.4f} {adj_c-far_c:>12.4f}")
print(f"  {'Embedding dimension':30s} {2048:>12d} {512:>12d}")
print(f"{'='*60}")

winner = "CLIP" if clip_m < vlm_m else "VLM"
print(f"\n  → {winner} has MORE diverse (lower similarity) embeddings across angles")

# Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.patch.set_facecolor('white')
titles = ['VLM 2B Hidden States', 'CLIP 63M Embeddings']
mats = [vlm_s.cpu().numpy(), clip_s.cpu().numpy()]

for ax, mat, title in zip(axes, mats, titles):
    im = ax.imshow(mat, cmap='viridis', vmin=0.5, vmax=1.0)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    labels = [f'{a}°' for a in ANGLES]
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=12)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=6)
    plt.colorbar(im, ax=ax)

fig.suptitle('Ablation: Encoder Sensitivity to Fine-Grained Pose Descriptions',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("/root/Qwen3-VL/paper_figures/encoding_sensitivity.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"\nFigure: paper_figures/encoding_sensitivity.png")

# Save JSON
results = {
    "angles": ANGLES,
    "vlm": {"dim": 2048, "mean_sim": vlm_m, "adj_sim": adj_v, "far_sim": far_v, "discrim": adj_v-far_v},
    "clip": {"dim": 512, "mean_sim": clip_m, "adj_sim": adj_c, "far_sim": far_c, "discrim": adj_c-far_c},
}
with open("/root/Qwen3-VL/checkpoints/encoding_sensitivity.json", 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: checkpoints/encoding_sensitivity.json")
print("Done!")
