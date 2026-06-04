"""
Cross-Attention Spatial Grounding Experiment (Paper Evidence)
Hypothesis: Qwen3-VL cross-attention cannot spatially localize facial features.
Evidence saved to /root/Qwen3-VL/attention_evidence/
"""
import os, sys, math, json, torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, '/root/Qwen3-VL')
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

EVIDENCE_DIR = '/root/Qwen3-VL/attention_evidence'
os.makedirs(EVIDENCE_DIR, exist_ok=True)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)

print("Loading Qwen3-VL...")
from transformers import AutoModelForImageTextToText, AutoProcessor
model = AutoModelForImageTextToText.from_pretrained(
    '/root/Qwen3-VL-2B-Instruct', torch_dtype=torch.bfloat16,
    device_map='cuda', attn_implementation='eager',
    local_files_only=True, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(
    '/root/Qwen3-VL-2B-Instruct', local_files_only=True, trust_remote_code=True)
print("Model loaded.")

# Load test images
IMG_DIR = '/root/output_phase1_test'
image_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.png') and f != 'grid_9views.png'])
test_images = []
for fname in image_files[:3]:
    img = Image.open(os.path.join(IMG_DIR, fname)).convert('RGB')
    test_images.append((fname, img))
    print(f"Image: {fname} {img.size}")

QUERIES_CN = ["鼻子", "眼睛", "嘴巴", "下巴", "额头", "胡须"]
QUERIES_CTRL = ["整个画面", "左下角区域"]
all_queries = QUERIES_CN + QUERIES_CTRL

def get_visual_token_count(image, processor):
    # Use the actual processor to count visual tokens
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "test"}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors='pt')
    if 'image_grid_thw' in inputs:
        g = inputs['image_grid_thw'][0]
        # Spatial merge divides each dimension by 2
        h_patches = g[1].item() // 2  # 32//2=16
        w_patches = g[2].item() // 2  # 32//2=16
        return h_patches * w_patches, h_patches
    return 256, 16

results = {}
for img_idx, (fname, img) in enumerate(test_images):
    print(f"\n--- Image {img_idx+1}: {fname} ---")
    nvt, nvt_dim = get_visual_token_count(img, processor)
    # Store grid info for later use
    result_img = {}
    
    for query in all_queries:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": f"请描述{query}"}
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, images=img, return_tensors='pt').to('cuda')
        
        attn_weights = []
        def hook(m, i, o):
            if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
                attn_weights.append(o[1].detach())
        
        lm = model.model.language_model
        handles = []
        for layer in lm.layers[-3:]:
            handles.append(layer.self_attn.register_forward_hook(hook))
        
        with torch.no_grad():
            out = model(**inputs, output_attentions=True)
        
        for h in handles:
            h.remove()
        
        nvt_h = nvt_w = nvt_dim
        if attn_weights:
            attn = torch.stack(attn_weights).mean(dim=0)
            attn = attn.mean(dim=[0, 1])  # [q_len, k_len]
            # Visual tokens are first nvt positions, text tokens are last
            # Take last 2 text tokens' attention to first nvt visual tokens
            vis_attn = attn[-3:-1, :nvt].mean(dim=0).cpu().float().numpy()
            heatmap = vis_attn.reshape(nvt_h, nvt_w)
        else:
            print(f"  '{query}': NO attention!")
            heatmap = np.zeros((nvt_h, nvt_w))
        
        result_img[query] = heatmap
        print(f"  '{query}': [{heatmap.min():.4f}, {heatmap.max():.4f}] shape={heatmap.shape}")
    
    results[fname] = result_img

# Compute cross-query similarity
print("\n--- Cross-query similarity ---")
all_similarities = {}
for fname, result_img in results.items():
    flat_maps = {q: hm.flatten() for q, hm in result_img.items()}
    nq = len(all_queries)
    sim_matrix = np.zeros((nq, nq))
    for i in range(nq):
        for j in range(nq):
            qi, qj = all_queries[i], all_queries[j]
            if flat_maps[qi].size == flat_maps[qj].size:
                r, _ = pearsonr(flat_maps[qi], flat_maps[qj])
                sim_matrix[i, j] = r
            else:
                sim_matrix[i, j] = -2
    all_similarities[fname] = sim_matrix.tolist()
    
    print(f"\n  {fname}:")
    feat_names = QUERIES_CN
    fi = [all_queries.index(q) for q in feat_names]
    print("  Face features cross-correlation:")
    for i in range(len(feat_names)):
        row = "  ".join([f"{sim_matrix[fi[i], fi[j]]:.3f}" for j in range(len(feat_names))])
        print(f"    {feat_names[i]}: {row}")
    
    feat_sims = [sim_matrix[fi[i], fi[j]] for i in range(len(fi)) for j in range(i+1, len(fi))]
    print(f"  Mean face feature similarity: {np.mean(feat_sims):.4f}")
    
    ci = [all_queries.index(q) for q in QUERIES_CTRL]
    fc_sims = [sim_matrix[fii, cjj] for fii in fi for cjj in ci]
    print(f"  Face vs control mean sim: {np.mean(fc_sims):.4f}")

# Save evidence
print("\n--- Saving evidence ---")
data = {
    'queries': all_queries,
    'face_features': QUERIES_CN,
    'cross_similarities': all_similarities,
}
json_path = os.path.join(EVIDENCE_DIR, 'similarity_data.json')
with open(json_path, 'w') as f:
    json.dump(data, f, indent=2, cls=NumpyEncoder)
print(f"Saved: {json_path}")

# Heatmap images per image
for fname, result_img in results.items():
    base = fname.replace('.png', '')
    nq = len(all_queries)
    cols = min(4, nq)
    rows = (nq + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5*cols, 3*rows))
    axes = axes.flatten()
    for idx, query in enumerate(all_queries):
        ax = axes[idx]
        hm = result_img[query]
        im = ax.imshow(hm, cmap='hot', interpolation='nearest')
        ax.set_title(f'"{query}"', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    for idx in range(nq, len(axes)):
        axes[idx].axis('off')
    plt.suptitle(f'Qwen3-VL Cross-Attention: {base}', fontsize=12)
    plt.tight_layout()
    p = os.path.join(EVIDENCE_DIR, f'heatmap_{base}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p}")

# Similarity matrix heatmap
for fname in results:
    base = fname.replace('.png', '')
    sim = np.array(all_similarities[fname])
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(sim, cmap='viridis', vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(all_queries)))
    ax.set_yticks(range(len(all_queries)))
    ax.set_xticklabels(all_queries, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(all_queries, fontsize=8)
    for i in range(len(all_queries)):
        for j in range(len(all_queries)):
            ax.text(j, i, f'{sim[i,j]:.2f}', ha='center', va='center', fontsize=6)
    plt.colorbar(im, ax=ax, label='Pearson r')
    plt.title(f'Cross-Query Attention Similarity: {base}', fontsize=11)
    plt.tight_layout()
    p = os.path.join(EVIDENCE_DIR, f'similarity_{base}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p}")

print(f"\nAll evidence saved to {EVIDENCE_DIR}/")
