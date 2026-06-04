"""Cross-Attention experiment - FIXED version with correct token indices."""
import os, sys, json, torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

OUT_DIR = "/root/Qwen3-VL/paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading Qwen3-VL...")
from transformers import AutoModelForImageTextToText, AutoProcessor
model = AutoModelForImageTextToText.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
    device_map="cuda", attn_implementation="eager",
    local_files_only=True, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
print("Model loaded.")

QUERIES = ["nose", "eyes", "mouth", "chin", "forehead", "beard",
           "whole image", "bottom-left area"]

IMG_DIR = "/root/output_phase1_test"
image_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".png") and f != "grid_9views.png"])[:3]

all_heatmaps = {}

for fname in image_files:
    img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
    print(f"\n--- {fname} ---")
    
    results = {}
    for query in QUERIES:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": f"describe the {query}"}
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"][0]
        
        # === CORRECT TOKEN INDEXING ===
        # Find vision_end token (id=151653)
        T_VISION_END = 151653
        vis_end_pos = (input_ids == T_VISION_END).nonzero(as_tuple=True)[0].item()
        
        # Find <|im_end|> (id=151645) - the end of user message
        T_IM_END = 151645
        im_end_pos = (input_ids == T_IM_END).nonzero(as_tuple=True)[0].item()
        
        # Visual tokens: after <|vision_start|> (position 3), before <|vision_end|>
        vis_start = 4  # position after <|vision_start|>
        vis_end = vis_end_pos  # position of <|vision_end|>
        nvt = vis_end - vis_start
        nvt_dim = int(nvt ** 0.5)
        
        # Query text tokens: after <|vision_end|>, before <|im_end|>
        q_start = vis_end_pos + 1
        q_end = im_end_pos
        
        if nvt != nvt_dim * nvt_dim:
            # Try different grid size
            nvt_dim = int(np.sqrt(nvt))
        
        # Extract attention
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
        
        if attn_weights:
            attn = torch.stack(attn_weights).mean(dim=0)
            attn = attn.mean(dim=[0, 1])
            # Take query tokens' attention to visual tokens
            vis_attn = attn[q_start:q_end, vis_start:vis_end].mean(dim=0).cpu().float().numpy()
            heatmap = vis_attn.reshape(nvt_dim, nvt_dim)
        else:
            heatmap = np.zeros((nvt_dim, nvt_dim))
        
        results[query] = heatmap
        print(f"  '{query}': vis=[{vis_start}:{vis_end}]({nvt}) q=[{q_start}:{q_end}] "
              f"heatmap=[{heatmap.min():.4f},{heatmap.max():.4f}]")
    
    all_heatmaps[fname] = results

# Generate figures
print("\nGenerating figures...")
for fname in image_files:
    base = fname.replace(".png", "")
    flat_maps = {q: all_heatmaps[fname][q].flatten() for q in QUERIES}
    nq = len(QUERIES)
    
    # Similarity matrix
    sim = np.zeros((nq, nq))
    for i in range(nq):
        for j in range(nq):
            r, _ = pearsonr(flat_maps[QUERIES[i]], flat_maps[QUERIES[j]])
            sim[i, j] = r
    
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")
    im = ax.imshow(sim, cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(nq)); ax.set_yticks(range(nq))
    ax.set_xticklabels(QUERIES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(QUERIES, fontsize=9)
    for i in range(nq):
        for j in range(nq):
            ax.text(j, i, f"{sim[i,j]:.4f}", ha="center", va="center", fontsize=6.5)
    plt.colorbar(im, ax=ax, label="Pearson r")
    plt.title("Cross-Query Attention Similarity", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"similarity_{base}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # Heatmap grid
    cols, rows = 4, 2
    fig, axes = plt.subplots(rows, cols, figsize=(3.5*cols, 3*rows))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()
    for idx, query in enumerate(QUERIES):
        ax = axes[idx]
        arr = all_heatmaps[fname][query]
        vmin, vmax = np.percentile(arr, 5), np.percentile(arr, 99.5)
        im = ax.imshow(arr, cmap="inferno", interpolation="bilinear", vmin=vmin, vmax=vmax)
        ax.set_title(f'"{query}"', fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, format="%.4f")
    for idx in range(nq, len(axes)):
        axes[idx].axis("off")
    plt.suptitle(f"Qwen3-VL Self-Attention ({base})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"heatmap_{base}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  heatmap_{base}.png + similarity_{base}.png")

# Summary
all_r = []
for fname in image_files:
    flat_maps = {q: all_heatmaps[fname][q].flatten() for q in QUERIES}
    for i, q1 in enumerate(QUERIES):
        for j, q2 in enumerate(QUERIES):
            if i < j:
                r, _ = pearsonr(flat_maps[q1], flat_maps[q2])
                all_r.append(r)

print(f"\n=== FIXED RESULT ===")
print(f"Min cross-query r: {min(all_r):.6f}")
print(f"Max cross-query r: {max(all_r):.6f}")
print(f"Mean cross-query r: {np.mean(all_r):.6f}")
print(f"Standard deviation: {np.std(all_r):.6f}")
print("Done!")
