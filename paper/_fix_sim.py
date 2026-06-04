"""Regenerate similarity matrices with better formatting."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

data = np.load("/root/Qwen3-VL/paper/figures/attention_data.npy", allow_pickle=True)
d = data.item() if hasattr(data, 'item') else data

OUT_DIR = "/root/Qwen3-VL/paper/figures"
EN_QUERIES = ["nose", "eyes", "mouth", "chin", "forehead", "beard",
              "whole image", "bottom-left"]

IMG_MAP = {
    "右侧15度平视.png": "right15",
    "右侧30度平视.png": "right30",
    "右侧45度平视.png": "right45",
}

for cn_name, en_name in IMG_MAP.items():
    queries = d["heatmaps"][cn_name]
    nq = len(EN_QUERIES)
    
    # Compute correlation matrix
    flat_maps = {q: np.array(queries[q]).flatten() for q in list(queries.keys())}
    sim = np.zeros((nq, nq))
    for i in range(nq):
        for j in range(nq):
            r, _ = pearsonr(flat_maps[list(queries.keys())[i]], flat_maps[list(queries.keys())[j]])
            sim[i, j] = r
    
    # Print actual values for paper
    print(f"\n=== {cn_name} ===")
    print(f"  Min non-diag r: {sim[~np.eye(nq, dtype=bool)].min():.6f}")
    print(f"  Max non-diag r: {sim[~np.eye(nq, dtype=bool)].max():.6f}")
    
    # Figure: show as "1 - r" for visibility, or use more decimal places
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")
    
    # Panel 1: Raw r values with 4 decimal places
    ax = axes[0]
    im = ax.imshow(sim, cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(nq))
    ax.set_yticks(range(nq))
    ax.set_xticklabels(EN_QUERIES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(EN_QUERIES, fontsize=8)
    for i in range(nq):
        for j in range(nq):
            ax.text(j, i, f"{sim[i,j]:.4f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Pearson r (raw)", fontsize=11)
    
    # Panel 2: 1 - r to amplify differences
    ax = axes[1]
    diff = 1.0 - sim
    im = ax.imshow(diff, cmap="YlOrRd", vmin=0, vmax=0.001)
    ax.set_xticks(range(nq))
    ax.set_yticks(range(nq))
    ax.set_xticklabels(EN_QUERIES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(EN_QUERIES, fontsize=8)
    for i in range(nq):
        for j in range(nq):
            val = diff[i, j]
            if val > 0.00001:
                ax.text(j, i, f"{val:.6f}", ha="center", va="center", fontsize=6, color="black")
    plt.colorbar(im, ax=ax, label="1 $-$ r")
    ax.set_title("1 $-$ r (amplified)", fontsize=11)
    
    plt.suptitle(f"Cross-Query Attention Similarity ({en_name})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, f"similarity_{en_name}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")

print("\nDone!")
