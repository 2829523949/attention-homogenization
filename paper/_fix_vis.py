"""Regenerate heatmap figures with better visualization."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

data = np.load("/root/Qwen3-VL/paper/figures/attention_data.npy", allow_pickle=True)
d = data.item() if hasattr(data, 'item') else data

OUT_DIR = "/root/Qwen3-VL/paper/figures"
EN_QUERIES = ["nose", "eyes", "mouth", "chin", "forehead", "beard",
              "whole image", "bottom-left area"]

IMG_MAP = {
    "右侧15度平视.png": "right15",
    "右侧30度平视.png": "right30",
    "右侧45度平视.png": "right45",
}

for cn_name, en_name in IMG_MAP.items():
    queries = d["heatmaps"][cn_name]
    nq = len(EN_QUERIES)
    cols, rows = 4, 2
    fig, axes = plt.subplots(rows, cols, figsize=(3.5*cols, 3*rows))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()
    
    for idx, query in enumerate(EN_QUERIES):
        ax = axes[idx]
        arr = np.array(queries[query])
        
        # Use log scale and percentile-based normalization
        vmin = np.percentile(arr, 5)
        vmax = np.percentile(arr, 99.5)
        
        im = ax.imshow(arr, cmap="inferno", interpolation="bilinear",
                       vmin=vmin, vmax=vmax)
        ax.set_title(f'"{query}"', fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, format="%.4f")
    
    for idx in range(nq, len(axes)):
        axes[idx].axis("off")
    
    plt.suptitle(f"Qwen3-VL Self-Attention ({en_name})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, f"heatmap_{en_name}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")

print("Done!")
