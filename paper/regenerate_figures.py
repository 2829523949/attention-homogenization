"""Regenerate similarity matrix figures with English labels."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVIDENCE_DIR = "/root/Qwen3-VL/attention_evidence"
OUT_DIR = "/root/Qwen3-VL/paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)

EN_QUERIES = ["nose", "eyes", "mouth", "chin", "forehead", "beard",
              "whole image", "bottom-left"]

with open(os.path.join(EVIDENCE_DIR, "similarity_data.json")) as f:
    data = json.load(f)

IMG_MAP = {
    "右侧15度平视.png": "right15",
    "右侧30度平视.png": "right30",
    "右侧45度平视.png": "right45",
}

for cn_name, en_name in IMG_MAP.items():
    sim = np.array(data["cross_similarities"][cn_name])
    nq = len(EN_QUERIES)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")
    im = ax.imshow(sim, cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(nq))
    ax.set_yticks(range(nq))
    ax.set_xticklabels(EN_QUERIES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(EN_QUERIES, fontsize=9)
    for i in range(nq):
        for j in range(nq):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center", fontsize=6.5)
    plt.colorbar(im, ax=ax, label="Pearson r")
    plt.title("Cross-Query Attention Similarity", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, f"similarity_{en_name}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")

print("Done!")
