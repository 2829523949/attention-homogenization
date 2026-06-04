"""
Layer-wise Ablation: VLM Self-Attention at Early / Middle / Late Layers.

Layer groups (28 layers total, 0-indexed):
  A (Early):   layers 0-2  (1-indexed: 1-3)
  B (Middle):  layers 7-9  (1-indexed: 8-10)
  C (Later):   layers 17-19 (1-indexed: 18-20)
  D (Last):    layers 25-27 (1-indexed: 26-28, same as original paper)

Expected: Early layers show LOWER cross-query similarity (better spatial differentiation),
supporting "semantic abstraction deepens in later layers".
"""
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

# ── Layer groups ──
LAYER_GROUPS = {
    "A_Early (L1-3)":  [0, 1, 2],
    "B_Mid (L8-10)":   [7, 8, 9],
    "C_Later (L18-20)":[17, 18, 19],
    "D_Last (L25-27)": [25, 26, 27],
}
GROUP_NAMES_SHORT = {k: k.split(" (")[0] for k in LAYER_GROUPS}

# ── Load model ──
print("Loading Qwen3-VL...")
from transformers import AutoModelForImageTextToText, AutoProcessor
model = AutoModelForImageTextToText.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
    device_map="cuda", attn_implementation="eager",
    local_files_only=True, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
lm = model.model.language_model
print(f"Model loaded. {len(lm.layers)} LM layers.")

# ── Queries (English, same as paper) ──
QUERIES = ["nose", "eyes", "mouth", "chin", "forehead", "beard",
           "whole image", "bottom-left area"]

# ── Images ──
IMG_DIR = "/root/output_phase1_test"
image_files = sorted([f for f in os.listdir(IMG_DIR)
                      if f.endswith(".png") and f != "grid_9views.png"])[:3]

# ═══════════════════════════════════════════════
#  Main experiment
# ═══════════════════════════════════════════════
results = {gk: {} for gk in LAYER_GROUPS}

for fname in image_files:
    img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
    print(f"\n{'='*60}")
    print(f"Image: {fname}")

    for gk, layer_indices in LAYER_GROUPS.items():
        print(f"\n  ── Group: {gk} (layers {layer_indices}) ──")
        img_results = {}

        for query in QUERIES:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"describe the {query}"}
            ]}]
            prompt = processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
            inputs = processor(text=prompt, images=img,
                               return_tensors="pt").to("cuda")
            input_ids = inputs["input_ids"][0]

            # ── Correct token indexing (from run_attention_en.py) ──
            T_VISION_END = 151653
            vis_end_pos = (input_ids == T_VISION_END).nonzero(as_tuple=True)[0].item()
            T_IM_END = 151645
            im_end_pos = (input_ids == T_IM_END).nonzero(as_tuple=True)[0].item()
            vis_start = 4
            vis_end = vis_end_pos
            nvt = vis_end - vis_start
            nvt_dim = int(nvt ** 0.5)
            q_start = vis_end_pos + 1
            q_end = im_end_pos

            if nvt != nvt_dim * nvt_dim:
                nvt_dim = int(np.sqrt(nvt))

            # ── Hook only the layers in this group ──
            attn_weights = []

            def make_hook(weights_list):
                def hook(m, i, o):
                    if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
                        weights_list.append(o[1].detach())
                return hook

            handles = []
            for lidx in layer_indices:
                handles.append(
                    lm.layers[lidx].self_attn.register_forward_hook(
                        make_hook(attn_weights))
                )

            with torch.no_grad():
                out = model(**inputs, output_attentions=True)
            for h in handles:
                h.remove()

            if attn_weights:
                attn = torch.stack(attn_weights).mean(dim=0)
                attn = attn.mean(dim=[0, 1])
                vis_attn = attn[q_start:q_end, vis_start:vis_end].mean(dim=0).cpu().float().numpy()
                heatmap = vis_attn.reshape(nvt_dim, nvt_dim)
            else:
                heatmap = np.zeros((nvt_dim, nvt_dim))

            img_results[query] = heatmap

        results[gk][fname] = img_results

        # ── Stats for this group on this image ──
        flat_maps = {q: img_results[q].flatten() for q in QUERIES}
        nq = len(QUERIES)
        sim = np.zeros((nq, nq))
        for i in range(nq):
            for j in range(nq):
                r, _ = pearsonr(flat_maps[QUERIES[i]], flat_maps[QUERIES[j]])
                sim[i, j] = r

        face_sims = [sim[i, j] for i in range(6) for j in range(i+1, 6)]
        fc_sims = [sim[i, j] for i in range(6) for j in range(6, 8)]
        cc_sim = sim[6, 7]

        print(f"    Within-face mean r: {np.mean(face_sims):.4f}  "
              f"face-vs-control mean r: {np.mean(fc_sims):.4f}  "
              f"ctrl-ctrl: {cc_sim:.4f}")
        results[gk][fname + "_sim"] = sim.tolist()

# ═══════════════════════════════════════════════
#  Aggregate stats across all images
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"{'SUMMARY: Layer-wise Cross-Query Similarity':^60s}")
print(f"{'='*60}")

header = f"{'Group':<22s} {'Within-Face r':>18s} {'Face-vs-Ctrl r':>20s} {'Ctrl-Ctrl r':>15s}"
print(header)
print("  " + "─" * 75)

agg_stats = {}
for gk in LAYER_GROUPS:
    all_face = []
    all_fc = []
    all_cc = []
    for fname in image_files:
        sim = np.array(results[gk][fname + "_sim"])
        all_face.extend([sim[i, j] for i in range(6) for j in range(i+1, 6)])
        all_fc.extend([sim[i, j] for i in range(6) for j in range(6, 8)])
        all_cc.append(sim[6, 7])
    face_m, face_s = np.mean(all_face), np.std(all_face)
    fc_m, fc_s = np.mean(all_fc), np.std(all_fc)
    cc_m = np.mean(all_cc)
    print(f"  {gk:<22s} {face_m:>10.4f} +/- {face_s:.4f}  {fc_m:>10.4f} +/- {fc_s:.4f}  {cc_m:>8.4f}")
    agg_stats[gk] = {
        "layers": LAYER_GROUPS[gk],
        "within_face_mean": face_m, "within_face_std": face_s,
        "face_vs_ctrl_mean": fc_m, "face_vs_ctrl_std": fc_s,
        "ctrl_ctrl_mean": cc_m,
    }

with open(os.path.join(OUT_DIR, "layer_ablation_stats.json"), "w") as f:
    json.dump(agg_stats, f, indent=2)
print(f"\nSaved: layer_ablation_stats.json")

# ═══════════════════════════════════════════════
#  Figures: per-group similarity matrices
# ═══════════════════════════════════════════════
for fname in image_files:
    base = fname.replace(".png", "")
    for gk in LAYER_GROUPS:
        sim = np.array(results[gk][fname + "_sim"])
        nq = len(QUERIES)

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
        short = GROUP_NAMES_SHORT[gk]
        ax.set_title(f"Layer Group {short} -- Cross-Query Attention Similarity",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        p = os.path.join(OUT_DIR, f"similarity_{base}_{short}.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p}")

# ═══════════════════════════════════════════════
#  4-panel comparison figure
# ═══════════════════════════════════════════════
gks = list(LAYER_GROUPS.keys())
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.patch.set_facecolor("white")
axes = axes.flatten()

for idx, gk in enumerate(gks):
    ax = axes[idx]
    sim_sum = None
    count = 0
    for fname in image_files:
        s = np.array(results[gk][fname + "_sim"])
        if sim_sum is None:
            sim_sum = s.copy()
        else:
            sim_sum += s
        count += 1
    sim_avg = sim_sum / count

    im = ax.imshow(sim_avg, cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(nq)); ax.set_yticks(range(nq))
    ax.set_xticklabels(QUERIES, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(QUERIES, fontsize=7)
    for i in range(nq):
        for j in range(nq):
            ax.text(j, i, f"{sim_avg[i,j]:.3f}", ha="center", va="center", fontsize=5.5)
    rect = plt.Rectangle((-0.5, -0.5), 6, 6, fill=False,
                         edgecolor="cyan", linewidth=2, linestyle="--")
    ax.add_patch(rect)
    ax.set_title(f"{gk}", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")

plt.suptitle("Layer-wise Ablation: VLM Self-Attention Cross-Query Similarity (avg over 3 images)",
             fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "layer_ablation_4panel.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: layer_ablation_4panel.png")

# ═══════════════════════════════════════════════
#  Bar chart
# ═══════════════════════════════════════════════
group_labels = [gk.replace("_", " ") for gk in gks]
face_means = [agg_stats[gk]["within_face_mean"] for gk in gks]
face_stds = [agg_stats[gk]["within_face_std"] for gk in gks]
fc_means = [agg_stats[gk]["face_vs_ctrl_mean"] for gk in gks]
fc_stds = [agg_stats[gk]["face_vs_ctrl_std"] for gk in gks]

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("white")
x = np.arange(len(gks))
w = 0.35
bars1 = ax.bar(x - w/2, face_means, w, yerr=face_stds, capsize=4,
               label="Within-face", color="steelblue")
bars2 = ax.bar(x + w/2, fc_means, w, yerr=fc_stds, capsize=4,
               label="Face vs Control", color="indianred")
ax.set_xticks(x)
ax.set_xticklabels(group_labels, fontsize=10)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("Pearson r", fontsize=12)
ax.set_title("Cross-Query Similarity by Layer Group", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

for bar, val in zip(bars1, face_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8)
for bar, val in zip(bars2, fc_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "layer_ablation_bars.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: layer_ablation_bars.png")

print(f"\n{'='*60}")
print("DONE! All figures + stats saved.")
print(f"Results: {OUT_DIR}/")
