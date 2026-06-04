"""
Compute within-face r on AFLW2000 real face images.
Compares: L18-20 (best) vs L28 (last) across N real images.
Reports mean +- std + t-test significance.
"""
import os, sys, gc, math, json, time
import numpy as np
from scipy.stats import pearsonr, ttest_rel
from PIL import Image
import torch
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, '/root/Qwen3-VL')
os.environ['TRANSFORMERS_OFFLINE'] = '1'; os.environ['HF_HUB_OFFLINE'] = '1'

BASE = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
IMG_DIR = os.path.join(BASE, "AFLW2000-3D_crop")
LIST_FILE = os.path.join(BASE, "AFLW2000-3D_crop.list")
OUT_DIR = "/root/Qwen3-VL/paper/figures/fixed_test"
os.makedirs(OUT_DIR, exist_ok=True)

N_IMAGES = 200       # AFLW2000 subset (wide pose range)
QUERIES = ["nose", "eyes", "mouth", "chin", "forehead", "beard"]
LAYER_GROUPS = [([17,18,19],"L18-20"), ([27],"L28")]
DEVICE = "cuda"

def compute_within_face_r(attention_maps):
    """attention_maps: [n_queries, nvt] flattened attn maps. Returns mean upper-tri r."""
    nq = len(attention_maps)
    sim = np.zeros((nq, nq))
    for i in range(nq):
        for j in range(nq):
            r, _ = pearsonr(attention_maps[i], attention_maps[j])
            sim[i,j] = r
    upper = [sim[i,j] for i in range(nq) for j in range(i+1, nq)]
    return float(np.mean(upper)), sim

print("="*60)
print("AFLW2000 WITHIN-FACE R (200 real images)")
print("="*60)

# Load Qwen
from transformers import AutoModelForImageTextToText, AutoProcessor
model = AutoModelForImageTextToText.from_pretrained(
    '/root/Qwen3-VL-2B-Instruct', torch_dtype=torch.bfloat16,
    device_map='cuda', attn_implementation='eager',
    local_files_only=True, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(
    '/root/Qwen3-VL-2B-Instruct', local_files_only=True, trust_remote_code=True)
lm = model.model.language_model

# Enable attention hooks for target layers
target_layers = {l for indices,_ in LAYER_GROUPS for l in indices}
for l in target_layers:
    lm.layers[l].self_attn.output_attentions = True

# Load image list
with open(LIST_FILE) as f:
    all_names = [l.strip() for l in f if l.strip()]
step = max(1, len(all_names) // N_IMAGES)
img_names = [all_names[i] for i in range(0, len(all_names), step)][:N_IMAGES]
print(f"Using {len(img_names)} images (uniformly sampled from {len(all_names)})")

# Per-layer-group storage: {name: [within_face_r_per_image]}
results = {name: [] for _, name in LAYER_GROUPS}
t0 = time.time()

for img_idx, img_name in enumerate(img_names):
    if img_idx % 20 == 0:
        print(f"  [{img_idx}/{N_IMAGES}] {time.time()-t0:.0f}s")
    img_path = os.path.join(IMG_DIR, img_name)
    if not os.path.exists(img_path):
        continue
    img = Image.open(img_path).convert("RGB")
    
    # We'll accumulate attention maps for each query
    # {layer_name: {query: attn_map_vector}}
    group_attns = {name: {} for _, name in LAYER_GROUPS}
    
    for query in QUERIES:
        msg = [{"role":"user","content":[{"type":"image","image":img},
                {"type":"text","text":f"describe the {query}"}]}]
        prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = processor(text=prompt, images=img, return_tensors='pt').to(DEVICE)
        ids = inp['input_ids'][0]
        ve = (ids == 151653).nonzero(as_tuple=True)[0].item()
        ie = (ids == 151645).nonzero(as_tuple=True)[0].item()
        nvt = ve - 4
        vs, qs = 4, ve + 1
        
        # Find keyword token
        all_toks = [processor.decode([t]).strip() for t in ids.tolist()]
        kw = [qs + i for i, t in enumerate(all_toks[qs:ie]) if query.lower() in t.lower()]
        
        # Hook for spec attention
        collected = {}
        hooks = []
        for l in sorted(target_layers):
            def mk(lid):
                def hk(m, i, o):
                    if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
                        collected[lid] = o[1].detach()
                return hk
            hooks.append(lm.layers[l].self_attn.register_forward_hook(mk(l)))
        
        with torch.no_grad():
            out = model(**inp, output_attentions=True)
        for h in hooks:
            h.remove()
        
        if collected:
            # Average over heads, extract spec attention
            for indices, name in LAYER_GROUPS:
                if len(indices) == 1:
                    attn = collected[indices[0]]
                else:
                    attn = torch.stack([collected[l] for l in indices]).mean(dim=0)
                # attn: [1, n_heads, q_len, k_len]  
                attn = attn[0].mean(dim=0)  # [q_len, k_len], averaged over heads
                
                if kw:
                    spec = attn[kw, vs:ve].mean(dim=0).cpu().float().numpy()
                else:
                    spec = np.zeros(nvt)
                group_attns[name][query] = spec
        
        del collected, inp
        torch.cuda.empty_cache()
    
    # Compute within-face r for each group
    for name in group_attns:
        if len(group_attns[name]) == len(QUERIES):
            maps = [group_attns[name][q] for q in QUERIES]
            r_val, _ = compute_within_face_r(maps)
            results[name].append(r_val)

# Statistics
print(f"\n{'='*60}")
print("WITHIN-FACE R ON AFLW2000 (200 real images)")
print(f"{'='*60}")
for name in [n for _, n in LAYER_GROUPS]:
    vals = results[name]
    print(f"\n{name} ({len(vals)} images):")
    print(f"  Mean: {np.mean(vals):.4f}")
    print(f"  Std:  {np.std(vals):.4f}")
    print(f"  Min:  {np.min(vals):.4f}")
    print(f"  Max:  {np.max(vals):.4f}")
    print(f"  25%:  {np.percentile(vals, 25):.4f}")
    print(f"  75%:  {np.percentile(vals, 75):.4f}")

# T-test: L18-20 vs L28
r18 = results["L18-20"]
r28 = results["L28"]
if r18 and r28:
    # Pair by image index
    n_paired = min(len(r18), len(r28))
    t_stat, p_val = ttest_rel(r18[:n_paired], r28[:n_paired])
    print(f"\n{'='*60}")
    print(f"PAIRED T-TEST: L18-20 vs L28 (n={n_paired})")
    print(f"  t = {t_stat:.3f}")
    print(f"  p = {p_val:.6f}")
    print(f"  L18-20 mean: {np.mean(r18):.4f}")
    print(f"  L28 mean:    {np.mean(r28):.4f}")
    print(f"  Difference:  {np.mean(r28) - np.mean(r18):.4f} (L28 is higher = more collapse)")

# Save
save_data = {
    "n_images": len(img_names),
    "layer_groups": [n for _, n in LAYER_GROUPS],
    "queries": QUERIES,
    "results": {
        name: {
            "mean": round(float(np.mean(results[name])), 4),
            "std": round(float(np.std(results[name])), 4),
            "min": round(float(np.min(results[name])), 4),
            "max": round(float(np.max(results[name])), 4),
            "q25": round(float(np.percentile(results[name], 25)), 4),
            "q75": round(float(np.percentile(results[name], 75)), 4),
            "values": [round(float(v), 4) for v in results[name]],
        }
        for name in [n for _, n in LAYER_GROUPS] if results[name]
    },
    "t_test": {
        "statistic": round(float(t_stat), 3),
        "p_value": round(float(p_val), 6),
    }
}
save_path = os.path.join(OUT_DIR, "aflw2000_within_face_r.json")
with open(save_path, "w") as f:
    json.dump(save_data, f, indent=2)
print(f"\nSaved: {save_path}")

del model, lm, processor
gc.collect()
torch.cuda.empty_cache()
print(f"\nDone! Total time: {time.time()-t0:.0f}s")
