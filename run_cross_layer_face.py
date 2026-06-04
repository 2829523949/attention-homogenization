"""
Cross-layer Face Attention: run face attention at 4 layer groups.
Uses original proven methodology: image tokens at [4:ve], text tokens at [ve+1:ie].
Usage: python run_cross_layer_face.py
"""
import os, sys, math, json, torch, numpy as np
from PIL import Image
from scipy.stats import pearsonr
sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"; os.environ["HF_HUB_OFFLINE"] = "1"
OUT = "/root/Qwen3-VL/paper/figures/fixed_test"; os.makedirs(OUT, exist_ok=True)

LAYER_GROUPS = {
    "L1-3":  [0, 1, 2],
    "L8-10": [7, 8, 9],
    "L18-20":[17, 18, 19],
    "L25-27":[25, 26, 27],
}

QUERIES = [
    ("nose", "nose"), ("eyes", "eye"), ("mouth", "mouth"),
    ("chin", "chin"), ("forehead", "forehead"), ("beard", "beard"),
    ("whole image", "image"), ("bottom-left area", "area"),
]

IMG_DIR = "/root/output_phase1_test"
images = sorted([f for f in os.listdir(IMG_DIR)
                if f.endswith((".png",".jpg",".jpeg")) and f!="grid_9views.png"])[:3]

# Load model once
from transformers import AutoModelForImageTextToText, AutoProcessor
model = AutoModelForImageTextToText.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
    device_map="cuda", attn_implementation="eager",
    local_files_only=True, trust_remote_code=True)
proc = AutoProcessor.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
lm = model.model.language_model

def find_keyword_pos(tok_strs, keyword):
    """Find token positions where keyword appears."""
    positions = []
    for j, t in enumerate(tok_strs):
        clean_t = t.replace("\u0120","").replace("\u2581","").strip()
        if keyword.lower() in clean_t.lower():
            positions.append(j)
    return positions

def run_group(layers, images):
    """Run face experiment for one layer group, return results dict."""
    # Enable attentions
    for l in layers: lm.layers[l].self_attn.output_attentions = True
    
    results = {"spec": {}, "avg": {}, "last": {}}
    
    for fname in images:
        img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
        
        for q_text, keyword in QUERIES:
            msg = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":f"describe the {q_text}"}]}]
            pr = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inp = proc(text=pr, images=img, return_tensors="pt").to("cuda")
            ids = inp["input_ids"][0]
            
            ve = (ids==151653).nonzero(as_tuple=True)[0].item()
            ie = (ids==151645).nonzero(as_tuple=True)[0].item()
            tok_strs = proc.tokenizer.convert_ids_to_tokens(ids[ve+1:ie].tolist())
            
            nvt = ve - 4
            spec_global = [ve + 1 + p for p in find_keyword_pos(tok_strs, keyword)]
            
            # Register hooks
            attn_maps = []
            def make_hook():
                def hook(m, i, o):
                    if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
                        attn_maps.append(o[1].detach())
                return hook
            
            hooks = [lm.layers[l].self_attn.register_forward_hook(make_hook()) for l in layers]
            with torch.no_grad():
                model(**inp)
            for h in hooks: h.remove()
            
            if attn_maps:
                a = torch.stack(attn_maps).mean(dim=0).mean(dim=[0,1])
                
                # spec
                if spec_global:
                    spec = a[spec_global, 4:ve].mean(dim=0).cpu().float().numpy()
                else:
                    spec = np.zeros(nvt)
                    print(f"    WARNING: keyword '{keyword}' not found!")
                
                # last
                last = a[ie-1, 4:ve].cpu().float().numpy()
                
                # avg
                avg = a[ve+1:ie, 4:ve].mean(dim=0).cpu().float().numpy()
                
                grid_sz = int(math.sqrt(nvt))
                results["spec"].setdefault(fname, {})[q_text] = spec.reshape(grid_sz, grid_sz)
                results["avg"].setdefault(fname, {})[q_text] = avg.reshape(grid_sz, grid_sz)
                results["last"].setdefault(fname, {})[q_text] = last.reshape(grid_sz, grid_sz)
            
            del attn_maps, inp; torch.cuda.empty_cache()
    
    # Cleanup
    for l in layers: lm.layers[l].self_attn.output_attentions = False
    return results

def compute_corrs(group_results):
    """Compute within-face and face-ctrl correlations."""
    face_keys = ["nose","eyes","mouth","chin","forehead","beard"]
    ctrl_keys = ["whole image", "bottom-left area"]
    
    # Flatten: collect all face-key attention maps across all images
    spec_face = []
    spec_ctrl = []
    for method in ["spec", "avg"]:
        within = []
        fc = []
        for fname in group_results[method]:
            maps = group_results[method][fname]
            f_maps = [maps[k].flatten() for k in face_keys if k in maps]
            c_maps = [maps[k].flatten() for k in ctrl_keys if k in maps]
            
            if len(f_maps) < 2: continue
            
            # Within-face pairs
            for i in range(len(f_maps)):
                for j in range(i+1, len(f_maps)):
                    r, _ = pearsonr(f_maps[i], f_maps[j])
                    within.append(r)
            
            # Face-ctrl pairs
            for f in f_maps:
                for c in c_maps:
                    r, _ = pearsonr(f, c)
                    fc.append(r)
        
        spec_face.append(float(np.mean(within)) if within else 1.0)
        spec_ctrl.append(float(np.mean(fc)) if fc else 1.0)
    
    return {
        "spec_within_face": spec_face[0],
        "spec_face_ctrl": spec_ctrl[0],
        "avg_within_face": spec_face[1],
        "avg_face_ctrl": spec_ctrl[1],
    }

# ---- Run for each layer group ----
all_results = {}

for gk, layers in LAYER_GROUPS.items():
    print(f"\n{'='*60}")
    print(f"Layer group: {gk} {layers}")
    print(f"{'='*60}")
    
    group_results = run_group(layers, images)
    corrs = compute_corrs(group_results)
    
    print(f"  spec_within_face={corrs['spec_within_face']:.4f}")
    print(f"  spec_face_ctrl={corrs['spec_face_ctrl']:.4f}")
    print(f"  avg_within_face={corrs['avg_within_face']:.4f}")
    print(f"  avg_face_ctrl={corrs['avg_face_ctrl']:.4f}")
    
    # Extract peaks
    peaks = {}
    for fname in group_results["spec"]:
        for q_text in group_results["spec"][fname]:
            peak = np.unravel_index(group_results["spec"][fname][q_text].argmax(),
                                    group_results["spec"][fname][q_text].shape)
            peaks[f"{fname}/{q_text}"] = [int(peak[0]), int(peak[1])]
    
    all_results[gk] = {
        "corrs": corrs,
        "peaks": peaks,
    }

# ---- Summary ----
print(f"\n{'='*70}")
print(f"{'CROSS-LAYER FACE ATTENTION SUMMARY':^70s}")
print(f"{'='*70}")
print(f"{'Group':<10s} {'Spec Within':>12s} {'Spec Ctrl':>12s} {'Avg Within':>12s} {'Avg Ctrl':>12s}")
print("  " + "-" * 58)
for gk in LAYER_GROUPS:
    if gk not in all_results: continue
    c = all_results[gk]["corrs"]
    print(f"  {gk:<10s} {c['spec_within_face']:>12.4f} {c['spec_face_ctrl']:>12.4f} {c['avg_within_face']:>12.4f} {c['avg_face_ctrl']:>12.4f}")

# ---- Save ----
with open(f"{OUT}/cross_layer_face_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {OUT}/cross_layer_face_results.json")
print("DONE!")
