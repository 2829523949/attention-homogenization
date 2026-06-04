"""
Cross-layer Dot Test: run dot attention at 4 layer groups.
Uses original proven methodology: image tokens at [4:ve], text tokens at [ve+1:ie].
Usage: python run_cross_layer_dot.py
"""
import os, sys, math, torch, numpy as np
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import json
sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"; os.environ["HF_HUB_OFFLINE"] = "1"
OUT_DIR = "/root/Qwen3-VL/paper/figures/dot_test"; os.makedirs(OUT_DIR, exist_ok=True)

LAYER_GROUPS = {
    "L1-3":  [0, 1, 2],
    "L8-10": [7, 8, 9],
    "L18-20":[17, 18, 19],
    "L25-27":[25, 26, 27],
}

# 256x256 image with 16x16 color blocks
SIZE = 256; CELL = 16
img = Image.new("RGB", (SIZE, SIZE), (0, 0, 0)); draw = ImageDraw.Draw(img)
blocks = {"red":(1,1), "green":(7,7), "blue":(14,14)}
colors = {"red":(255,0,0), "green":(0,255,0), "blue":(0,0,255)}
for n, (gx, gy) in blocks.items():
    x, y = gx*CELL, gy*CELL; draw.rectangle([x, y, x+CELL-1, y+CELL-1], fill=colors[n])
img.save(f"{OUT_DIR}/synthetic_dots.png")
print(f"Image: {SIZE}x{SIZE}")

# Load Qwen once
from transformers import AutoModelForImageTextToText, AutoProcessor
qwen = AutoModelForImageTextToText.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
    device_map="cuda", attn_implementation="eager",
    local_files_only=True, trust_remote_code=True)
qproc = AutoProcessor.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
qlm = qwen.model.language_model

QUERIES = {"red":"red", "green":"green", "blue":"blue", "image":None}
results = {}  # {layer_group: {query: {spec_peak, avg_peak, last_peak, spec_map, ...}}}

def find_keyword_pos(tok_strs, keyword):
    """Find token position(s) where keyword appears in token list."""
    positions = []
    for j, t in enumerate(tok_strs):
        clean_t = t.replace("\u0120","").replace("\u2581","").strip()
        if keyword.lower() in clean_t.lower():
            positions.append(j)
    return positions

for gk, LAYERS in LAYER_GROUPS.items():
    print(f"\n{'='*60}")
    print(f"Layer group: {gk} {LAYERS}")
    print(f"{'='*60}")
    
    # Enable attentions for this group
    for l in LAYERS: qlm.layers[l].self_attn.output_attentions = True
    
    group_results = {}
    
    for q_text, color_word in QUERIES.items():
        msg = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":f"describe the {q_text}"}]}]
        pr = qproc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
        inp = qproc(text=pr,images=img,return_tensors="pt").to("cuda")
        ids = inp["input_ids"][0]
        
        # Find token boundaries (same as original code)
        ve = (ids==151653).nonzero(as_tuple=True)[0].item()  # <|vision_end|>
        ie = (ids==151645).nonzero(as_tuple=True)[0].item()  # </s>
        tok_strs = qproc.tokenizer.convert_ids_to_tokens(ids[ve+1:ie].tolist())
        
        nvt = ve - 4  # number of visual tokens
        
        # Find keyword positions
        kw = color_word if color_word else q_text
        spec_positions = find_keyword_pos(tok_strs, kw)
        if spec_positions:
            # Convert to global token indices
            spec_global = [ve + 1 + p for p in spec_positions]
        
        # Register hooks (same pattern as original: collect from attention output)
        attn_maps = []
        def make_hook():
            def hook(m, i, o):
                if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
                    attn_maps.append(o[1].detach())
            return hook
        
        hooks = [qlm.layers[l].self_attn.register_forward_hook(make_hook()) for l in LAYERS]
        with torch.no_grad():
            out = qwen(**inp)
        for h in hooks: h.remove()
        
        if attn_maps:
            a = torch.stack(attn_maps).mean(dim=0).mean(dim=[0,1])  # avg over layers+heads
            # a shape: [q_len, k_len]
            
            # spec: from keyword token positions
            if spec_positions:
                spec = a[spec_global, 4:ve].mean(dim=0).cpu().float().numpy()
            else:
                spec = a[ve+1:ie, 4:ve].mean(dim=(0,1)).cpu().float().numpy()
            
            # avg: from all query tokens
            avg = a[ve+1:ie, 4:ve].mean(dim=0).cpu().float().numpy()
            
            # last: from last query token
            last = a[ie-1, 4:ve].cpu().float().numpy()
            
            # Reshape to grid
            grid_sz = int(math.sqrt(nvt))
            spec_map = spec.reshape(grid_sz, grid_sz)
            avg_map = avg.reshape(grid_sz, grid_sz)
            last_map = last.reshape(grid_sz, grid_sz)
            
            spec_peak = [int(x) for x in np.unravel_index(spec_map.argmax(), spec_map.shape)]
            avg_peak = [int(x) for x in np.unravel_index(avg_map.argmax(), avg_map.shape)]
            last_peak = [int(x) for x in np.unravel_index(last_map.argmax(), last_map.shape)]
            
            print(f"  {q_text:>10s}: spec={spec_peak} avg={avg_peak} last={last_peak}, nvt={nvt}")
            
            group_results[q_text] = {
                "spec_peak": spec_peak,
                "avg_peak": avg_peak,
                "last_peak": last_peak,
                "spec_map": spec_map.tolist(),
                "avg_map": avg_map.tolist(),
            }
        
        torch.cuda.empty_cache()
    
    results[gk] = group_results
    
    # Disable attentions for this group
    for l in LAYERS: qlm.layers[l].self_attn.output_attentions = False

# ---- Cross-r calculation ----
# Cross-r between spec maps of different colors: how different are they?
print(f"\n{'='*60}")
print(f"{'CROSS-R (lower = better differentiation)':^60s}")
print(f"{'='*60}")
for gk in LAYER_GROUPS:
    if gk not in results: continue
    r = results[gk]
    if "red" in r and "green" in r and "blue" in r:
        rg = np.corrcoef(np.array(r["red"]["spec_map"]).flatten(), np.array(r["green"]["spec_map"]).flatten())[0,1]
        rb = np.corrcoef(np.array(r["red"]["spec_map"]).flatten(), np.array(r["blue"]["spec_map"]).flatten())[0,1]
        gb = np.corrcoef(np.array(r["green"]["spec_map"]).flatten(), np.array(r["blue"]["spec_map"]).flatten())[0,1]
        print(f"  {gk}: r_red-green={rg:.4f}  r_red-blue={rb:.4f}  r_green-blue={gb:.4f}")
        results[gk]["cross_r"] = {
            "red_green": round(float(rg), 4),
            "red_blue": round(float(rb), 4),
            "green_blue": round(float(gb), 4),
        }

# ---- Summary table ----
print(f"\n{'='*70}")
print(f"{'CROSS-LAYER DOT TEST SUMMARY':^70s}")
print(f"{'='*70}")
print(f"{'Group':<10s} {'Red spec':>10s} {'Green spec':>10s} {'Blue spec':>10s}")
print("  " + "-" * 42)
for gk in LAYER_GROUPS:
    if gk not in results: continue
    r = results[gk]
    red = str(tuple(r["red"]["spec_peak"]))
    grn = str(tuple(r["green"]["spec_peak"]))
    blu = str(tuple(r["blue"]["spec_peak"]))
    print(f"  {gk:<10s} {red:>10s} {grn:>10s} {blu:>10s}")

# ---- Save ----
with open(f"{OUT_DIR}/cross_layer_dot_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT_DIR}/cross_layer_dot_results.json")
print("DONE!")
