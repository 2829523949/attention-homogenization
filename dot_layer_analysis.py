"""
Dot test: per-layer cross-color correlation + peak analysis.
4 layer groups, spec method, 3 colors.
Output: dot_layer_results.json (cross_r + peaks per layer)
"""
import os, sys, gc, torch, json, math, time
import numpy as np
from PIL import Image
from scipy.stats import pearsonr
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

OUT_DIR = "/root/Qwen3-VL/paper/figures/fixed_test"
os.makedirs(OUT_DIR, exist_ok=True)

SIZE = 256
CELL = 16
BLOCK_CENTERS = {"red": (24, 24), "green": (120, 120), "blue": (232, 232)}
COLORS = ["red", "green", "blue"]

LAYER_GROUPS = {
    "L1-3":    [0, 1, 2],
    "L8-10":   [7, 8, 9],
    "L18-20":  [17, 18, 19],
    "L25-27":  [25, 26, 27],
}
GROUP_NAMES = list(LAYER_GROUPS.keys())

GRID_SZ = 16
EXPECTED_PEAKS = {
    "red":   (1, 1),
    "green": (7, 7),
    "blue":  (14, 14),
}


def generate_dot_image():
    img = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    from PIL import ImageDraw; draw = ImageDraw.Draw(img)
    for name, (cx, cy) in BLOCK_CENTERS.items():
        rgb = {"red": (255,0,0), "green": (0,255,0), "blue": (0,0,255)}[name]
        half = CELL // 2
        x0, y0 = max(0, cx-half), max(0, cy-half)
        x1, y1 = min(SIZE-1, cx+half), min(SIZE-1, cy+half)
        draw.rectangle([x0, y0, x1, y1], fill=rgb)
    return img


def compute_within_r(flat_maps):
    nq = len(flat_maps)
    sim = np.zeros((nq, nq))
    for i in range(nq):
        for j in range(nq):
            r, _ = pearsonr(flat_maps[i], flat_maps[j])
            sim[i, j] = r
    vals = [sim[i, j] for i in range(nq) for j in range(i + 1, nq)]
    return float(np.mean(vals)), sim


def main():
    print("="*60)
    print("DOT TEST: PER-LAYER CROSS-R + PEAK ANALYSIS")
    print("="*60)

    img = generate_dot_image()
    print(f"\n[1] Dot image: {SIZE}x{SIZE}")

    print("\n[2] Loading Qwen3-VL-2B...")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager",
        local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
    lm = model.model.language_model

    print("\n[3] Extracting attention per layer group...")
    results = {}

    for gk in GROUP_NAMES:
        layer_ids = LAYER_GROUPS[gk]
        print(f"  {gk}: {layer_ids}")
        maps_by_color = {}
        peaks_by_color = {}

        for color in COLORS:
            query = f"describe the {color} area"
            msg = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":query}]}]
            prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inp = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
            ids = inp["input_ids"][0]
            ve = (ids==151653).nonzero(as_tuple=True)[0].item()
            ie = (ids==151645).nonzero(as_tuple=True)[0].item()
            vs, nvt = 4, ve-4
            nd = int(math.isqrt(nvt))
            qs, qe = ve+1, ie

            all_toks = [processor.decode([t]).strip() for t in ids.tolist()]
            kw_pos = [qs+i for i,t in enumerate(all_toks[qs:qe]) if color in t.lower()]

            for l in layer_ids:
                lm.layers[l].self_attn.output_attentions = True

            aw = []
            def mk(w):
                def hk(m,i,o):
                    if isinstance(o,tuple) and len(o)>1 and o[1] is not None:
                        w.append(o[1].detach())
                return hk
            handles = [lm.layers[l].self_attn.register_forward_hook(mk(aw)) for l in layer_ids]
            with torch.no_grad():
                model(**inp)
            for h in handles:
                h.remove()

            if aw:
                a = torch.stack(aw).mean(dim=0).mean(dim=[0,1])
                if kw_pos:
                    spec = a[kw_pos, vs:ve].mean(dim=0).cpu().float().numpy()
                else:
                    spec = np.ones(nvt)/nvt
            else:
                spec = np.ones(nvt)/nvt

            spec_map = spec.reshape(nd, nd)
            peak = np.unravel_index(spec_map.argmax(), spec_map.shape)
            maps_by_color[color] = spec
            peaks_by_color[color] = (int(peak[1]), int(peak[0]))

            del aw, a, inp
            torch.cuda.empty_cache()

        # Cross-color correlation
        flat_maps = [maps_by_color[c] for c in COLORS]
        cross_r, sim_mat = compute_within_r(flat_maps)
        pair_r = {}
        for i, c1 in enumerate(COLORS):
            for j, c2 in enumerate(COLORS):
                if i < j:
                    pair_r[f"{c1}_{c2}"] = round(float(sim_mat[i, j]), 4)

        # Peak distances
        peak_dist = {}
        for c in COLORS:
            pk = peaks_by_color[c]
            exp = EXPECTED_PEAKS[c]
            dist = math.sqrt((pk[0]-exp[0])**2 + (pk[1]-exp[1])**2)
            peak_dist[c] = {"peak": pk, "expected": exp, "dist": round(dist, 2)}

        results[gk] = {
            "cross_color_r": round(cross_r, 4),
            "pair_r": pair_r,
            "peaks": {c: peaks_by_color[c] for c in COLORS},
            "peak_distances": peak_dist,
        }
        print(f"    cross_r={cross_r:.4f}")
        for c in COLORS:
            pd = peak_dist[c]
            print(f"    {c}: peak={pd['peak']}, target={pd['expected']}, d={pd['dist']}")

        gc.collect()
        torch.cuda.empty_cache()

    print("\n[4] Saving...")
    path = os.path.join(OUT_DIR, "dot_layer_results.json")
    with open(path, "w") as f:
        json.dump({
            "config": {"layer_groups": LAYER_GROUPS, "colors": COLORS, "grid_size": GRID_SZ},
            "results": results,
        }, f, indent=2)
    print(f"   Saved: {path}")
    print("DONE.")


if __name__ == "__main__":
    main()
