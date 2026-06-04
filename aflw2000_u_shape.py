"""
AFLW2000 U-Shape Analysis: Validate attention U-shape on 200 real face images.

Loads Qwen3-VL-2B (bf16), runs 4 layer groups × 6 queries × N images,
computes per-image within-face r for spec and avg methods.
Output: JSON with per-image + summary statistics.
"""
import os, sys, gc, torch, json, math, time
import numpy as np
from PIL import Image
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ============ CONFIG ============
BASE = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
IMG_DIR = os.path.join(BASE, "AFLW2000-3D_crop")
LIST_FILE = os.path.join(BASE, "AFLW2000-3D_crop.list")
OUT_DIR = "/root/Qwen3-VL/paper/figures/fixed_test"
os.makedirs(OUT_DIR, exist_ok=True)

N_IMAGES = 200
QUERIES = ["nose", "eyes", "mouth", "chin", "forehead", "beard"]
LAYER_GROUPS = {
    "L1-3":    [0, 1, 2],
    "L8-10":   [7, 8, 9],
    "L18-20":  [17, 18, 19],
    "L25-27":  [25, 26, 27],
}
GROUP_NAMES = list(LAYER_GROUPS.keys())


def compute_within_r(flat_maps):
    """flat_maps: list of 1D arrays. Returns mean upper-triangle Pearson r."""
    nq = len(flat_maps)
    sim = np.zeros((nq, nq))
    for i in range(nq):
        for j in range(nq):
            r, _ = pearsonr(flat_maps[i], flat_maps[j])
            sim[i, j] = r
    face_sims = [sim[i, j] for i in range(nq) for j in range(i + 1, nq)]
    return float(np.mean(face_sims))


def main():
    print("=" * 70)
    print("AFLW2000 U-SHAPE ANALYSIS")
    print("=" * 70)

    # ---- 1. Load Qwen ----
    print("\n[1/4] Loading Qwen3-VL-2B (bfloat16)...")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager",
        local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
    lm = model.model.language_model
    print(f"   Loaded. {len(lm.layers)} layers.")

    # ---- 2. Load image list ----
    print("\n[2/4] Loading AFLW2000 image list...")
    with open(LIST_FILE) as f:
        all_names = [l.strip() for l in f if l.strip()]
    step = max(1, len(all_names) // N_IMAGES)
    sample_indices = list(range(0, len(all_names), step))[:N_IMAGES]
    img_names = [all_names[i] for i in sample_indices]
    print(f"   {len(all_names)} total, sampling {N_IMAGES} at step={step}.")

    # ---- 3. Extract ----
    print("\n[3/4] Extracting attention across layers and queries...")
    results = {gk: {} for gk in GROUP_NAMES}
    t_start = time.time()

    for img_idx, img_name in enumerate(img_names):
        elapsed = time.time() - t_start
        eta = (elapsed / max(1, img_idx)) * (N_IMAGES - img_idx) if img_idx > 0 else 0
        print(f"   [{img_idx + 1}/{N_IMAGES}] {img_name[:30]:30s}  "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"      SKIP (not found)")
            continue
        img = Image.open(img_path).convert("RGB")

        for gk in GROUP_NAMES:
            layers = LAYER_GROUPS[gk]
            spec_maps, avg_maps = [], []

            for query in QUERIES:
                msg = [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": f"describe the {query}"}
                ]}]
                prompt = processor.apply_chat_template(msg, tokenize=False,
                                                       add_generation_prompt=True)
                inp = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
                ids = inp["input_ids"][0]
                ve = (ids == 151653).nonzero(as_tuple=True)[0].item()
                ie = (ids == 151645).nonzero(as_tuple=True)[0].item()
                vs, nvt = 4, ve - 4
                nd = int(math.isqrt(nvt))
                qs, qe = ve + 1, ie

                all_toks = [processor.decode([t]).strip() for t in ids.tolist()]
                kw = [qs + i for i, t in enumerate(all_toks[qs:qe])
                      if query.lower() in t.lower()]

                for l in layers:
                    lm.layers[l].self_attn.output_attentions = True

                aw = []
                def make_hook(w):
                    def hk(m, i, o):
                        if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
                            w.append(o[1].detach())
                    return hk
                handles = [lm.layers[l].self_attn.register_forward_hook(make_hook(aw))
                           for l in layers]

                with torch.no_grad():
                    out = model(**inp)

                for h in handles:
                    h.remove()

                if aw:
                    a = torch.stack(aw).mean(dim=0).mean(dim=[0, 1])
                    if kw:
                        spec = a[kw, vs:ve].mean(dim=0).cpu().float().numpy()
                    else:
                        spec = np.zeros(nvt)
                    avg = a[qs:qe, vs:ve].mean(dim=0).cpu().float().numpy()
                else:
                    spec = np.zeros(nvt)
                    avg = np.zeros(nvt)

                spec_maps.append(spec.reshape(nd, nd))
                avg_maps.append(avg.reshape(nd, nd))

                del aw, a, out, inp
                torch.cuda.empty_cache()

            fs = [m.flatten() for m in spec_maps]
            fa = [m.flatten() for m in avg_maps]
            results[gk][img_name] = {
                "spec_within": compute_within_r(fs),
                "avg_within": compute_within_r(fa),
            }

        gc.collect()
        torch.cuda.empty_cache()

    # ---- 4. Summary ----
    print("\n[4/4] Computing summary statistics...")
    print("\n" + "=" * 70)
    print("AFLW2000 U-SHAPE SUMMARY (mean +/- std across N images)")
    print("=" * 70)
    print(f"{'Layer':12s} {'Spec Mean':>10s} {'Spec Std':>9s} "
          f"{'Avg Mean':>10s} {'Avg Std':>9s}")
    print("-" * 50)
    summary = {}
    for gk in GROUP_NAMES:
        specs = [results[gk][f]["spec_within"] for f in img_names if f in results[gk]]
        avgs = [results[gk][f]["avg_within"] for f in img_names if f in results[gk]]
        sm = float(np.mean(specs))
        ss = float(np.std(specs))
        am = float(np.mean(avgs))
        as_ = float(np.std(avgs))
        n_valid = len(specs)
        summary[gk] = {
            "spec_mean": round(sm, 4),
            "spec_std": round(ss, 4),
            "avg_mean": round(am, 4),
            "avg_std": round(as_, 4),
            "n_images": n_valid,
        }
        print(f"{gk:12s} {sm:>10.4f} {ss:>9.4f}   {am:>10.4f} {as_:>9.4f}")

    save_data = {
        "config": {
            "n_images": N_IMAGES,
            "n_sampled": len(img_names),
            "queries": QUERIES,
            "layer_groups": GROUP_NAMES,
            "models": ["Qwen3-VL-2B"],
            "model_precision": "bfloat16",
            "dataset": "AFLW2000-3D_crop",
            "image_size": "120x120",
        },
        "summary": summary,
        "per_image": results,
    }
    save_path = os.path.join(OUT_DIR, "aflw2000_u_shape.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")

    total = time.time() - t_start
    print(f"\nTotal time: {total:.0f}s ({total / 60:.1f}min)")

    del model, lm, processor
    gc.collect()
    torch.cuda.empty_cache()
    print("Done!")


if __name__ == "__main__":
    main()
