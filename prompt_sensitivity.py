"""
Prompt sensitivity test: compare within-face r across alternative phrasings.
20 AFLW2000 images, L18-20 only, 3 prompt sets.
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

BASE = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
IMG_DIR = os.path.join(BASE, "AFLW2000-3D_crop")
LIST_FILE = os.path.join(BASE, "AFLW2000-3D_crop.list")
OUT_DIR = "/root/Qwen3-VL/paper/figures/fixed_test"
os.makedirs(OUT_DIR, exist_ok=True)

N_IMAGES = 20
LAYERS = [17, 18, 19]  # L18-20

QUERY_SETS = {
    "original":     ["nose", "eyes", "mouth", "chin", "forehead", "beard"],
    "verbose":      ["nose area", "eye region", "mouth area", "chin area", "forehead area", "beard area"],
    "synonym":      ["nose tip", "eye area", "lips", "jaw", "brow", "facial hair"],
}

def compute_within_r(flat_maps):
    nq = len(flat_maps)
    sim = np.zeros((nq, nq))
    for i in range(nq):
        for j in range(nq):
            r_val, _ = pearsonr(flat_maps[i], flat_maps[j])
            sim[i, j] = r_val
    face_sims = [sim[i, j] for i in range(nq) for j in range(i + 1, nq)]
    return float(np.mean(face_sims))


def main():
    print("=" * 60)
    print("PROMPT SENSITIVITY TEST")
    print("=" * 60)

    print("\n[1/4] Loading Qwen...")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager",
        local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
    lm = model.model.language_model
    print(f"   Loaded. {len(lm.layers)} layers.")

    print("\n[2/4] Loading images...")
    with open(LIST_FILE) as f:
        all_names = [l.strip() for l in f if l.strip()]
    step = max(1, len(all_names) // N_IMAGES)
    img_names = [all_names[i] for i in list(range(0, len(all_names), step))[:N_IMAGES]]
    print(f"   Sampling {N_IMAGES} images.")

    print("\n[3/4] Running prompts...")
    results = {name: {} for name in QUERY_SETS}
    t_start = time.time()

    for img_idx, img_name in enumerate(img_names):
        elapsed = time.time() - t_start
        eta = (elapsed / max(1, img_idx)) * (N_IMAGES - img_idx) if img_idx > 0 else 0
        print(f"   [{img_idx+1}/{N_IMAGES}] {img_name[:25]:25s}  elap={elapsed:.0f}s eta={eta:.0f}s")

        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")

        for set_name, queries in QUERY_SETS.items():
            spec_maps = []
            for query in queries:
                msg = [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": f"describe the {query}"},
                ]}]
                prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                inp = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
                ids = inp["input_ids"][0]
                ve = (ids == 151653).nonzero(as_tuple=True)[0].item()
                ie = (ids == 151645).nonzero(as_tuple=True)[0].item()
                vs, nvt = 4, ve - 4
                nd = int(math.isqrt(nvt))
                qs, qe = ve + 1, ie

                all_toks = [processor.decode([t]).strip() for t in ids.tolist()]
                kw_word = query.split()[0]  # use first word for multi-word queries
                kw = [qs + i for i, t in enumerate(all_toks[qs:qe]) if kw_word.lower() in t.lower()]

                for l in LAYERS:
                    lm.layers[l].self_attn.output_attentions = True

                aw = []
                def mk(w):
                    def hk(m, i, o):
                        if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
                            w.append(o[1].detach())
                    return hk
                handles = [lm.layers[l].self_attn.register_forward_hook(mk(aw)) for l in LAYERS]
                with torch.no_grad():
                    model(**inp)
                for h in handles:
                    h.remove()

                if aw:
                    a = torch.stack(aw).mean(dim=0).mean(dim=[0, 1])
                    if kw:
                        spec = a[kw, vs:ve].mean(dim=0).cpu().float().numpy()
                    else:
                        spec = np.zeros(nvt)
                else:
                    spec = np.zeros(nvt)

                spec_maps.append(spec.reshape(nd, nd))
                del aw, a, inp
                torch.cuda.empty_cache()

            fs = [m.flatten() for m in spec_maps]
            results[set_name][img_name] = compute_within_r(fs)

        gc.collect()
        torch.cuda.empty_cache()

    print("\n[4/4] Summary...")
    print(f"\n{'Set':12s} {'Mean r':>8s} {'Std':>8s} {'vs Original':>12s}")
    print("-" * 42)
    ref = [results["original"][f] for f in img_names if f in results["original"]]
    ref_mean = np.mean(ref)
    for set_name in QUERY_SETS:
        vals = [results[set_name][f] for f in img_names if f in results[set_name]]
        diff = np.mean(vals) - ref_mean
        print(f"{set_name:12s} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} {diff:>+11.4f}")

    save_data = {
        "config": {
            "n_images": N_IMAGES,
            "layers": LAYERS,
            "query_sets": {k: v for k, v in QUERY_SETS.items()},
            "model": "Qwen3-VL-2B",
        },
        "per_set": results,
        "summary": {},
    }
    for set_name in QUERY_SETS:
        vals = [results[set_name][f] for f in img_names if f in results[set_name]]
        save_data["summary"][set_name] = {
            "mean_r": round(float(np.mean(vals)), 4),
            "std_r": round(float(np.std(vals)), 4),
        }

    save_path = os.path.join(OUT_DIR, "prompt_sensitivity_results.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")
    total = time.time() - t_start
    print(f"Time: {total:.0f}s")
    print("Done!")


if __name__ == "__main__":
    main()
