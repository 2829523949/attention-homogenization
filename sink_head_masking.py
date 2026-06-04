"""
Sink Head Masking (方案一)
==========================
Goal: In L25-27, mask the most collapsed attention heads,
      then measure whether within-face r decreases (improves).

No forward-pass modification - we compute directly from extracted weights.
"""

import os, sys, gc, torch, json, math, time
import numpy as np
from PIL import Image
from scipy.stats import pearsonr
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

DATA_DIR = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
IMG_DIR = os.path.join(DATA_DIR, "AFLW2000-3D_crop")
LIST_FILE = os.path.join(DATA_DIR, "AFLW2000-3D_crop.list")
OUT_DIR = "/root/Qwen3-VL/paper/figures/fixed_test"
os.makedirs(OUT_DIR, exist_ok=True)

N_IMAGES = 50
LAYER_GROUP = [25, 26, 27]
N_HEADS = 16
QUERIES = ["nose", "eyes", "mouth", "chin", "forehead", "beard"]


def load_images():
    with open(LIST_FILE) as f:
        all_names = [l.strip() for l in f if l.strip()]
    step = max(1, len(all_names) // N_IMAGES)
    return [all_names[i] for i in list(range(0, len(all_names), step))[:N_IMAGES]]


def extract_per_head(model, processor, lm, img, query, layers):
    """Returns [n_heads, nvt] for spec token"""
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
    qs, qe = ve + 1, ie

    all_toks = [processor.decode([t]).strip() for t in ids.tolist()]
    kw_word = query.split()[0]
    kw = [qs + i for i, t in enumerate(all_toks[qs:qe]) if kw_word.lower() in t.lower()]
    q_pos = kw[0] if kw else qs

    for l in layers:
        lm.layers[l].self_attn.output_attentions = True

    per_head = []
    for l in layers:
        aw = []
        def hook(m, i, o):
            if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
                aw.append(o[1].detach())
        handle = lm.layers[l].self_attn.register_forward_hook(hook)
        with torch.no_grad():
            model(**inp)
        handle.remove()
        if aw:
            per_head.append(aw[0][0, :, q_pos, vs:ve].cpu().float())
        else:
            per_head.append(torch.zeros(N_HEADS, nvt))
        del aw; torch.cuda.empty_cache()

    del inp; gc.collect(); torch.cuda.empty_cache()
    return torch.stack(per_head).mean(dim=0)  # [n_heads, nvt]


def head_within_r(ph_dict):
    """ph_dict: {q: [h, nvt]} -> per_head_r [h]"""
    h_r = []
    for h in range(N_HEADS):
        maps = [ph_dict[q][h].numpy() for q in QUERIES]
        sims = []
        for i in range(len(maps)):
            for j in range(i+1, len(maps)):
                r, _ = pearsonr(maps[i], maps[j])
                sims.append(r)
        h_r.append(float(np.mean(sims)))
    return np.array(h_r)


def within_r(maps):
    sims = []
    for i in range(len(maps)):
        for j in range(i+1, len(maps)):
            r, _ = pearsonr(maps[i], maps[j])
            sims.append(r)
    return float(np.mean(sims))


def simulate(ph_dict, keep_heads):
    all_maps = [ph_dict[q].mean(dim=0).numpy() for q in QUERIES]
    all_r = within_r(all_maps)
    if not keep_heads:
        return all_r, 0.0
    m = [ph_dict[q][keep_heads].mean(dim=0).numpy() for q in QUERIES]
    return all_r, within_r(m)


def main():
    print("=" * 60)
    print("SINK HEAD MASKING EXPERIMENT")
    print("=" * 60)

    print("\n[1/5] Loading Qwen3-VL-2B...")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager",
        local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
    lm = model.model.language_model

    print("\n[2/5] Loading images...")
    img_names = load_images()
    print(f"   {len(img_names)} images")

    print("\n[3/5] Extracting per-head attention...")
    results = {}
    t0 = time.time()
    for idx, name in enumerate(img_names):
        et = time.time() - t0
        eta = (et / max(1, idx)) * (N_IMAGES - idx) if idx > 0 else 0
        print(f"   [{idx+1}/{N_IMAGES}] {name[:30]:30s}  elap={et:.0f}s eta={eta:.0f}s")
        p = os.path.join(IMG_DIR, name)
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("RGB")
        res = {}
        for q in QUERIES:
            res[q] = extract_per_head(model, processor, lm, img, q, LAYER_GROUP).numpy().tolist()
        results[name] = res
        gc.collect(); torch.cuda.empty_cache()

    print("\n[4/5] Computing per-head within-face r...")
    all_ph_r = []
    per_img = []
    for name in img_names:
        if name not in results:
            continue
        d = {q: torch.tensor(results[name][q]) for q in QUERIES}
        hr = head_within_r(d)
        all_ph_r.append(hr)
        ar, _ = simulate(d, list(range(N_HEADS)))
        per_img.append({"image": name, "per_head_r": hr.tolist(), "all_heads_r": ar})

    ph_avg = np.mean(all_ph_r, axis=0)
    ph_std = np.std(all_ph_r, axis=0)
    print(f"\n   {'Head':>5s} {'Mean r':>8s} {'Std r':>8s}")
    print(f"   {'-'*23}")
    for h in range(N_HEADS):
        print(f"   {h:>4d}  {ph_avg[h]:>8.4f} {ph_std[h]:>8.4f}")

    best_i = int(ph_avg.argmin())
    worst_i = int(ph_avg.argmax())
    print(f"\n   Mean: {ph_avg.mean():.4f}")
    print(f"   Best: head {best_i} r={ph_avg[best_i]:.4f}")
    print(f"   Worst: head {worst_i} r={ph_avg[worst_i]:.4f}")

    print("\n[5/5] Simulating head masking...")
    si = np.argsort(ph_avg)
    strategies = {
        "mask_top2": sorted(si[-2:].tolist()),
        "mask_top4": sorted(si[-4:].tolist()),
        "mask_top8": sorted(si[-8:].tolist()),
        "keep_best4": sorted(si[:4].tolist()),
        "keep_best1": sorted(si[:1].tolist()),
    }

    baseline_r = np.mean([p["all_heads_r"] for p in per_img])
    mask_res = {}
    print(f"\n   {'Strategy':>20s} {'Keep':>20s} {'Before':>8s} {'After':>8s} {'Delta':>8s}")
    print(f"   {'-'*66}")
    for sn, keep in strategies.items():
        mask = [h for h in range(N_HEADS) if h not in keep]
        diffs = []
        for name in img_names:
            if name not in results:
                continue
            d = {q: torch.tensor(results[name][q]) for q in QUERIES}
            _, mr = simulate(d, [h for h in range(N_HEADS) if h not in mask])
            diffs.append(mr - baseline_r)
        md = np.mean(diffs)
        mask_res[sn] = {
            "keep_heads": keep,
            "baseline": round(baseline_r, 4),
            "masked": round(baseline_r + md, 4),
            "delta": round(md, 4),
        }
        print(f"   {sn:>20s} {str(keep):>20s} {baseline_r:>8.4f} {baseline_r+md:>8.4f} {md:>+8.4f}")

    save = {
        "config": {"n_images": N_IMAGES, "layer_group": LAYER_GROUP, "model": "Qwen3-VL-2B", "queries": QUERIES},
        "per_head_summary": {
            "mean_r": [round(float(r), 4) for r in ph_avg.tolist()],
            "std_r": [round(float(r), 4) for r in ph_std.tolist()],
            "best_head": int(best_i), "best_head_r": round(float(ph_avg[best_i]), 4),
            "worst_head": int(worst_i), "worst_head_r": round(float(ph_avg[worst_i]), 4),
        },
        "masking_results": mask_res,
        "baseline_within_r": round(baseline_r, 4),
        "per_image": per_img,
    }
    sp = os.path.join(OUT_DIR, "sink_head_masking_results.json")
    with open(sp, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n   Saved: {sp}")
    print(f"   Time: {time.time()-t0:.0f}s")
    print("\nDone!")


if __name__ == "__main__":
    main()
