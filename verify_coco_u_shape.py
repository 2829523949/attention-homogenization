"""
COCO U-Shape Validation (§4.1) - Spec method
===============================================
Runs spec-method attention analysis on COCO val2017 images
across L1-3, L18-20, L28 using spatial keyword queries.
Output: coco_u_shape.json
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

N_IMAGES = 200
QUERY_PAIRS = [("left side", "right side"), ("top", "bottom"), ("person", "animal")]
LAYER_GROUPS = {"L1-3": [0, 1, 2], "L18-20": [17, 18, 19], "L28": [27]}
GROUP_NAMES = list(LAYER_GROUPS.keys())
COCO_DIR = "/datasets/coco/val2017"


def compute_within_r(flat_maps):
    nq = len(flat_maps)
    sim = np.zeros((nq, nq))
    for i in range(nq):
        for j in range(nq):
            r, _ = pearsonr(flat_maps[i], flat_maps[j])
            sim[i, j] = 0.0 if math.isnan(r) else r
    vals = [sim[i, j] for i in range(nq) for j in range(i + 1, nq)]
    return float(np.mean(vals)) if vals else 0.0


def main():
    print("=" * 70)
    print("COCO U-SHAPE VALIDATION (SPEC METHOD)")
    print("=" * 70)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager",
        local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
    lm = model.model.language_model
    print(f"   {len(lm.layers)} layers.")

    if not os.path.exists(COCO_DIR):
        print(f"   ERROR: COCO not found"); return
    all_images = sorted(os.listdir(COCO_DIR))[:N_IMAGES]
    print(f"   {N_IMAGES} images.")

    results = {gk: {qp[0]: [] for qp in QUERY_PAIRS} for gk in GROUP_NAMES}
    t_start = time.time()

    for idx, img_name in enumerate(all_images):
        elapsed = time.time() - t_start
        img = Image.open(os.path.join(COCO_DIR, img_name)).convert("RGB")

        for gk in GROUP_NAMES:
            layers = LAYER_GROUPS[gk]

            for q1, q2 in QUERY_PAIRS:
                maps = []
                for q in [q1, q2]:
                    msg = [{"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": f"describe the {q}"}
                    ]}]
                    prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    inp = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
                    ids = inp["input_ids"][0]
                    ve = (ids == 151653).nonzero(as_tuple=True)[0].item()
                    ie = (ids == 151645).nonzero(as_tuple=True)[0].item()
                    vs, nvt = 4, ve - 4
                    qs, qe = ve + 1, ie

                    # Find keyword token
                    all_toks = [processor.decode([t]).strip() for t in ids.tolist()]
                    kw = [qs + i for i, t in enumerate(all_toks[qs:qe]) if q.lower() in t.lower()]

                    for l in layers:
                        lm.layers[l].self_attn.output_attentions = True

                    aw = []
                    def make_hook(w):
                        def hk(m, i, o):
                            if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
                                w.append(o[1].detach())
                        return hk
                    handles = [lm.layers[l].self_attn.register_forward_hook(make_hook(aw)) for l in layers]
                    with torch.no_grad():
                        model(**inp)
                    for h in handles: h.remove()

                    if aw and kw:
                        a = torch.stack(aw).mean(dim=0).mean(dim=[0, 1])
                        spec = a[kw, vs:ve].mean(dim=0).cpu().float().numpy()
                    else:
                        spec = np.zeros(nvt)
                    maps.append(spec)
                    del aw, inp; torch.cuda.empty_cache()
                    gc.collect()
                # end q-loop

                # pearsonr between the two query maps
                if len(maps) == 2 and maps[0].size > 0 and maps[1].size > 0:
                    r_val, _ = pearsonr(maps[0], maps[1])
                    if not math.isnan(r_val):
                        results[gk][q1].append(r_val)
            # end q-pair
        # end gk
        print(f"   [{idx+1}/{N_IMAGES}] {img_name:20s} {time.time()-t_start:.0f}s")
    # end image

    # Summary
    summary = {}
    print(f"\n{'Layer':12s} {'Pair':20s} {'Mean r':>8s} {'Std r':>8s}")
    print("-" * 48)
    for gk in GROUP_NAMES:
        summary[gk] = {}
        for q1, q2 in QUERY_PAIRS:
            vals = results[gk][q1]
            if vals:
                m = float(np.mean(vals))
                s = float(np.std(vals))
                summary[gk][f"{q1}-{q2}"] = {"mean": round(m, 4), "std": round(s, 4)}
                print(f"{gk:12s} {q1}-{q2:15s} {m:8.4f} {s:8.4f}")

    output = {"config": {"n_images": N_IMAGES, "queries": list(set(sum(qp for qp in QUERY_PAIRS))) if False else
                          ["left side", "right side", "top", "bottom", "person", "animal"],
                   "layer_groups": LAYER_GROUPS, "dataset": "COCO val2017"},
             "results": summary}
    path = os.path.join(OUT_DIR, "coco_u_shape.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
