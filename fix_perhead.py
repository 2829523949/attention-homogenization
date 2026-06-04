"""
Per-Head Attention Analysis (§4.1)
===================================
Analyzes each of Qwen's 16 heads at L25-27 for attention collapse.
Computes per-head spec within-face r. Shows best/worst head stats.

Output: headwise_attention_results.json
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

# EG3D face images
EG3D_DIR = "/root/Qwen3-VL/data/eg3d_faces"
if not os.path.exists(EG3D_DIR):
    EG3D_DIR = "/root/Qwen3-VL/eg3d_faces"
if not os.path.exists(EG3D_DIR):
    EG3D_DIR = os.path.expanduser("~/data/eg3d_faces")

QUERIES = ["nose", "eyes", "mouth", "chin", "forehead", "beard"]
LAYERS = [25, 26, 27]  # L25-27
N_HEADS = 16


def compute_within_r(flat_maps):
    nq = len(flat_maps)
    if nq < 2:
        return 0.0
    sim = np.zeros((nq, nq))
    for i in range(nq):
        for j in range(nq):
            r, _ = pearsonr(flat_maps[i], flat_maps[j])
            sim[i, j] = r
    vals = [sim[i, j] for i in range(nq) for j in range(i + 1, nq)]
    return float(np.mean(vals)) if vals else 0.0


def main():
    print("=" * 70)
    print("PER-HEAD ATTENTION ANALYSIS (L25-27)")
    print("=" * 70)

    # 1. Load model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager",
        local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
    lm = model.model.language_model

    # 2. Load EG3D images
    img_dir = EG3D_DIR
    if not os.path.exists(img_dir):
        print(f"   WARNING: EG3D dir not found. Using AFLW2000 instead...")
        img_dir = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data/AFLW2000-3D_crop"
        if not os.path.exists(img_dir):
            print("   No images found. Exiting.")
            return
    img_files = sorted(os.listdir(img_dir))[:9]
    print(f"   {len(img_files)} images.")

    # 3. Extract per-head attention
    head_r = {h: [] for h in range(N_HEADS)}
    t_start = time.time()

    for img_idx, img_name in enumerate(img_files):
        elapsed = time.time() - t_start
        img_path = os.path.join(img_dir, img_name)
        print(f"   [{img_idx+1}/{len(img_files)}] {img_name:30s} {elapsed:.0f}s")
        img = Image.open(img_path).convert("RGB")

        for query in QUERIES:
            msg = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"describe the {query}"}
            ]}]
            prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inp = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
            ids = inp["input_ids"][0]
            ve = (ids == 151653).nonzero(as_tuple=True)[0].item()
            vs, nvt = 4, ve - 4
            nd = int(math.isqrt(nvt))

            all_toks = [processor.decode([t]).strip() for t in ids.tolist()]
            kw = [ve + 1 + i for i, t in enumerate(all_toks[ve+1:])
                  if query.lower() in t.lower()]

            for l in LAYERS:
                lm.layers[l].self_attn.output_attentions = True

            aw = []
            def make_hook(w):
                def hk(m, i, o):
                    if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
                        w.append(o[1].detach())
                return hk
            handles = [lm.layers[l].self_attn.register_forward_hook(make_hook(aw)) for l in LAYERS]
            with torch.no_grad():
                out = model(**inp)
            for h in handles: h.remove()


            if aw:
                # Stack: [3 layers, batch, n_heads, seq, seq] -> mean over layers
                attn = torch.stack(aw).mean(dim=0)  # [1, n_heads, seq, seq]
                for h in range(N_HEADS):
                    head_map = attn[0, h]  # [seq, seq]
                    if kw:
                        spec = head_map[kw, vs:ve].mean(dim=0).cpu().float().numpy()
                    else:
                        spec = np.zeros(nvt)
                    head_r[h].append(spec.flatten())

            del aw, attn, out, inp; torch.cuda.empty_cache()
            gc.collect()
    # end image

    # ---- Diagnostic: check how much data each head collected ----
    # 4. Compute per-head statistics
    print(f"\n{'Head':>6s} {'Mean r':>8s} {'Min':>8s}")
    print("-" * 24)
    per_head = {}
    for h in range(N_HEADS):
        if len(head_r[h]) >= len(QUERIES):
            # Group by query (every len(QUERIES) entries form a set)
            n_img = len(head_r[h]) // len(QUERIES)
            within_rs = []
            for i in range(n_img):
                maps = head_r[h][i * len(QUERIES):(i + 1) * len(QUERIES)]
                within_rs.append(compute_within_r(maps))
            m = float(np.mean(within_rs))
            per_head[f"head_{h}"] = {"mean_within_r": round(m, 4)}
            print(f"{h:6d} {m:8.4f}")

    if not per_head:
        print("   ERROR: per_head is empty — no heads had enough data.")
        return
    all_means = [v["mean_within_r"] for v in per_head.values()]
    stats = {
        "n_heads": N_HEADS,
        "n_layers": LAYERS,
        "mean_r": round(float(np.mean(all_means)), 4),
        "std_r": round(float(np.std(all_means)), 4),
        "best_head": int(np.argmin(all_means)),
        "best_r": round(float(np.min(all_means)), 4),
        "worst_head": int(np.argmax(all_means)),
        "worst_r": round(float(np.max(all_means)), 4),
        "per_head": per_head,
    }
    print(f"\n   Mean: {stats['mean_r']:.4f} +/- {stats['std_r']:.4f}")
    print(f"   Best head #{stats['best_head']}: r={stats['best_r']:.4f}")
    print(f"   Worst head #{stats['worst_head']}: r={stats['worst_r']:.4f}")

    path = os.path.join(OUT_DIR, "headwise_attention_results.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved to {path}")
    print("DONE.")


if __name__ == "__main__":
    main()
