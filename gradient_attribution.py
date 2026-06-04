"""
Gradient Attribution U-Shape (§4.1)
====================================
Computes ∂||h_q||/∂h_v for query tokens w.r.t. visual patch embeddings
at L1-3, L18-20, L28. Reports gradient within-face r for left-right
and top-bottom query pairs.

Output: gradient_attribution_results.json
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

N_IMAGES = 15
QUERY_PAIRS = [("left side", "right side"), ("top", "bottom")]
LAYER_GROUPS = {"L1-3": [0, 1, 2], "L18-20": [17, 18, 19], "L28": [27]}
GRAD_LAYERS = {"L1-3": [0, 1, 2], "L18-20": [17, 18, 19], "L28": [26]}  # grad from layer-1 for Last group

COCO_DIR = "/datasets/coco/val2017"




def main():
    print("=" * 70)
    print("GRADIENT ATTRIBUTION U-SHAPE")
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

    # 2. Load COCO images
    if not os.path.exists(COCO_DIR):
        print(f"   ERROR: COCO not found")
        return
    all_images = sorted(os.listdir(COCO_DIR))[:N_IMAGES]
    print(f"   {len(all_images)} images.")

    # 3. Compute gradient attribution
    results = {gk: {qp[0]: [] for qp in QUERY_PAIRS} for gk in LAYER_GROUPS}
    t_start = time.time()

    for img_idx, img_name in enumerate(all_images):
        elapsed = time.time() - t_start
        print(f"   [{img_idx+1}/{N_IMAGES}] {img_name:20s} {elapsed:.0f}s")
        img_path = os.path.join(COCO_DIR, img_name)
        img = Image.open(img_path).convert("RGB")

        for gk in LAYER_GROUPS:
            grad_layers = GRAD_LAYERS[gk]
            for q1, q2 in QUERY_PAIRS:
                grad_maps = []
                for q in [q1, q2]:
                    msg = [{"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": f"describe the {q}"}
                    ]}]
                    prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    inp = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
                    inp["pixel_values"].requires_grad_(True)
                    ids = inp["input_ids"][0]
                    ve = (ids == 151653).nonzero(as_tuple=True)[0].item()
                    vs, nvt = 4, ve - 4
                    q_pos = ve + 1  # first text token after vision_end (content token, not format)

                    # Hook grad layer to capture hidden state
                    hs_out = []
                    def hs_hook(m, i, o):
                        hs_out.append(o[0])
                    hs_handle = lm.layers[grad_layers[0]].register_forward_hook(hs_hook)

                    # Forward
                    out = model(**inp)
                    hs_handle.remove()

                    if hs_out:
                        h_q = hs_out[0][0, q_pos]  # [hidden_dim]
                        grad = torch.autograd.grad(h_q.norm(), inp["pixel_values"],
                                                    retain_graph=False)[0]
                        # grad: [1, num_patches, patch_dim] -> norm over patch_dim
                        if grad is not None:
                            sal = grad.norm(dim=-1).detach().cpu().float().numpy()
                            # Take first nvt entries (visual tokens only)
                            sal = sal[:nvt] if len(sal) >= nvt else np.pad(sal, (0, nvt - len(sal)))
                        else:
                            sal = np.zeros(nvt)
                    else:
                        sal = np.zeros(nvt)
                    grad_maps.append(sal)
                    del hs_out, out, inp, grad, h_q, sal
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()
                # end q1,q2

                if len(grad_maps) == 2:
                    r_val, _ = pearsonr(grad_maps[0], grad_maps[1])
                    if not math.isnan(r_val):
                        results[gk][q1].append(r_val)
            # end q-pair
        # end gk
        torch.cuda.empty_cache()
        gc.collect()
    # end image

    # 4. Summary
    summary = {}
    print(f"\n{'Layer':12s} {'Pair':20s} {'Mean r':>8s} {'Std r':>8s}")
    print("-" * 44)
    for gk in LAYER_GROUPS:
        summary[gk] = {}
        for q1, q2 in QUERY_PAIRS:
            vals = results[gk][q1]
            if vals:
                m = float(np.mean(vals))
                s = float(np.std(vals))
                summary[gk][f"{q1}-{q2}"] = {"mean": round(m, 4), "std": round(s, 4)}
                print(f"{gk:12s} {q1}-{q2:15s} {m:8.4f} {s:8.4f}")

    output = {
        "config": {"n_images": N_IMAGES, "layer_groups": LAYER_GROUPS,
                   "grad_layers": GRAD_LAYERS},
        "results": summary,
    }
    path = os.path.join(OUT_DIR, "gradient_attribution_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {path}")
    print("DONE.")


if __name__ == "__main__":
    main()
