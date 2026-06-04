"""
Attention Weight Injection (方案二 更简洁版)
============================================
Directly replace L28's attention weights with L18's captured ones.
No Q/K/RoPE/norm issues - just replace softmax output then @ V.

Two experiments:
  1. Within-face r: computed directly from captured L18 attn_weights
  2. Dot test cross-r: computed from captured L18 attn_weights
  3. (Optional) PCK: inject into L28, extract hidden states, train probe
"""

import os, sys, gc, torch, json, math, time
import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import pearsonr
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    eager_attention_forward, repeat_kv
)

DATA_DIR = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
IMG_DIR = os.path.join(DATA_DIR, "AFLW2000-3D_crop")
LIST_FILE = os.path.join(DATA_DIR, "AFLW2000-3D_crop.list")
OUT_DIR = "/root/Qwen3-VL/paper/figures/fixed_test"
os.makedirs(OUT_DIR, exist_ok=True)

N_IMAGES = 30
# Use L18 only (single layer, not averaged)
CAPTURE = 17   # L18 (0-indexed)
INJECT = 27    # L28
QUERIES = ["nose", "eyes", "mouth", "chin", "forehead", "beard"]


def load_images():
    with open(LIST_FILE) as f:
        names = [l.strip() for l in f if l.strip()]
    step = max(1, len(names) // N_IMAGES)
    return [names[i] for i in range(0, len(names), step)][:N_IMAGES]


def within_r(maps):
    sims = []
    for i in range(len(maps)):
        for j in range(i+1, len(maps)):
            r, _ = pearsonr(maps[i], maps[j])
            sims.append(r)
    return float(np.mean(sims))


def cross_r(r, g, b):
    return float(np.mean([pearsonr(r, g)[0], pearsonr(r, b)[0], pearsonr(g, b)[0]]))


def prepare_input(model, processor, img, query):
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
    return inp, vs, ve, nvt, q_pos


def capture_attn_weights(model, lm, inp, capture_layer):
    """
    Run one forward pass, capture the FULL attention weights [1, n_heads, seq, seq]
    from capture_layer's eager_attention_forward output (after softmax).
    """
    saved = {}
    orig_fwd = lm.layers[capture_layer].self_attn.forward

    def capturer(hidden_states, position_embeddings, attention_mask=None, **kw):
        attn = lm.layers[capture_layer].self_attn
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        q = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Normal attention computation to get weights
        attn_o, attn_w = eager_attention_forward(
            attn, q, k, v, attention_mask,
            dropout=0.0 if not attn.training else attn.attention_dropout,
            scaling=attn.scaling,
        )
        # Save the full attention weights [1, n_heads, seq, seq]
        saved["weights"] = attn_w.detach().cpu()

        attn_o = attn_o.reshape(*input_shape, -1).contiguous()
        attn_o = attn.o_proj(attn_o)
        return attn_o, attn_w

    lm.layers[capture_layer].self_attn.forward = capturer

    with torch.no_grad():
        model(**inp)

    lm.layers[capture_layer].self_attn.forward = orig_fwd
    gc.collect(); torch.cuda.empty_cache()

    if "weights" in saved:
        return saved["weights"]  # [1, n_heads, seq, seq]
    return None


def extract_spec_attn_from_full(full_weights, vs, ve, q_pos):
    """Extract spec-token->image attention from full weights, averaged over heads."""
    # full_weights: [1, n_heads, seq, seq]
    # Return: [nvt] spec token's attention to image tokens, averaged over heads
    return full_weights[0, :, q_pos, vs:ve].mean(dim=0).cpu().float().numpy()


def run_injected_forward(model, lm, inp, weights_to_inject):
    """
    Forward pass with L28's attention weights replaced.
    Returns: attn_weights from L28 (for verification) and None (no hidden states yet).
    """
    lm.layers[INJECT].self_attn.output_attentions = True
    orig_fwd = lm.layers[INJECT].self_attn.forward
    inj_w = weights_to_inject  # [1, n_heads, seq, seq] captured from L18

    def injector(hidden_states, position_embeddings, attention_mask=None, **kw):
        attn = lm.layers[INJECT].self_attn
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        q = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # REPLACE: skip Q@K^T/softmax, use captured weights
        k = repeat_kv(k, attn.num_key_value_groups)
        v = repeat_kv(v, attn.num_key_value_groups)
        attn_weights = inj_w.to(q.device, q.dtype)  # [1, 16, seq, seq]
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn.o_proj(attn_output)
        return attn_output, attn_weights

    lm.layers[INJECT].self_attn.forward = injector

    aw = []
    def hook(m, i, o):
        if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
            aw.append(o[1].detach())
    handle = lm.layers[INJECT].self_attn.register_forward_hook(hook)

    with torch.no_grad():
        model(**inp)

    handle.remove()
    lm.layers[INJECT].self_attn.forward = orig_fwd

    result = aw[0] if aw else None
    del aw; gc.collect(); torch.cuda.empty_cache()
    return result


# ============ MAIN ============
def main():
    print("=" * 60)
    print("ATTENTION WEIGHT INJECTION")
    print("=" * 60)

    print("\n[1] Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager",
        local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
    lm = model.model.language_model
    print(f"   28 layers, Capture L{CAPTURE+1}, Inject L{INJECT+1}")

    img_names = load_images()
    print(f"   {len(img_names)} images")

    # ========================
    # PART A: FACE within-face r
    # ========================
    print("\n" + "="*60)
    print("PART A: FACE WITHIN-FACE r")
    print("="*60)
    print("   Computing directly from L18 captured attention weights")
    print("   (No injection needed - L18 attn is already differentiated)")

    l18_r_list = []
    l28_r_list = []
    t0 = time.time()

    for idx, name in enumerate(img_names):
        et = time.time() - t0
        eta = (et / max(1, idx)) * (N_IMAGES - idx) if idx > 0 else 0
        print(f"\n[{idx+1}/{N_IMAGES}] {name[:30]:30s}  elap={et:.0f}s eta={eta:.0f}s")

        p = os.path.join(IMG_DIR, name)
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("RGB")

        # Get L28 baseline attn for each query
        l28_maps = []
        for q in QUERIES:
            inp, vs, ve, nvt, qp = prepare_input(model, processor, img, q)
            # Normal forward, capture L28 attn
            lm.layers[INJECT].self_attn.output_attentions = True
            aw = []
            hh = lm.layers[INJECT].self_attn.register_forward_hook(lambda m,i,o: aw.append(o[1].detach()) if len(o)>1 and o[1] is not None else None)
            with torch.no_grad(): model(**inp)
            hh.remove()
            if aw:
                l28_maps.append(aw[0][0, :, qp, vs:ve].mean(dim=0).cpu().float().numpy())
            del inp, aw; torch.cuda.empty_cache()

        if len(l28_maps) != len(QUERIES):
            print(f"   SKIP: L28 baseline")
            continue
        l28_r = within_r(l28_maps)
        l28_r_list.append(l28_r)

        # Capture L18 attn weights for "nose" query (use first query as representative)
        # Actually for within-face r, we need ALL queries' L18 attn
        l18_maps = []
        for q in QUERIES:
            inp, vs, ve, nvt, qp = prepare_input(model, processor, img, q)
            full_w = capture_attn_weights(model, lm, inp, CAPTURE)
            if full_w is not None:
                l18_maps.append(extract_spec_attn_from_full(full_w, vs, ve, qp))
            del inp; gc.collect(); torch.cuda.empty_cache()

        if len(l18_maps) != len(QUERIES):
            print(f"   SKIP: L18 capture")
            l18_r_list.append(None)
            continue
        l18_r = within_r(l18_maps)
        l18_r_list.append(l18_r)

        print(f"   L28 r={l28_r:.4f}   L18 r={l18_r:.4f}")

    # Stats
    l28_arr = np.array(l28_r_list)
    l18_arr = np.array([v for v in l18_r_list if v is not None])
    print(f"\n=== FACE RESULTS ===")
    print(f"Valid: {len(l28_arr)}/{N_IMAGES}")
    print(f"L28 within-face r:      {l28_arr.mean():.4f} +/- {l28_arr.std():.4f}")
    print(f"L18 within-face r:      {l18_arr.mean():.4f} +/- {l18_arr.std():.4f}")

    # ========================
    # PART B: INJECTION VERIFICATION
    # ========================
    print("\n" + "="*60)
    print("PART B: INJECTION VERIFICATION (L28 with L18 weights)")
    print("="*60)
    print("   First 5 images: inject L18 attn into L28, check within-face r")

    inj_r_list = []
    for idx, name in enumerate(img_names[:5]):
        et = time.time() - t0
        print(f"\n[{idx+1}/5] {name[:30]:30s}")

        p = os.path.join(IMG_DIR, name)
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("RGB")

        # Capture full L18 attn weights using "nose" query
        inp, vs, ve, nvt, qp = prepare_input(model, processor, img, "nose")
        l18_full = capture_attn_weights(model, lm, inp, CAPTURE)
        del inp; gc.collect(); torch.cuda.empty_cache()

        if l18_full is None:
            continue

        # Verify: inject L18 weights into L28, check within-face r
        i_maps = []
        for q in QUERIES:
            inp, vs, ve, nvt, qp = prepare_input(model, processor, img, q)
            attn = run_injected_forward(model, lm, inp, l18_full)
            if attn is not None:
                i_maps.append(attn[0, :, qp, vs:ve].mean(dim=0).cpu().float().numpy())
            del inp; gc.collect(); torch.cuda.empty_cache()

        if len(i_maps) == len(QUERIES):
            inj_r = within_r(i_maps)
            inj_r_list.append(inj_r)
            print(f"   L28 normal r={l28_arr[idx]:.4f}  Injected r={inj_r:.4f}")
        else:
            print(f"   Injection failed")

    # ========================
    # PART C: DOT TEST
    # ========================
    print("\n" + "="*60)
    print("PART C: DOT TEST")
    print("="*60)

    SIZE = 256
    dot_img = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(dot_img)
    centers = {"red": (24, 24), "green": (120, 120), "blue": (232, 232)}
    for name, (cx, cy) in centers.items():
        rgb = {"red": (255,0,0), "green": (0,255,0), "blue": (0,0,255)}[name]
        draw.rectangle([cx-8, cy-8, cx+8, cy+8], fill=rgb)

    # L28 baseline
    d28 = {}
    for c in ["red", "green", "blue"]:
        inp, vs, ve, nvt, qp = prepare_input(model, processor, dot_img, f"{c} area")
        lm.layers[INJECT].self_attn.output_attentions = True
        aw = []
        hh = lm.layers[INJECT].self_attn.register_forward_hook(lambda m,i,o: aw.append(o[1].detach()) if len(o)>1 and o[1] is not None else None)
        with torch.no_grad(): model(**inp)
        hh.remove()
        if aw:
            d28[c] = aw[0][0, :, qp, vs:ve].mean(dim=0).cpu().float().numpy()
        del inp, aw; torch.cuda.empty_cache()

    if all(c in d28 for c in ["red","green","blue"]):
        print(f"L28 dot cross-r: {cross_r(d28['red'], d28['green'], d28['blue']):.4f}")

    # L18 capture (use "red area")
    inp, vs, ve, nvt, qp = prepare_input(model, processor, dot_img, "red area")
    d18_full = capture_attn_weights(model, lm, inp, CAPTURE)
    del inp; gc.collect(); torch.cuda.empty_cache()

    if d18_full is not None:
        # Also get L18 spec attn for each color (from captured weights)
        d18 = {}
        for c, q_str in [("red", "red area"), ("green", "green area"), ("blue", "blue area")]:
            inp, vs, ve, nvt, qp = prepare_input(model, processor, dot_img, q_str)
            fw = capture_attn_weights(model, lm, inp, CAPTURE)
            if fw is not None:
                d18[c] = extract_spec_attn_from_full(fw, vs, ve, qp)
            del inp; gc.collect(); torch.cuda.empty_cache()

        if all(c in d18 for c in ["red","green","blue"]):
            l18_dot_r = cross_r(d18["red"], d18["green"], d18["blue"])
            print(f"L18 dot cross-r:       {l18_dot_r:.4f}")

        # Inject L18 weights into L28 for dot
        di = {}
        for c in ["red", "green", "blue"]:
            inp, vs, ve, nvt, qp = prepare_input(model, processor, dot_img, f"{c} area")
            attn = run_injected_forward(model, lm, inp, d18_full)
            if attn is not None:
                di[c] = attn[0, :, qp, vs:ve].mean(dim=0).cpu().float().numpy()
            del inp; gc.collect(); torch.cuda.empty_cache()

        if all(c in di for c in ["red","green","blue"]):
            dot_inj_r = cross_r(di["red"], di["green"], di["blue"])
            print(f"Injected L28 dot r:    {dot_inj_r:.4f}")
            if all(c in d28 for c in ["red","green","blue"]):
                print(f"L28 dot baseline:      {cross_r(d28['red'], d28['green'], d28['blue']):.4f}")

    # Save results
    save = {
        "config": {"n_images": N_IMAGES, "capture_layer": CAPTURE, "inject_layer": INJECT},
        "face_within_r": {
            "L28_mean": round(float(l28_arr.mean()), 4),
            "L28_std": round(float(l28_arr.std()), 4),
            "L18_mean": round(float(l18_arr.mean()), 4) if len(l18_arr) else None,
            "L18_std": round(float(l18_arr.std()), 4) if len(l18_arr) else None,
        },
        "injection_verification": {
            "L28_baseline_r": round(float(l28_arr.mean()), 4),
            "injected_mean": round(float(np.mean(inj_r_list)), 4) if inj_r_list else None,
        },
    }
    with open(os.path.join(OUT_DIR, "attn_injection_results.json"), "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved. Time: {time.time()-t0:.0f}s")
    print("Done!")


if __name__ == "__main__":
    main()
