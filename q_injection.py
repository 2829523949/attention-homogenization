"""
Q Injection v4: FIXED keyword argument signatures.
Diagnostic confirmed: Q18 != Q28 (cos_sim=-0.037), injection works (delta=-0.0296).
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
    eager_attention_forward, apply_rotary_pos_emb, repeat_kv
)

DATA_DIR = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
IMG_DIR = os.path.join(DATA_DIR, "AFLW2000-3D_crop")
LIST_FILE = os.path.join(DATA_DIR, "AFLW2000-3D_crop.list")
OUT_DIR = "/root/Qwen3-VL/paper/figures/fixed_test"
os.makedirs(OUT_DIR, exist_ok=True)

N_IMAGES = 30
CAPTURE = [17, 18, 19]
INJECT = 27
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


def run_and_get_attn(model, lm, inp, vs, ve, q_pos, inject_q=None):
    """
    Run forward, extract L28 attn_weights at (q_pos -> vs:ve).
    If inject_q is not None, monkey-patch L28 forward.
    """
    lm.layers[INJECT].self_attn.output_attentions = True

    # Store original for restoration
    orig_forward = lm.layers[INJECT].self_attn.forward

    if inject_q is not None:
        iq = inject_q  # capture in closure

        def injected_forward(hidden_states, position_embeddings, attention_mask=None, **kw):
            attn = lm.layers[INJECT].self_attn
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attn.head_dim)

            q = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            v = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            nvt = iq.shape[2]

            q = q.clone()
            q[:, :, :nvt, :] = iq.to(q.device, q.dtype)


            attn_o, attn_w = eager_attention_forward(
                attn, q, k, v, attention_mask,
                dropout=0.0 if not attn.training else attn.attention_dropout,
                scaling=attn.scaling,
            )
            attn_o = attn_o.reshape(*input_shape, -1).contiguous()
            attn_o = attn.o_proj(attn_o)
            return attn_o, attn_w

        lm.layers[INJECT].self_attn.forward = injected_forward

    # Hook to capture attention weights
    aw = []
    def hook(m, i, o):
        if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
            aw.append(o[1].detach())
    handle = lm.layers[INJECT].self_attn.register_forward_hook(hook)

    with torch.no_grad():
        model(**inp)

    handle.remove()
    lm.layers[INJECT].self_attn.forward = orig_forward

    attn_map = None
    if aw:
        attn_map = aw[0][0, :, q_pos, vs:ve].mean(dim=0).cpu().float().numpy()

    del aw; gc.collect(); torch.cuda.empty_cache()
    return attn_map


def capture_q(model, lm, inp, nvt, layers):
    """Captures Q from specified layers (after q_norm + RoPE). Returns averaged Q [1,16,nvt,128]."""
    saved = {}
    originals = {}

    for l in layers:
        originals[l] = lm.layers[l].self_attn.forward

        def make_capturer(layer):
            def capturer(hidden_states, position_embeddings, attention_mask=None, **kw):
                attn = lm.layers[layer].self_attn
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn.head_dim)

                q = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                k = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                v = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                saved[layer] = q[:, :, :nvt, :].detach().cpu()


                attn_o, attn_w = eager_attention_forward(
                    attn, q, k, v, attention_mask,
                    dropout=0.0 if not attn.training else attn.attention_dropout,
                    scaling=attn.scaling,
                )
                attn_o = attn_o.reshape(*input_shape, -1).contiguous()
                attn_o = attn.o_proj(attn_o)
                return attn_o, attn_w
            return capturer

        lm.layers[l].self_attn.forward = make_capturer(l)

    with torch.no_grad():
        model(**inp)

    for l, orig in originals.items():
        lm.layers[l].self_attn.forward = orig

    gc.collect(); torch.cuda.empty_cache()
    return saved


def main():
    print("=" * 60)
    print("Q INJECTION v4: FIXED KWARGS SIGNATURE")
    print("=" * 60)

    print("\n[1] Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager",
        local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
    lm = model.model.language_model
    n_heads = lm.layers[0].self_attn.config.num_attention_heads
    print(f"   {n_heads} heads")

    img_names = load_images()
    print(f"   {len(img_names)} images")
    print(f"   Capture L{CAPTURE[0]+1}-{CAPTURE[-1]+1}, Inject L{INJECT+1}")

    # ========================
    # PART A: FACE
    # ========================
    print("\n" + "="*60)
    print("PART A: FACE WITHIN-FACE r")
    print("="*60)

    base_r = []
    inj_r = []
    t0 = time.time()

    for idx, name in enumerate(img_names):
        et = time.time() - t0
        eta = (et / max(1, idx)) * (N_IMAGES - idx) if idx > 0 else 0
        print(f"\n[{idx+1}/{N_IMAGES}] {name[:30]:30s}  elap={et:.0f}s eta={eta:.0f}s")

        p = os.path.join(IMG_DIR, name)
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("RGB")

        # --- Baseline ---
        b_maps = []
        for q in QUERIES:
            inp, vs, ve, nvt, qp = prepare_input(model, processor, img, q)
            attn = run_and_get_attn(model, lm, inp, vs, ve, qp, inject_q=None)
            if attn is not None:
                b_maps.append(attn)
            del inp; gc.collect(); torch.cuda.empty_cache()

        if len(b_maps) < len(QUERIES):
            print(f"   SKIP: baseline")
            continue
        base_r.append(within_r(b_maps))
        print(f"   Baseline r={base_r[-1]:.4f}")

        # --- Capture Q ---
        inp_q, vs_q, ve_q, nvt_q, _ = prepare_input(model, processor, img, "nose")
        saved = capture_q(model, lm, inp_q, nvt_q, CAPTURE)
        del inp_q; gc.collect(); torch.cuda.empty_cache()

        if CAPTURE[0] not in saved:
            print(f"   SKIP: Q capture")
            continue
        inject_q = saved[CAPTURE[0]]
        print(f"   Q={inject_q.shape}")

        # --- Injected ---
        i_maps = []
        for q in QUERIES:
            inp, vs, ve, nvt, qp = prepare_input(model, processor, img, q)
            attn = run_and_get_attn(model, lm, inp, vs, ve, qp, inject_q=inject_q)
            if attn is not None:
                i_maps.append(attn)
            del inp; gc.collect(); torch.cuda.empty_cache()

        if len(i_maps) < len(QUERIES):
            print(f"   SKIP: injected")
            continue
        inj_r.append(within_r(i_maps))
        print(f"   Injected r={inj_r[-1]:.4f}  Delta={inj_r[-1]-base_r[-1]:+.4f}")

    b_arr = np.array(base_r)
    i_arr = np.array(inj_r)
    print(f"\n=== FACE RESULTS ===")
    print(f"Valid: {len(b_arr)}/{N_IMAGES}")
    print(f"Baseline L28:      {b_arr.mean():.4f} +/- {b_arr.std():.4f}")
    print(f"Injected L28+Q:    {i_arr.mean():.4f} +/- {i_arr.std():.4f}")
    print(f"Delta:             {i_arr.mean() - b_arr.mean():+.4f}")

    # ========================
    # PART B: DOT TEST
    # ========================
    print("\n" + "="*60)
    print("PART B: DOT TEST")
    print("="*60)

    SIZE = 256
    dot_img = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(dot_img)
    centers = {"red": (24, 24), "green": (120, 120), "blue": (232, 232)}
    for name, (cx, cy) in centers.items():
        rgb = {"red": (255,0,0), "green": (0,255,0), "blue": (0,0,255)}[name]
        draw.rectangle([cx-8, cy-8, cx+8, cy+8], fill=rgb)

    # Baseline
    db = {}
    for c in ["red", "green", "blue"]:
        inp, vs, ve, nvt, qp = prepare_input(model, processor, dot_img, f"{c} area")
        attn = run_and_get_attn(model, lm, inp, vs, ve, qp, inject_q=None)
        if attn is not None:
            db[c] = attn
        del inp; gc.collect(); torch.cuda.empty_cache()

    if all(c in db for c in ["red","green","blue"]):
        dot_base = cross_r(db["red"], db["green"], db["blue"])
        print(f"Baseline dot: {dot_base:.4f}")
    else:
        dot_base = None; print("Dot baseline FAILED")

    # Capture + Inject for dot
    inp_q, vs_q, ve_q, nvt_d, _ = prepare_input(model, processor, dot_img, "red area")
    saved_d = capture_q(model, lm, inp_q, nvt_d, CAPTURE)
    del inp_q; gc.collect(); torch.cuda.empty_cache()

    dot_inj = None
    if saved_d:
        qd = saved_d[CAPTURE[0]]
        di = {}
        for c in ["red", "green", "blue"]:
            inp, vs, ve, nvt, qp = prepare_input(model, processor, dot_img, f"{c} area")
            attn = run_and_get_attn(model, lm, inp, vs, ve, qp, inject_q=qd)
            if attn is not None:
                di[c] = attn
            del inp; gc.collect(); torch.cuda.empty_cache()
        if all(c in di for c in ["red","green","blue"]):
            dot_inj = cross_r(di["red"], di["green"], di["blue"])
            print(f"Injected dot: {dot_inj:.4f}  Delta: {dot_inj - dot_base:+.4f}" if dot_base is not None else f"Injected dot: {dot_inj:.4f}")

    # Save
    save = {
        "config": {"n_images": N_IMAGES, "capture": CAPTURE, "inject": INJECT},
        "face": {
            "baseline_mean": round(float(b_arr.mean()), 4),
            "baseline_std": round(float(b_arr.std()), 4),
            "injected_mean": round(float(i_arr.mean()), 4),
            "injected_std": round(float(i_arr.std()), 4),
            "delta": round(float(i_arr.mean() - b_arr.mean()), 4),
        },
        "dot": {
            "baseline": round(float(dot_base), 4) if dot_base is not None else None,
            "injected": round(float(dot_inj), 4) if dot_inj is not None else None,
        },
    }
    with open(os.path.join(OUT_DIR, "q_injection_results.json"), "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved. Time: {time.time()-t0:.0f}s")
    print("Done!")


if __name__ == "__main__":
    main()
