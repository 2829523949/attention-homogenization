"""
Mid-layer routing demonstration on dot test.
Compare L18-20 vs L25-27: predict block color from attention maps.
Simple probe: k-NN on spec peaks + linear classifier on full maps.
"""
import os, sys, gc, torch, json, math, time
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

OUT_DIR = "/root/Qwen3-VL/paper/figures/fixed_test"
os.makedirs(OUT_DIR, exist_ok=True)

# Dot test image: 256x256 with 3 colored blocks
# Generate it (same as original dot test)
SIZE = 256
BLOCK_CENTERS = {"red": (24, 24), "green": (120, 120), "blue": (232, 232)}
CELL = 16

COLORS = ["red", "green", "blue"]
LAYER_CONDITIONS = [([17,18,19], "L18-20"), ([24,25,26], "L25-27"), ([27], "Last")]


def generate_dot_image():
    img = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    draw = __import__("PIL").ImageDraw.Draw(img)
    for color_name, (cx, cy) in BLOCK_CENTERS.items():
        rgb = {"red": (255,0,0), "green": (0,255,0), "blue": (0,0,255)}[color_name]
        half = CELL // 2
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(SIZE-1, cx + half), min(SIZE-1, cy + half)
        draw.rectangle([x0, y0, x1, y1], fill=rgb)
    return img


def main():
    print("="*60)
    print("MID-LAYER ROUTING ON DOT TEST")
    print("="*60)

    # Generate dot image
    img = generate_dot_image()
    img.save(os.path.join(OUT_DIR, "dot_test_routing.png"))
    print(f"\n[1/4] Dot image generated ({SIZE}x{SIZE})")

    # Load Qwen
    print("\n[2/4] Loading Qwen3-VL-2B...")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager",
        local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
    lm = model.model.language_model

    # Extract features
    print("\n[3/4] Extracting attention per layer condition...")
    features = {}
    peaks = {}

    for layer_indices, name in LAYER_CONDITIONS:
        print(f"  {name}: layers {layer_indices}")
        maps_by_color = {c: [] for c in COLORS}

        for color in COLORS:
            query = f"describe the {color} area"
            msg = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":query}]}]
            prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inp = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
            ids = inp["input_ids"][0]
            ve = (ids == 151653).nonzero(as_tuple=True)[0].item()
            ie = (ids == 151645).nonzero(as_tuple=True)[0].item()
            vs, nvt = 4, ve - 4
            nd = int(math.isqrt(nvt))
            qs, qe = ve + 1, ie

            all_toks = [processor.decode([t]).strip() for t in ids.tolist()]
            kw_col = color  # match "red", "green", "blue" in "red area", etc.
            kw = [qs + i for i, t in enumerate(all_toks[qs:qe]) if kw_col in t.lower()]

            for l in layer_indices:
                lm.layers[l].self_attn.output_attentions = True

            aw = []
            def mk(w):
                def hk(m,i,o):
                    if isinstance(o,tuple) and len(o)>1 and o[1] is not None:
                        w.append(o[1].detach())
                return hk
            handles = [lm.layers[l].self_attn.register_forward_hook(mk(aw)) for l in layer_indices]
            with torch.no_grad():
                model(**inp)
            for h in handles:
                h.remove()

            if aw:
                a = torch.stack(aw).mean(dim=0).mean(dim=[0,1])
                if kw:
                    spec = a[kw, vs:ve].mean(dim=0).cpu().float().numpy()
                else:
                    spec = np.zeros(nvt)
                spec_map = spec.reshape(nd, nd)
                peak = np.unravel_index(spec_map.argmax(), spec_map.shape)
            else:
                spec = np.zeros(nvt)
                peak = (0, 0)

            maps_by_color[color] = spec
            if color not in peaks:
                peaks[color] = {}
            peaks[color][name] = (int(peak[1]), int(peak[0]))  # (col, row)

            del aw, a, inp
            torch.cuda.empty_cache()

        # Build feature matrix: [3 samples, nvt features]
        X = np.array([maps_by_color[c] for c in COLORS])
        y = np.array([0, 1, 2])  # red=0, green=1, blue=2
        features[name] = (X, y)

        gc.collect()
        torch.cuda.empty_cache()

    # ---- Classify ----
    print("\n[4/4] Classification results...")
    results = {}
    for name in [n for _, n in LAYER_CONDITIONS]:
        X, y = features[name]

        # Nearest-centroid: classify by nearest peak
        centroids = {}
        for ci, color in enumerate(COLORS):
            centroids[color] = X[ci]

        # Leave-one-out cross-validation
        correct_nc = 0
        correct_lr = 0
        for i in range(3):
            # Nearest centroid
            dists = [np.linalg.norm(X[i] - centroids[c]) for c in COLORS]
            pred_nc = np.argmin(dists)
            correct_nc += (pred_nc == i)

            # Logistic regression (train on 2, test on 1)
            train_idx = [j for j in range(3) if j != i]
            clf = LogisticRegression(max_iter=1000, multi_class='ovr')
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[i].reshape(1, -1))
            clf.fit(X_train, y[train_idx])
            pred_lr = clf.predict(X_test)[0]
            correct_lr += (pred_lr == i)

        acc_nc = correct_nc / 3
        acc_lr = correct_lr / 3

        print(f"\n  {name}:")
        print(f"    Nearest-centroid accuracy: {acc_nc:.0%}")
        print(f"    Logistic regression accuracy: {acc_lr:.0%}")
        for c in COLORS:
            print(f"    {c}: peak={peaks[c][name]}")

        results[name] = {
            "nearest_centroid_acc": round(acc_nc, 3),
            "logistic_regression_acc": round(acc_lr, 3),
            "peaks": {c: peaks[c][name] for c in COLORS},
        }

    save_path = os.path.join(OUT_DIR, "dot_routing_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {save_path}")
    del model, lm, processor
    gc.collect()
    torch.cuda.empty_cache()
    print("Done!")


if __name__ == "__main__":
    main()
