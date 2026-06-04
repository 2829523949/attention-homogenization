"""
Dot test attention heatmap comparison: L18-20 vs L25-27.
Shows that mid-layer attention peaks at correct block positions,
while last-layer attention collapses to a common position.
"""
import os, sys, gc, torch, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

OUT_DIR = "/root/Qwen3-VL/paper/figures/fixed_test"
SIZE = 256
BLOCK_CENTERS = {"red": (24, 24), "green": (120, 120), "blue": (232, 232)}
CELL = 16
LAYER_CONDITIONS = [([17, 18, 19], "L18-20"), ([24, 25, 26], "L25-27"), ([27], "Last")]


def generate_dot_image():
    img = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for cname, (cx, cy) in BLOCK_CENTERS.items():
        rgb = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}[cname]
        half = CELL // 2
        draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=rgb)
    return img


def main():
    print("=" * 60)
    print("DOT TEST HEATMAP COMPARISON")
    print("=" * 60)

    img = generate_dot_image()
    img.save(os.path.join(OUT_DIR, "dot_test_source.png"))

    print("\n[1/3] Loading Qwen...")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager",
        local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
    lm = model.model.language_model

    print("\n[2/3] Extracting attention...")
    results = {}

    for idx, (layer_indices, name) in enumerate(LAYER_CONDITIONS):
        maps = {}
        for color in ["red", "green", "blue"]:
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
            kw = [qs + i for i, t in enumerate(all_toks[qs:qe]) if color in t.lower()]

            for l in layer_indices:
                lm.layers[l].self_attn.output_attentions = True

            aw = []
            def mk(w):
                def hk(m, i, o):
                    if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
                        w.append(o[1].detach())
                return hk
            handles = [lm.layers[l].self_attn.register_forward_hook(mk(aw)) for l in layer_indices]
            with torch.no_grad():
                model(**inp)
            for h in handles:
                h.remove()

            if aw:
                a = torch.stack(aw).mean(dim=0).mean(dim=[0, 1])
                spec = a[kw, vs:ve].mean(dim=0).cpu().float().numpy() if kw else np.zeros(nvt)
            else:
                spec = np.zeros(nvt)
            maps[color] = spec.reshape(nd, nd)

            del aw, a, inp
            torch.cuda.empty_cache()

        results[name] = maps
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  {name}: red={np.unravel_index(maps['red'].argmax(), maps['red'].shape)}, "
              f"green={np.unravel_index(maps['green'].argmax(), maps['green'].shape)}, "
              f"blue={np.unravel_index(maps['blue'].argmax(), maps['blue'].shape)}")

    del model, lm, processor
    gc.collect()
    torch.cuda.empty_cache()

    print("\n[3/3] Generating figure...")
    layer_names = [n for _, n in LAYER_CONDITIONS]
    colors_viz = ["red", "green", "blue"]
    rgb_colors = [(1, 0, 0, 0.7), (0, 1, 0, 0.7), (0, 0, 1, 0.7)]
    gt_pos = [(1, 1), (7, 7), (14, 14)]

    fig, axes = plt.subplots(3, len(layer_names) + 1, figsize=(10, 9))

    for row, color in enumerate(colors_viz):
        # Column 0: source image with block outline
        ax = axes[row, 0]
        ax.imshow(img)
        cx, cy = BLOCK_CENTERS[color]
        rect = plt.Rectangle((cx - CELL//2, cy - CELL//2), CELL, CELL,
                             fill=False, edgecolor=rgb_colors[row][:3], linewidth=3)
        ax.add_patch(rect)
        ax.set_title(f"GT {color}" if row == 0 else "", fontsize=10)
        ax.axis("off")

        for col, layer_name in enumerate(layer_names):
            ax = axes[row, col + 1]
            attn_map = results[layer_name][color]
            im = ax.imshow(attn_map, cmap="hot", interpolation="nearest")
            peak = np.unravel_index(attn_map.argmax(), attn_map.shape)
            ax.plot(peak[1], peak[0], 'c*', markersize=12, markeredgecolor='white', markeredgewidth=1)
            ax.set_title(f"{layer_name}" if row == 0 else "", fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "dot_heatmap_comparison.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
    print("Done!")


if __name__ == "__main__":
    main()
