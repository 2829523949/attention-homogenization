"""
Linear probe on ATTENTION MAPS (not hidden states).
attention_map [64] -> Linear(64, 2) -> landmark (x, y)
Truly no spatial conv, no position decoding.
Measures: does collapsed attention lose linearly accessible spatial info?
"""
import os, sys, gc, torch, json, math, time, numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

BASE = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
IMG_DIR = os.path.join(BASE, "AFLW2000-3D_crop")
EVAL_DIR = os.path.join(BASE, "eval")
LIST_FILE = os.path.join(BASE, "AFLW2000-3D_crop.list")
OUT_DIR = "/root/Qwen3-VL/paper/figures/fixed_test"
os.makedirs(OUT_DIR, exist_ok=True)

N_IMAGES = 500
QUERIES = ["nose", "eyes", "mouth", "chin", "forehead", "beard"]
LAYERS = [17, 18, 19, 24, 25, 26, 27]
TRAIN_FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 100
DEVICE = "cuda"


class AttnLinearProbe(nn.Module):
    """Linear(64 -> 2): predict landmark (x,y) from 64-dim attention map."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        return self.fc(x)  # [B, 2]


def get_landmarks(pts, roi_box, idx):
    xo, yo = pts[idx, 0, :], pts[idx, 1, :]
    l, t, r, b = roi_box[idx]
    cw, ch = r - l, b - t
    if cw <= 0 or ch <= 0:
        return np.zeros((68, 2), dtype=np.float32)
    xc = np.clip((xo - l) * 8 / cw, 0, 7)  # scale to token grid [0, 7]
    yc = np.clip((yo - t) * 8 / ch, 0, 7)
    return np.stack([xc, yc], axis=-1).astype(np.float32)


class SimpleDataset(Dataset):
    def __init__(self, attn, coords):
        self.attn = attn
        self.coords = coords
    def __len__(self):
        return len(self.attn)
    def __getitem__(self, i):
        return self.attn[i], self.coords[i]


def main():
    print("=" * 70)
    print("ATTENTION LINEAR PROBE + LOW-DATA")
    print("=" * 70)

    print("\n[1/3] Extracting attention maps...")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager",
        local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
    lm = model.model.language_model

    with open(LIST_FILE) as f:
        all_names = [l.strip() for l in f if l.strip()]
    step = max(1, len(all_names) // N_IMAGES)
    sample_idx = list(range(0, len(all_names), step))[:N_IMAGES]
    img_names = [all_names[i] for i in sample_idx]
    pts68 = np.load(os.path.join(EVAL_DIR, "AFLW2000-3D.pts68.npy"))
    roi = np.load(os.path.join(EVAL_DIR, "AFLW2000-3D_crop.roi_box.npy"))

    # Per-query, per-layer attention maps
    all_attn = {q: {l: [] for l in LAYERS} for q in QUERIES}
    all_gt = {q: [] for q in QUERIES}
    t0 = time.time()

    for gi, gname in enumerate(img_names):
        if gi % 100 == 0:
            print(f"  [{gi}/{N_IMAGES}] {time.time()-t0:.0f}s")
        imp = os.path.join(IMG_DIR, gname)
        if not os.path.exists(imp):
            continue
        img = Image.open(imp).convert("RGB")

        for query in QUERIES:
            msg = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":f"describe the {query}"}]}]
            prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inp = processor(text=prompt, images=img, return_tensors='pt').to('cuda')
            ids = inp['input_ids'][0]
            ve = (ids == 151653).nonzero(as_tuple=True)[0].item()
            nvt = ve - 4
            qs, qe = ve + 1, (ids == 151645).nonzero(as_tuple=True)[0].item()
            all_toks = [processor.decode([t]).strip() for t in ids.tolist()]
            kw = [qs + i for i, t in enumerate(all_toks[qs:qe]) if query.lower() in t.lower()]

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
                a = torch.stack(aw)  # [n_layers, heads, q, k]
                for li, l in enumerate(LAYERS):
                    a_l = a[li].mean(dim=0)  # [q, k]
                    if kw:
                        spec = a_l[kw, 4:4+nvt].mean(dim=0).cpu().float().numpy()
                    else:
                        spec = np.zeros(nvt)
                    all_attn[query][l].append(torch.from_numpy(spec))
            del aw, a, inp
            torch.cuda.empty_cache()

            coords = get_landmarks(pts68, roi, sample_idx[gi])
            all_gt[query].append(torch.from_numpy(coords))

        gc.collect()
        torch.cuda.empty_cache()

    del model, lm, processor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Done: {time.time()-t0:.0f}s")

    # ---- Train probes per query ----
    print("\n[2/3] Training linear probes per query...")

    layer_groups = {
        "L18-20": [17, 18, 19],
        "L25-27": [24, 25, 26],
        "Last": [27],
    }

    all_results = {}
    for q in QUERIES:
        all_results[q] = {}
        for gname, glayers in layer_groups.items():
            # Average attention across layers in group
            attn_list = []
            for l in glayers:
                attn_list.append(torch.stack(all_attn[q][l]))
            attn = torch.stack(attn_list).mean(dim=0)  # [N, 64]
            gt = torch.stack(all_gt[q])  # [N, 68, 2]

            N = attn.size(0)
            rng = np.random.RandomState(42).permutation(N)
            n_train = int(N * 0.8)

            layer_results = {}
            for frac in TRAIN_FRACTIONS:
                n_frac = max(4, int(n_train * frac))
                tr_i = rng[:n_frac]
                va_i = rng[n_train:]

                # Train SEPARATE probe per landmark (Linear(64, 2) each)
                # Use mean landmark position for simplicity
                gt_mean = gt[:, :, :].mean(dim=1)  # [N, 2] - face center

                tr_attn = attn[tr_i].float()
                tr_gt = gt_mean[tr_i].float()
                va_attn = attn[va_i].float()
                va_gt = gt_mean[va_i].float()

                td = DataLoader(SimpleDataset(tr_attn, tr_gt), BATCH_SIZE, shuffle=True)
                vd = DataLoader(SimpleDataset(va_attn, va_gt), BATCH_SIZE, shuffle=False)

                probe = AttnLinearProbe().to(DEVICE)
                opt = torch.optim.Adam(probe.parameters(), lr=LR)
                best_loss = 1e9
                patience = 0
                for ep in range(EPOCHS):
                    probe.train()
                    for fb, cb in td:
                        fb, cb = fb.to(DEVICE), cb.to(DEVICE)
                        loss = nn.MSELoss()(probe(fb), cb)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    probe.eval()
                    vl = 0
                    with torch.no_grad():
                        for fb, cb in vd:
                            fb, cb = fb.to(DEVICE), cb.to(DEVICE)
                            vl += nn.MSELoss()(probe(fb), cb).item() * fb.size(0)
                    vl /= len(vd.dataset)
                    if vl < best_loss:
                        best_loss = vl
                        patience = 0
                    else:
                        patience += 1
                        if patience >= 30:
                            break

                probe.eval()
                with torch.no_grad():
                    pred = probe(va_attn.to(DEVICE)).cpu()
                    err = torch.sqrt((pred[:, 0] - va_gt[:, 0])**2 + (pred[:, 1] - va_gt[:, 1])**2)
                    mean_err = err.mean().item()

                layer_results[f"frac_{frac}"] = {
                    "mean_dist_tokens": round(mean_err, 3),
                    "n_train": n_frac,
                }

            all_results[q][gname] = layer_results
            # Print per-query summary
            print(f"  {q:10s} {gname:8s}: ", end="")
            for f in TRAIN_FRACTIONS:
                v = layer_results[f"frac_{f}"]["mean_dist_tokens"]
                print(f"{v:.3f} ", end="")
            print()

    # ---- Summary across queries ----
    print("\n[3/3] Cross-query summary (mean face center error):")
    for gname in layer_groups:
        print(f"\n{gname}:")
        print(f"{'Query':12s}", end="")
        for f in TRAIN_FRACTIONS:
            print(f"{f*100:>7.0f}%", end="")
        print()
        print("-" * 60)
        for q in QUERIES:
            print(f"{q:12s}", end="")
            for f in TRAIN_FRACTIONS:
                v = all_results[q][gname][f"frac_{f}"]["mean_dist_tokens"]
                print(f"{v:>7.3f}", end="")
            print()

    save_data = {
        "config": {
            "model": "Qwen3-VL-2B", "n_images": N_IMAGES,
            "probe": "Linear(64->2) on spec attention maps",
            "layer_groups": list(layer_groups.keys()),
            "queries": QUERIES,
            "train_fractions": TRAIN_FRACTIONS,
        },
        "results": all_results,
    }
    save_path = os.path.join(OUT_DIR, "attn_linear_probe_results.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")
    print(f"Total: {time.time()-t0:.0f}s")
    print("Done!")


if __name__ == "__main__":
    main()
