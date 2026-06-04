"""
Linear probe + low-data scaling on AFLW2000.
Reveals: collapse destroys linear accessibility more than total information.
"""
import os, sys, gc, torch, json, math, time
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
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
LAYER_CONDITIONS = [([17, 18, 19], "L18-20"), ([24, 25, 26], "L25-27"), ([27], "Last")]
TRAIN_FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
HIDDEN_DIM = 2048
GRID_SZ = 8
HEATMAP_SZ = 32
N_LANDMARKS = 68
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 100
DEVICE = "cuda"


class LinearProbe(nn.Module):
    """No spatial convolution: 1x1 conv per pixel, then avg pool + linear."""
    def __init__(self, in_dim=2048, n_landmarks=68):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_dim, 128, kernel_size=1)
        self.fc = nn.Linear(128, n_landmarks * 2)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, 8, 8, -1).permute(0, 3, 1, 2)  # [B, 2048, 8, 8]
        x = self.conv1x1(x)                              # [B, 128, 8, 8]
        x = x.mean(dim=[2, 3])                           # [B, 128]
        return self.fc(x).view(-1, 68, 2)               # [B, 68, 2]


def get_landmarks(pts, roi_box, idx):
    xo, yo = pts[idx, 0, :], pts[idx, 1, :]
    l, t, r, b = roi_box[idx]
    cw, ch = r - l, b - t
    if cw <= 0 or ch <= 0:
        return np.zeros((68, 2), dtype=np.float32)
    xc = np.clip((xo - l) * HEATMAP_SZ / cw, 0, HEATMAP_SZ - 1)
    yc = np.clip((yo - t) * HEATMAP_SZ / ch, 0, HEATMAP_SZ - 1)
    return np.stack([xc, yc], axis=-1).astype(np.float32)


def compute_pck(pred, gt):
    dist = torch.sqrt((pred[:, :, 0] - gt[:, :, 0])**2 + (pred[:, :, 1] - gt[:, :, 1])**2)
    fd = math.sqrt(HEATMAP_SZ**2 + HEATMAP_SZ**2)
    results = {}
    for th in [0.03, 0.05, 0.1, 0.2]:
        results[th] = round(100 * (dist < th * fd).float().mean().item(), 1)
    return results, dist.mean().item()


class SimpleDataset(Dataset):
    def __init__(self, feats, coords):
        self.feats = feats
        self.coords = coords
    def __len__(self):
        return len(self.feats)
    def __getitem__(self, i):
        return self.feats[i], self.coords[i]


def main():
    print("=" * 70)
    print("LINEAR PROBE + LOW-DATA SCALING")
    print("=" * 70)

    # ---- Extract features ----
    print("\n[1/3] Extracting features...")
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

    features = {n: [] for _, n in LAYER_CONDITIONS}
    coords_list = []
    t0 = time.time()

    for gi, gname in enumerate(img_names):
        if gi % 100 == 0:
            print(f"  [{gi}/{N_IMAGES}] {time.time()-t0:.0f}s")
        imp = os.path.join(IMG_DIR, gname)
        if not os.path.exists(imp):
            continue
        img = Image.open(imp).convert("RGB")
        msg = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"describe this image"}]}]
        prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = processor(text=prompt, images=img, return_tensors='pt').to('cuda')
        ids = inp['input_ids'][0]
        ve = (ids == 151653).nonzero(as_tuple=True)[0].item()
        nvt = ve - 4

        collected = {}
        hooks = []
        tl = set()
        for li, _ in LAYER_CONDITIONS:
            for l in li:
                tl.add(l)
        for l in sorted(tl):
            def mkh(lid):
                def hk(m, i, o):
                    oo = o[0] if isinstance(o, tuple) else o
                    collected[lid] = oo[:, 4:4+nvt].detach().cpu().float()
                return hk
            hooks.append(lm.layers[l].register_forward_hook(mkh(l)))
        with torch.no_grad():
            model(**inp)
        for h in hooks:
            h.remove()

        for li, nm in LAYER_CONDITIONS:
            if len(li) == 1:
                feat = collected[li[0]]
            else:
                feat = torch.stack([collected[l] for l in li]).mean(dim=0)
            features[nm].append(feat.squeeze(0))

        coords = get_landmarks(pts68, roi, sample_idx[gi])
        coords_list.append(torch.from_numpy(coords))
        del collected, inp
        torch.cuda.empty_cache()

    feat_tensors = {n: torch.stack(lst) for n, lst in features.items()}
    all_coords = torch.stack(coords_list)
    del model, lm, processor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Done: {len(img_names)} imgs in {time.time()-t0:.0f}s")

    # ---- Train linear probes ----
    print("\n[2/3] Training linear probes...")
    N = feat_tensors["L18-20"].size(0)
    rng = np.random.RandomState(42).permutation(N)
    n_train = int(N * 0.8)
    train_idx, val_idx = rng[:n_train], rng[n_train:]

    all_results = {}
    for li, nm in LAYER_CONDITIONS:
        print(f"\n  --- {nm} ---")
        feat_all = feat_tensors[nm].float()
        layer_results = {}

        for frac in TRAIN_FRACTIONS:
            n_frac = max(4, int(n_train * frac))
            frac_idx = train_idx[:n_frac]

            ft = feat_all[frac_idx].clone()
            fv = feat_all[val_idx].clone()
            fm = ft.mean(dim=(0, 1), keepdim=True)
            fs = ft.std(dim=(0, 1), keepdim=True) + 1e-8
            ft = (ft - fm) / fs
            fv = (fv - fm) / fs

            td = DataLoader(SimpleDataset(ft, all_coords[frac_idx]), BATCH_SIZE, shuffle=True)
            vd = DataLoader(SimpleDataset(fv, all_coords[val_idx]), BATCH_SIZE, shuffle=False)

            probe = LinearProbe().to(DEVICE)
            opt = torch.optim.Adam(probe.parameters(), lr=LR)
            best_loss = 1e9
            best_state = None
            patience = 0

            for ep in range(EPOCHS):
                probe.train()
                for fb, cb in td:
                    fb, cb = fb.to(DEVICE), cb.to(DEVICE)
                    p = probe(fb)
                    loss = F.mse_loss(p, cb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                probe.eval()
                vl = 0.0
                with torch.no_grad():
                    for fb, cb in vd:
                        fb, cb = fb.to(DEVICE), cb.to(DEVICE)
                        p = probe(fb)
                        vl += F.mse_loss(p, cb).item() * fb.size(0)
                vl /= len(vd.dataset)
                if vl < best_loss:
                    best_loss = vl
                    best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= 30:
                        break

            probe.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
            probe.eval()
            with torch.no_grad():
                all_pred, all_gt = [], []
                for fb, cb in vd:
                    fb = fb.to(DEVICE)
                    all_pred.append(probe(fb).cpu())
                    all_gt.append(cb.cpu())
                pck, mean_dist = compute_pck(torch.cat(all_pred), torch.cat(all_gt))

            layer_results[f"frac_{frac}"] = {
                "pck_005": pck[0.05],
                "mean_dist": round(mean_dist, 2),
                "n_train": n_frac,
            }
            print(f"    {frac*100:5.0f}% ({n_frac:3d} imgs): PCK@0.05={pck[0.05]}%  dist={mean_dist:.2f}px")

        all_results[nm] = layer_results

    # ---- Summary ----
    print("\n[3/3] Summary:")
    print(f"{'Layer':10s}", end="")
    for f in TRAIN_FRACTIONS:
        print(f"{f*100:>7.0f}%", end="")
    print()
    print("-" * 60)
    for nm in [n for _, n in LAYER_CONDITIONS]:
        print(f"{nm:10s}", end="")
        for f in TRAIN_FRACTIONS:
            print(f"{all_results[nm][f'frac_{f}']['pck_005']:>7.1f}", end="")
        print()

    save_data = {
        "config": {
            "model": "Qwen3-VL-2B", "n_images": N_IMAGES,
            "probe": "Linear(2048->136)",
            "layer_conditions": [n for _, n in LAYER_CONDITIONS],
            "train_fractions": TRAIN_FRACTIONS,
        },
        "results": all_results,
    }
    save_path = os.path.join(OUT_DIR, "linear_probe_results.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")
    print(f"Total: {time.time()-t0:.0f}s")
    print("Done!")


if __name__ == "__main__":
    main()
