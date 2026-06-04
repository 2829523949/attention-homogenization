"""
Landmark prediction visualization: train probes on L18-20 and Last layer,
then generate comparison figure for 4 AFLW2000 images.
"""
import os, sys, gc, torch, json, math, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
LAYER_CONDITIONS = [([17, 18, 19], "L18-20"), ([27], "Last")]
HIDDEN_DIM = 2048
GRID_SZ = 8
HEATMAP_SZ = 32
N_LANDMARKS = 68
BATCH_SIZE = 32
LR = 5e-4
EPOCHS = 50
DEVICE = "cuda"

VIZ_NAMES = ["image00002.jpg", "image00783.jpg", "image01409.jpg", "image01749.jpg"]
VIZ_LABELS = ["Frontal", "Left Yaw", "Right Yaw", "Difficult Pose"]


class FeatureDataset(Dataset):
    def __init__(self, features, coords):
        self.features = features.cpu()
        self.coords = coords.cpu()
    def __len__(self):
        return self.features.size(0)
    def __getitem__(self, idx):
        return self.features[idx], self.coords[idx]


class CoordProbe(nn.Module):
    def __init__(self, in_dim=HIDDEN_DIM, grid_sz=GRID_SZ, n_landmarks=N_LANDMARKS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 256, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(grid_sz * grid_sz * 128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, n_landmarks * 2),
        )
    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, GRID_SZ, GRID_SZ, -1).permute(0, 3, 1, 2)
        return self.net(x).view(B, N_LANDMARKS, 2)


def get_landmarks(pts, roi_box, idx):
    xo, yo = pts[idx, 0, :], pts[idx, 1, :]
    l, t, r, b = roi_box[idx]
    cw, ch = r - l, b - t
    if cw <= 0 or ch <= 0:
        return np.zeros((68, 2), dtype=np.float32)
    xc = np.clip((xo - l) * HEATMAP_SZ / cw, 0, HEATMAP_SZ - 1)
    yc = np.clip((yo - t) * HEATMAP_SZ / ch, 0, HEATMAP_SZ - 1)
    return np.stack([xc, yc], axis=-1).astype(np.float32)


def main():
    print("=" * 60)
    print("LANDMARK VIZ")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading Qwen...")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager",
        local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
    lm = model.model.language_model

    # Load images
    print("\n[2/4] Extracting features...")
    with open(LIST_FILE) as f:
        all_names = [l.strip() for l in f if l.strip()]
    step = max(1, len(all_names) // N_IMAGES)
    # Step-sample (reserve 4 slots for viz images)
    n_base = N_IMAGES - len(VIZ_NAMES)
    base_idx = list(range(0, len(all_names), step))[:n_base]
    sample_idx_set = set(base_idx)
    # Append any missing viz images (guarantees inclusion)
    for vi, vn in enumerate(all_names):
        if vn in VIZ_NAMES and vi not in sample_idx_set:
            base_idx.append(vi)
    sample_idx = base_idx[:N_IMAGES]
    img_names = [all_names[i] for i in sample_idx]
    pts68 = np.load(os.path.join(EVAL_DIR, "AFLW2000-3D.pts68.npy"))
    roi = np.load(os.path.join(EVAL_DIR, "AFLW2000-3D_crop.roi_box.npy"))

    features = {n: [] for _, n in LAYER_CONDITIONS}
    coords_list = []
    viz_data = {}

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

        coll = {}
        hooks = []
        tl = set()
        for li, _ in LAYER_CONDITIONS:
            for l in li:
                tl.add(l)
        for l in sorted(tl):
            def mkh(lid):
                def hk(m, i, o):
                    oo = o[0] if isinstance(o, tuple) else o
                    coll[lid] = oo[:, 4:4+nvt].detach().cpu().float()
                return hk
            hooks.append(lm.layers[l].register_forward_hook(mkh(l)))
        with torch.no_grad():
            model(**inp)
        for h in hooks:
            h.remove()

        for li, nm in LAYER_CONDITIONS:
            if len(li) == 1:
                feat = coll[li[0]]
            else:
                feat = torch.stack([coll[l] for l in li]).mean(dim=0)
            features[nm].append(feat.squeeze(0))

        coords = get_landmarks(pts68, roi, sample_idx[gi])
        coords_list.append(torch.from_numpy(coords))

        if gname in VIZ_NAMES:
            viz_data[gname] = img.copy()

        del coll, inp
        torch.cuda.empty_cache()

    feat_tensors = {n: torch.stack(lst) for n, lst in features.items()}
    all_coords = torch.stack(coords_list)
    del model, lm, processor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Done: {len(img_names)} images in {time.time()-t0:.0f}s")

    # Train probes
    print("\n[3/4] Training probes...")
    N = feat_tensors["L18-20"].size(0)
    rng = np.random.RandomState(42).permutation(N)
    n_tr = int(N * 0.8)
    tr_idx, va_idx = rng[:n_tr], rng[n_tr:]

    probes = {}
    for nm in [n for _, n in LAYER_CONDITIONS]:
        print(f"  Training {nm}...")
        ft = feat_tensors[nm][tr_idx].float().clone()
        fv = feat_tensors[nm][va_idx].float().clone()
        fm = ft.mean(dim=(0, 1), keepdim=True)
        fs = ft.std(dim=(0, 1), keepdim=True) + 1e-8
        ft = (ft - fm) / fs
        fv = (fv - fm) / fs
        td = DataLoader(FeatureDataset(ft, all_coords[tr_idx]), BATCH_SIZE, shuffle=True)
        vd = DataLoader(FeatureDataset(fv, all_coords[va_idx]), BATCH_SIZE, shuffle=False)

        probe = CoordProbe().to(DEVICE)
        opt = torch.optim.Adam(probe.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
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
            vl = 0
            with torch.no_grad():
                for fb, cb in vd:
                    fb, cb = fb.to(DEVICE), cb.to(DEVICE)
                    p = probe(fb)
                    vl += F.mse_loss(p, cb).item() * fb.size(0)
            vl /= len(vd.dataset)
            sched.step(vl)
            if vl < best_loss:
                best_loss = vl
                best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 15:
                    break
            if (ep + 1) % 15 == 0:
                print(f"    ep {ep+1}: val={vl:.6f}")

        probe.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        probe._mean = fm
        probe._std = fs
        probes[nm] = probe

    # Visualize
    print("\n[4/4] Generating figure...")
    fig, axes = plt.subplots(4, 4, figsize=(14, 16))

    for row, vname in enumerate(VIZ_NAMES):
        if vname not in img_names:
            print(f"  WARNING: {vname} not in dataset, skipping")
            continue
        gi = img_names.index(vname)
        if vname not in viz_data:
            print(f"  WARNING: {vname} not in viz_data, skipping")
            continue
        img = viz_data[vname]
        gt = get_landmarks(pts68, roi, sample_idx[gi])
        img_small = img.resize((120, 120))

        axes[row, 0].imshow(img_small)
        axes[row, 0].set_title(VIZ_LABELS[row], fontsize=12, fontweight='bold')
        axes[row, 0].axis("off")

        # GT
        axes[row, 1].imshow(img_small, alpha=0.6)
        axes[row, 1].scatter(gt[:, 0] * 120 / HEATMAP_SZ, gt[:, 1] * 120 / HEATMAP_SZ,
                             c='lime', s=6, alpha=0.8)
        axes[row, 1].set_title("Ground Truth", fontsize=11)
        axes[row, 1].axis("off")

        # Last-layer
        p_last = probes["Last"]
        p_last.eval()
        with torch.no_grad():
            fl = feat_tensors["Last"][gi:gi+1].float().to(DEVICE)
            fl = (fl - p_last._mean.to(DEVICE)) / p_last._std.to(DEVICE)
            pr_l = p_last(fl)[0].cpu().numpy()
        axes[row, 2].imshow(img_small, alpha=0.6)
        axes[row, 2].scatter(pr_l[:, 0] * 120 / HEATMAP_SZ, pr_l[:, 1] * 120 / HEATMAP_SZ,
                             c='red', s=6, alpha=0.8)
        axes[row, 2].set_title("Last-Layer Probe", fontsize=11)
        axes[row, 2].axis("off")

        # Mid-layer
        p_mid = probes["L18-20"]
        p_mid.eval()
        with torch.no_grad():
            fm = feat_tensors["L18-20"][gi:gi+1].float().to(DEVICE)
            fm = (fm - p_mid._mean.to(DEVICE)) / p_mid._std.to(DEVICE)
            pr_m = p_mid(fm)[0].cpu().numpy()
        axes[row, 3].imshow(img_small, alpha=0.6)
        axes[row, 3].scatter(pr_m[:, 0] * 120 / HEATMAP_SZ, pr_m[:, 1] * 120 / HEATMAP_SZ,
                             c='cyan', s=6, alpha=0.8)
        axes[row, 3].set_title("Mid-Layer (L18-20) Probe", fontsize=11)
        axes[row, 3].axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "landmark_prediction_comparison.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
    print("Done!")


if __name__ == "__main__":
    main()
