"""
Multi-seed mid-layer probe: 5 seeds, L18-20 vs Last, 4 fractions.
Extracts hidden states once, re-splits per seed.
Output: multiseed_midlayer_results.json
"""
import os, sys, gc, torch, json, math, time, numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"; os.environ["HF_HUB_OFFLINE"] = "1"

BASE = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
IMG_DIR = os.path.join(BASE, "AFLW2000-3D_crop")
LIST_FILE = os.path.join(BASE, "AFLW2000-3D_crop.list")
EVAL_DIR = os.path.join(BASE, "eval")
OUT_DIR = "/root/Qwen3-VL/paper/figures/fixed_test"
os.makedirs(OUT_DIR, exist_ok=True)

N_IMAGES = 500
N_SEEDS = 5
TRAIN_FRACTIONS = [0.1, 0.25, 0.5, 1.0]
LAYER_CONDITIONS = [([17, 18, 19], "L18-20"), ([27], "Last")]
HEATMAP_SZ = 32
BATCH_SIZE = 32
LR = 5e-4
EPOCHS = 100
PATIENCE = 15

def load_landmarks(eval_dir):
    pts = np.load(os.path.join(eval_dir, "AFLW2000-3D.pts68.npy"))
    roi = np.load(os.path.join(eval_dir, "AFLW2000-3D_crop.roi_box.npy"))
    lms = []
    for i in range(len(pts)):
        x1, y1, x2, y2 = roi[i]
        w, h = (x2 - x1), (y2 - y1)
        if w <= 0 or h <= 0:
            lms.append(np.zeros((68, 2)))
            continue
        lx = (pts[i, 0] - x1) * HEATMAP_SZ / w
        ly = (pts[i, 1] - y1) * HEATMAP_SZ / h
        lms.append(np.stack([lx, ly], axis=-1))
    return np.array(lms)

class CoordProbe(torch.nn.Module):
    def __init__(self, in_dim=2048, grid_sz=8, n_landmarks=68):
        super().__init__()
        self.grid_sz = grid_sz
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, 256, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 128, 3, padding=1), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(128),
            torch.nn.Flatten(),
            torch.nn.Linear(grid_sz*grid_sz*128, 256), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, n_landmarks*2))
    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.grid_sz, self.grid_sz, -1).permute(0, 3, 1, 2)
        return self.net(x).view(B, 68, 2)

def compute_pck(pred, gt, thresh=0.05):
    dist = torch.sqrt((pred[:, :, 0] - gt[:, :, 0])**2 + (pred[:, :, 1] - gt[:, :, 1])**2)
    face_diag = math.sqrt(HEATMAP_SZ**2 + HEATMAP_SZ**2)
    return (dist < thresh * face_diag).float().mean().item() * 100

def extract_features():
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
    step = max(1, len(all_names)//N_IMAGES)
    idxs = list(range(0, len(all_names), step))[:N_IMAGES]
    img_names = [all_names[i] for i in idxs]
    pts68 = np.load(os.path.join(EVAL_DIR, "AFLW2000-3D.pts68.npy"))
    roi = np.load(os.path.join(EVAL_DIR, "AFLW2000-3D_crop.roi_box.npy"))

    features = {n: [] for _, n in LAYER_CONDITIONS}
    coords_list = []
    t0 = time.time()
    for img_idx, img_name in enumerate(img_names):
        if img_idx % 50 == 0:
            print(f"  [{img_idx}/{N_IMAGES}] {time.time()-t0:.0f}s")
        p = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(p): continue
        img = Image.open(p).convert("RGB")
        msg = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"describe this image"}]}]
        prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
        ids = inp["input_ids"][0]
        ve = (ids==151653).nonzero(as_tuple=True)[0].item()
        nvt = ve-4
        collected = {}
        target_layers = set()
        for lii, _ in LAYER_CONDITIONS:
            for l in lii: target_layers.add(l)
        hooks = []
        for l in sorted(target_layers):
            def mk(lid):
                def hook(m,i,o):
                    out = o[0] if isinstance(o, tuple) else o
                    collected[lid] = out[:, 4:4+nvt].detach().cpu().float()
                return hook
            hooks.append(lm.layers[l].register_forward_hook(mk(l)))
        with torch.no_grad(): model(**inp)
        for h in hooks: h.remove()
        for lii, name in LAYER_CONDITIONS:
            if len(lii) == 1:
                feat = collected[lii[0]]
            else:
                feat = torch.stack([collected[l] for l in lii]).mean(dim=0)
            features[name].append(feat.squeeze(0))
        coords_list.append(torch.from_numpy(load_landmarks(EVAL_DIR)[idxs[img_idx]]))
        del collected, inp; torch.cuda.empty_cache()
    feat_tensors = {n: torch.stack(lst) for n, lst in features.items()}
    coords = torch.stack(coords_list)
    del model, lm, processor; gc.collect(); torch.cuda.empty_cache()
    print(f"  Extracted {len(img_names)} images. Time: {time.time()-t0:.0f}s")
    return feat_tensors, coords

def train_probe(train_loader, val_loader, device="cuda"):
    model = CoordProbe().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    best_pck = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for f, c in train_loader:
            f, c = f.to(device), c.to(device)
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(f), c)
            loss.backward(); opt.step()
        model.eval()
        val_loss = 0.0
        all_p, all_g = [], []
        with torch.no_grad():
            for f, c in val_loader:
                f, c = f.to(device), c.to(device)
                pred = model(f)
                val_loss += torch.nn.functional.mse_loss(pred, c).item() * f.size(0)
                all_p.append(pred.cpu()); all_g.append(c.cpu())
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        all_p = torch.cat(all_p); all_g = torch.cat(all_g)
        pck_val = compute_pck(all_p, all_g)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_pck = pck_val
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE: break
    return round(best_pck, 1)

def main():
    print("="*60)
    print("MULTI-SEED MID-LAYER PROBE")
    print("="*60)
    feat_tensors, coords = extract_features()
    N = feat_tensors[LAYER_CONDITIONS[0][1]].size(0)
    TRAIN_RATIO = 0.8
    print(f"\nN={N}, train_ratio={TRAIN_RATIO}")

    all_results = {}
    for lii, name in LAYER_CONDITIONS:
        print(f"\n=== {name} ===")
        feats = feat_tensors[name]
        layer_results = {}
        for frac in TRAIN_FRACTIONS:
            seed_pcks = []
            for seed in range(N_SEEDS):
                torch.manual_seed(42+seed)
                perm = torch.randperm(N)
                n_train = int(N * TRAIN_RATIO)
                train_idx = perm[:n_train]
                val_idx = perm[n_train:]
                n_frac = max(1, int(n_train * frac))
                frac_train_idx = train_idx[:n_frac]
                ft = feats[frac_train_idx].float().clone()
                fv = feats[val_idx].float().clone()
                fm = ft.mean(dim=(0,1), keepdim=True)
                fs = ft.std(dim=(0,1), keepdim=True) + 1e-8
                ft = (ft-fm)/fs; fv = (fv-fm)/fs
                tr_ds = torch.utils.data.TensorDataset(ft, coords[frac_train_idx])
                vl_ds = torch.utils.data.TensorDataset(fv, coords[val_idx])
                tr_ld = DataLoader(tr_ds, BATCH_SIZE, shuffle=True)
                vl_ld = DataLoader(vl_ds, BATCH_SIZE, shuffle=False)
                pck = train_probe(tr_ld, vl_ld)
                seed_pcks.append(pck)
                print(f"  seed={seed} {name} {frac*100:.0f}%: PCK={pck}%")
            mean_pck = round(float(np.mean(seed_pcks)), 1)
            std_pck = round(float(np.std(seed_pcks)), 1)
            layer_results[f"frac_{frac}"] = {"mean": mean_pck, "std": std_pck, "per_seed": seed_pcks}
            print(f"  => {name} {frac*100:.0f}%: {mean_pck}±{std_pck}%")
        all_results[name] = layer_results

    path = os.path.join(OUT_DIR, "multiseed_midlayer_results.json")
    with open(path, "w") as f:
        json.dump({"config":{"n_images":N_IMAGES,"n_seeds":N_SEEDS,"train_ratio":TRAIN_RATIO},"results":all_results}, f, indent=2)
    print(f"\nSaved: {path}")
    print("DONE.")

if __name__ == "__main__":
    main()
