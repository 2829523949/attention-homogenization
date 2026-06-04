"""
PoolMLP / PoolLinear with 500 epochs.
Same feature extraction as geometry_probe_exp.py, but only probes.
Output: pool_probe_results.json
"""
import os, sys, gc, torch, json, math, time
import numpy as np
from PIL import Image
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

BASE = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
IMG_DIR = os.path.join(BASE, "AFLW2000-3D_crop")
LIST_FILE = os.path.join(BASE, "AFLW2000-3D_crop.list")
EVAL_DIR = os.path.join(BASE, "eval")
OUT_DIR = "/root/Qwen3-VL/paper/figures/fixed_test"
os.makedirs(OUT_DIR, exist_ok=True)

N_IMAGES = 200
N_EPOCHS = 500
LR = 1e-3
TRAIN_FRACTIONS = [0.1, 0.25, 0.5, 1.0]
N_SEEDS = 3
N_TEST = 40
HEATMAP_SZ = 32
CROP_SZ = 120

LAYER_GROUPS = {"L1-3": [0, 1, 2], "L18-20": [17, 18, 19], "Last": [27]}
GROUP_NAMES = list(LAYER_GROUPS.keys())

def load_landmarks(eval_dir):
    pts = np.load(os.path.join(eval_dir, "AFLW2000-3D.pts68.npy"))
    roi = np.load(os.path.join(eval_dir, "AFLW2000-3D_crop.roi_box.npy"))
    lms = []
    for i in range(len(pts)):
        x1, y1, x2, y2 = roi[i]
        w, h = (x2 - x1), (y2 - y1)
        lx = (pts[i, 0] - x1) * HEATMAP_SZ / w
        ly = (pts[i, 1] - y1) * HEATMAP_SZ / h
        lms.append(np.stack([lx, ly], axis=-1))
    return np.array(lms)

class PoolMLPProbe(torch.nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 512), torch.nn.ReLU(), torch.nn.Linear(512, 68*2))
    def forward(self, x):
        x = x.mean(dim=1); return self.net(x).view(-1, 68, 2)

class PoolLinearProbe(torch.nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        self.net = torch.nn.Linear(in_dim, 68*2)
    def forward(self, x):
        x = x.mean(dim=1); return self.net(x).view(-1, 68, 2)

def pck(pred, gt, threshold=0.05):
    dists = torch.norm(pred - gt, dim=-1)
    face_diag = math.sqrt(HEATMAP_SZ**2 + HEATMAP_SZ**2)
    return (dists < threshold * face_diag).float().mean().item() * 100

def compute_baseline(train_lms, test_lms):
    mean_per_lm = train_lms.mean(axis=0)
    face_diag = math.sqrt(HEATMAP_SZ**2 + HEATMAP_SZ**2)
    dists = np.sqrt(((test_lms - mean_per_lm[None, :, :])**2).sum(axis=-1))
    return (dists < 0.05 * face_diag).mean() * 100

def main():
    print("="*70); print("POOL PROBE TEST (500 epochs)"); print("="*70)

    all_lms_t = torch.from_numpy(load_landmarks(EVAL_DIR)).float()
    print(f"\n[0] {len(all_lms_t)} landmarks loaded")

    print("[1] Loading Qwen3-VL-2B...")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct",torch_dtype=torch.bfloat16,
        device_map="cuda",attn_implementation="eager",
        local_files_only=True,trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct",local_files_only=True,trust_remote_code=True)
    lm = model.model.language_model

    print("[2] Selecting images...")
    with open(LIST_FILE) as f:
        all_names = [l.strip() for l in f if l.strip()]
    step = max(1, len(all_names)//N_IMAGES)
    idxs = list(range(0, len(all_names), step))[:N_IMAGES]
    img_names = [all_names[i] for i in idxs]
    lms_sel = all_lms_t[idxs]

    print("[3] Extracting hidden states...")
    hs_cache = {gk: [] for gk in GROUP_NAMES}
    t_start = time.time()
    for img_idx, img_name in enumerate(img_names):
        et = time.time()-t_start
        print(f"   [{img_idx+1}/{N_IMAGES}] {img_name[:30]} {et:.0f}s", end="\r")
        p = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(p): continue
        img = Image.open(p).convert("RGB")
        msg = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"describe this face"}]}]
        prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
        ids = inp["input_ids"][0]
        ve = (ids==151653).nonzero(as_tuple=True)[0].item()
        vs = 4
        hs_raw = {}
        def mk(gk, l):
            def hk(m,i,o):
                if gk not in hs_raw: hs_raw[gk]=[]
                hs_raw[gk].append(o[0,vs:ve].detach())
            return hk
        handles = []
        for gk in GROUP_NAMES:
            for l in LAYER_GROUPS[gk]:
                handles.append(lm.layers[l].register_forward_hook(mk(gk,l)))
        with torch.no_grad(): model(**inp)
        for h in handles: h.remove()
        for gk in GROUP_NAMES:
            if gk in hs_raw and hs_raw[gk]:
                hs_cache[gk].append(torch.stack(hs_raw[gk]).mean(dim=0).cpu())
        del hs_raw, inp; torch.cuda.empty_cache(); gc.collect()
    print()
    hs_stacked = {gk: torch.stack(hs_cache[gk]) for gk in GROUP_NAMES if hs_cache[gk]}
    N_total = hs_stacked[GROUP_NAMES[0]].shape[0]

    # Baselines
    print(f"\n   Baselines...")
    base_results = {}
    for seed in range(N_SEEDS):
        torch.manual_seed(42+seed)
        perm = torch.randperm(N_total)
        test_idx = perm[:N_TEST]; pool_idx = perm[N_TEST:]
        base_results[f"seed_{seed}"] = round(compute_baseline(lms_sel[pool_idx].numpy(), lms_sel[test_idx].numpy()), 1)
    print(f"   Per-landmark baseline: {np.mean(list(base_results.values())):.1f}±{np.std(list(base_results.values())):.1f}%")

    for probe_name in ["PoolMLP", "PoolLinear"]:
        print(f"\n   === {probe_name} (N_EPOCHS={N_EPOCHS}) ===")
        ProbeClass = {"PoolMLP": PoolMLPProbe, "PoolLinear": PoolLinearProbe}[probe_name]
        results = {}
        for gk in GROUP_NAMES:
            if gk not in hs_stacked: continue
            hs = hs_stacked[gk]; in_dim = hs.shape[-1]; N = hs.shape[0]
            results[gk] = {}
            for frac in TRAIN_FRACTIONS:
                seed_pcks = []
                for seed in range(N_SEEDS):
                    torch.manual_seed(42+seed)
                    perm = torch.randperm(N)
                    test_idx = perm[:N_TEST]; pool_idx = perm[N_TEST:]
                    n_train = min(max(2, int(len(pool_idx)*frac)), len(pool_idx))
                    train_shuffle = torch.randperm(len(pool_idx))
                    train_idx = pool_idx[train_shuffle[:n_train]]
                    train_hs = hs[train_idx].cuda().float()
                    test_hs = hs[test_idx].cuda().float()
                    train_gt = lms_sel[train_idx].cuda().float()
                    test_gt = lms_sel[test_idx].cuda().float()
                    feat_mean = train_hs.mean(dim=(0,1), keepdim=True)
                    feat_std = train_hs.std(dim=(0,1), keepdim=True)+1e-8
                    train_hs = (train_hs-feat_mean)/feat_std
                    test_hs = (test_hs-feat_mean)/feat_std
                    probe = ProbeClass(in_dim).cuda()
                    opt = torch.optim.Adam(probe.parameters(), lr=LR)
                    for ep in range(N_EPOCHS):
                        probe.train()
                        pred = probe(train_hs)
                        loss = torch.nn.functional.mse_loss(pred, train_gt)
                        opt.zero_grad(); loss.backward(); opt.step()
                    probe.eval()
                    with torch.no_grad():
                        test_pck = pck(probe(test_hs), test_gt, 0.05)
                    seed_pcks.append(test_pck)
                    del train_hs, test_hs, train_gt, test_gt, probe, opt
                    torch.cuda.empty_cache()
                mean_pck = round(float(np.mean(seed_pcks)), 1)
                std_pck = round(float(np.std(seed_pcks)), 1)
                results[gk][f"frac_{frac}"] = {"pck_mean": mean_pck, "pck_std": std_pck, "pck_per_seed": seed_pcks}
                print(f"      {gk} {frac*100:.0f}%: {mean_pck}±{std_pck}%")
        path = os.path.join(OUT_DIR, f"{probe_name.lower()}_500epoch_results.json")
        with open(path, "w") as f:
            json.dump({"config":{"n_epochs":N_EPOCHS,"n_seeds":N_SEEDS},"results":results}, f, indent=2)
        print(f"   Saved: {path}")

    # Save baseline
    with open(os.path.join(OUT_DIR, "pool_baseline.json"), "w") as f:
        json.dump({"mean":round(np.mean(list(base_results.values())),1),"std":round(np.std(list(base_results.values())),1),"per_seed":base_results}, f, indent=2)

    del model, lm, processor; gc.collect(); torch.cuda.empty_cache()
    print("\nDONE.")

if __name__=="__main__":
    main()
