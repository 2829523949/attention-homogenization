"""
Geometry Homogenization + Probe Decomposability (§4.7)
=======================================================
Rigorous version with:
  - Hold-out TEST set (40 images, never touched during training)
  - Multi-seed (3 seeds, mean+/-std)
  - Fixed training pool (160 images), fractions sampled from pool
  - Proper per-landmark baseline (train mean -> test)

Output: geometry_probe_results.json
"""
import os, sys, gc, torch, json, math, time
import numpy as np
from PIL import Image
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ============ CONFIG ============
BASE = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
IMG_DIR = os.path.join(BASE, "AFLW2000-3D_crop")
LIST_FILE = os.path.join(BASE, "AFLW2000-3D_crop.list")
EVAL_DIR = os.path.join(BASE, "eval")
OUT_DIR = "/root/Qwen3-VL/paper/figures/fixed_test"
os.makedirs(OUT_DIR, exist_ok=True)

N_IMAGES = 200
N_EPOCHS = 100
PROBE_EPOCHS = {"ConvNet": 100, "PoolMLP": 500, "PoolLinear": 500}
LR = 1e-3
TRAIN_FRACTIONS = [0.1, 0.25, 0.5, 1.0]
N_SEEDS = 3

CROP_SZ = 120
HEATMAP_SZ = 32
N_TEST = 40

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


def adjacent_token_sim(h):
    n, d = h.shape
    nd = int(math.isqrt(n))
    h_2d = h.view(nd, nd, d)
    right = torch.nn.functional.cosine_similarity(h_2d[:, :-1], h_2d[:, 1:], dim=-1).mean().item()
    down  = torch.nn.functional.cosine_similarity(h_2d[:-1, :], h_2d[1:, :], dim=-1).mean().item()
    return (right + down) / 2


def effective_rank(h):
    u, s, v = torch.svd(h.float())
    p = s / s.sum()
    H = -(p * torch.log(p + 1e-10)).sum().item()
    return math.exp(H)


def top3_energy(h):
    u, s, v = torch.svd(h.float())
    var = s ** 2
    return (var[:3].sum() / var.sum()).item()


class ConvNetProbe(torch.nn.Module):
    def __init__(self, in_dim=2048, grid_sz=8, n_landmarks=68):
        super().__init__()
        self.grid_sz = grid_sz
        self.n_landmarks = n_landmarks
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(128),
            torch.nn.Flatten(),
            torch.nn.Linear(grid_sz * grid_sz * 128, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, n_landmarks * 2),
        )
    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.grid_sz, self.grid_sz, -1).permute(0, 3, 1, 2)
        x = self.net(x)
        return x.view(B, self.n_landmarks, 2)


class PoolMLPProbe(torch.nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 68 * 2),
        )
    def forward(self, x):
        x = x.mean(dim=1)
        return self.net(x).view(-1, 68, 2)


class PoolLinearProbe(torch.nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        self.net = torch.nn.Linear(in_dim, 68 * 2)
    def forward(self, x):
        x = x.mean(dim=1)
        return self.net(x).view(-1, 68, 2)


def pck(pred, gt, threshold=0.05):
    dists = torch.norm(pred - gt, dim=-1)
    face_diag = math.sqrt(HEATMAP_SZ**2 + HEATMAP_SZ**2)
    return (dists < threshold * face_diag).float().mean().item() * 100


def compute_baseline(train_lms, test_lms):
    """Per-landmark mean from train -> PCK on test."""
    mean_per_lm = train_lms.mean(axis=0)
    face_diag = math.sqrt(HEATMAP_SZ**2 + HEATMAP_SZ**2)
    dists = np.sqrt(((test_lms - mean_per_lm[None, :, :])**2).sum(axis=-1))
    return (dists < 0.05 * face_diag).mean() * 100


def train_one_frac(probe, train_hs, train_gt, test_hs, test_gt, lr=LR, epochs=100):
    """Train for fixed epochs. Evaluate on test ONCE at the end. No test leakage."""
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    for ep in range(epochs):
        probe.train()
        pred = probe(train_hs)
        loss = torch.nn.functional.mse_loss(pred, train_gt)
        opt.zero_grad(); loss.backward(); opt.step()

    probe.eval()
    with torch.no_grad():
        test_pred = probe(test_hs)
        test_pck = pck(test_pred, test_gt, 0.05)

    return test_pck, epochs


def main():
    print("=" * 70)
    print("GEOMETRY + PROBE (RIGOROUS)")
    print("=" * 70)

    # ---- 0. Load landmarks ----
    print("\n[0] Loading AFLW2000 landmarks...")
    all_lms_t = torch.from_numpy(load_landmarks(EVAL_DIR)).float()
    print(f"   {len(all_lms_t)} landmark sets loaded.")

    # ---- 1. Load model ----
    print("\n[1] Loading Qwen3-VL-2B...")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager",
        local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
    lm = model.model.language_model
    print(f"   {len(lm.layers)} layers.")

    # ---- 2. Select 200 images ----
    print("\n[2] Selecting images...")
    with open(LIST_FILE) as f:
        all_names = [l.strip() for l in f if l.strip()]
    step = max(1, len(all_names) // N_IMAGES)
    idxs = list(range(0, len(all_names), step))[:N_IMAGES]
    img_names = [all_names[i] for i in idxs]
    lms_sel = all_lms_t[idxs]
    print(f"   {len(img_names)} images.")

    # ---- 3. Extract hidden states ----
    print("\n[3] Extracting hidden states...")
    hs_cache = {gk: [] for gk in GROUP_NAMES}
    geometry = {gk: {} for gk in GROUP_NAMES}

    t_start = time.time()
    for img_idx, img_name in enumerate(img_names):
        elapsed = time.time() - t_start
        print(f"   [{img_idx+1}/{N_IMAGES}] {img_name[:30]} {elapsed:.0f}s", end="\r")
        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")

        msg = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "describe this face"}
        ]}]
        prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
        ids = inp["input_ids"][0]
        ve = (ids == 151653).nonzero(as_tuple=True)[0].item()
        vs = 4

        hs_raw = {}
        def make_hook(gk, l):
            def hk(m, i, o):
                if gk not in hs_raw:
                    hs_raw[gk] = []
                hs_raw[gk].append(o[0, vs:ve].detach())
            return hk
        handles = []
        for gk in GROUP_NAMES:
            for l in LAYER_GROUPS[gk]:
                handles.append(lm.layers[l].register_forward_hook(make_hook(gk, l)))

        with torch.no_grad():
            model(**inp)

        for h in handles:
            h.remove()

        for gk in GROUP_NAMES:
            if gk in hs_raw and hs_raw[gk]:
                h_avg = torch.stack(hs_raw[gk]).mean(dim=0)
                hs_cache[gk].append(h_avg.cpu())
                adj_sim = adjacent_token_sim(h_avg)
                eff_rank = effective_rank(h_avg)
                t3e = top3_energy(h_avg)
                geometry[gk][img_name] = {"adj_sim": adj_sim, "eff_rank": eff_rank, "top3": t3e}

        del hs_raw, inp; torch.cuda.empty_cache()
        gc.collect()
    print()

    # Geometry summary
    geo_summary = {}
    for gk in GROUP_NAMES:
        adj = [geometry[gk][f]["adj_sim"] for f in geometry[gk]]
        eff = [geometry[gk][f]["eff_rank"] for f in geometry[gk]]
        t3 = [geometry[gk][f]["top3"] for f in geometry[gk]]
        geo_summary[gk] = {
            "adj_sim_mean": round(float(np.mean(adj)), 4),
            "adj_sim_std": round(float(np.std(adj)), 4),
            "eff_rank_mean": round(float(np.mean(eff)), 2),
            "eff_rank_std": round(float(np.std(eff)), 2),
            "top3_energy_ratio_mean": round(float(np.mean(t3)), 4),
            "top3_energy_ratio_std": round(float(np.std(t3)), 4),
        }
        print(f"   {gk}: adj_sim={geo_summary[gk]['adj_sim_mean']:.4f} "
              f"eff_rank={geo_summary[gk]['eff_rank_mean']:.1f} "
              f"top3={geo_summary[gk]['top3_energy_ratio_mean']:.4f}")

    hs_stacked = {gk: torch.stack(hs_cache[gk]) for gk in GROUP_NAMES if hs_cache[gk]}
    N_total = hs_stacked[GROUP_NAMES[0]].shape[0]
    print(f"\n   Total: {N_total} images, test={N_TEST}, pool={N_total-N_TEST}")

    # ---- 4. Multi-seed probe training ----
    print("\n[4] Multi-seed probe training...")

    probe_results = {}
    for probe_name in ["ConvNet", "PoolMLP", "PoolLinear"]:
        print(f"\n   === {probe_name} ===")
        ProbeClass = {"ConvNet": ConvNetProbe, "PoolMLP": PoolMLPProbe,
                      "PoolLinear": PoolLinearProbe}[probe_name]
        probe_results[probe_name] = {}

        for gk in GROUP_NAMES:
            if gk not in hs_stacked:
                continue
            hs = hs_stacked[gk]
            in_dim = hs.shape[-1]
            N = hs.shape[0]
            probe_results[probe_name][gk] = {}

            for frac in TRAIN_FRACTIONS:
                seed_pcks = []
                seed_epochs = []

                for seed in range(N_SEEDS):
                    torch.manual_seed(42 + seed)
                    perm = torch.randperm(N)
                    test_idx = perm[:N_TEST]
                    pool_idx = perm[N_TEST:]

                    n_train = max(2, int(len(pool_idx) * frac))
                    n_train = min(n_train, len(pool_idx))
                    train_shuffle = torch.randperm(len(pool_idx))
                    train_idx = pool_idx[train_shuffle[:n_train]]

                    train_hs = hs[train_idx].cuda().float()
                    test_hs = hs[test_idx].cuda().float()
                    train_gt = lms_sel[train_idx].cuda().float()
                    test_gt = lms_sel[test_idx].cuda().float()

                    # Z-score normalize hidden states (per-channel, across tokens)
                    feat_mean = train_hs.mean(dim=(0, 1), keepdim=True)  # [1, 1, 2048]
                    feat_std = train_hs.std(dim=(0, 1), keepdim=True) + 1e-8
                    train_hs = (train_hs - feat_mean) / feat_std
                    test_hs = (test_hs - feat_mean) / feat_std

                    probe = ProbeClass(in_dim).cuda()
                    epochs = PROBE_EPOCHS.get(probe_name, N_EPOCHS)
                    best_pck, best_ep = train_one_frac(
                        probe, train_hs, train_gt, test_hs, test_gt, epochs=epochs)

                    seed_pcks.append(best_pck)
                    seed_epochs.append(best_ep)

                    del train_hs, test_hs, train_gt, test_gt, probe
                    torch.cuda.empty_cache()

                mean_pck = round(float(np.mean(seed_pcks)), 1)
                std_pck = round(float(np.std(seed_pcks)), 1)
                probe_results[probe_name][gk][f"frac_{frac}"] = {
                    "pck_mean": mean_pck,
                    "pck_std": std_pck,
                    "pck_per_seed": [round(p, 1) for p in seed_pcks],
                    "best_epochs": seed_epochs,
                }
                print(f"      {gk} {frac*100:.0f}%: {mean_pck}+/-{std_pck}%")

    # ---- Baseline proper ----
    print(f"\n   Computing baselines...")
    baseline_results = {}
    for seed in range(N_SEEDS):
        torch.manual_seed(42 + seed)
        perm = torch.randperm(N_total)
        test_idx = perm[:N_TEST]
        pool_idx = perm[N_TEST:]
        base = compute_baseline(lms_sel[pool_idx].numpy(), lms_sel[test_idx].numpy())
        baseline_results[f"seed_{seed}"] = round(base, 1)

    baseline_mean = round(float(np.mean(list(baseline_results.values()))), 1)
    baseline_std = round(float(np.std(list(baseline_results.values()))), 1)
    print(f"      Per-landmark baseline: {baseline_mean}+/-{baseline_std}%")

    # ---- 5. Save ----
    print("\n[5] Saving...")
    output = {
        "config": {
            "n_images": N_IMAGES,
            "n_test": N_TEST,
            "train_fractions": TRAIN_FRACTIONS,
            "n_seeds": N_SEEDS,
            "n_epochs": N_EPOCHS,
            "lr": LR,
            "layer_groups": LAYER_GROUPS,
        },
        "geometry_homogenization": geo_summary,
        "probe_results": probe_results,
        "baseline_per_landmark_mean": {
            "mean": baseline_mean,
            "std": baseline_std,
            "per_seed": baseline_results,
        },
    }
    path = os.path.join(OUT_DIR, "geometry_probe_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"   Saved to {path}")
    print("DONE.")


if __name__ == "__main__":
    main()
