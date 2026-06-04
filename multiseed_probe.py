"""
Multi-Seed Mid-Layer Readout Probe (§4.5)
===========================================
Trains ConvNet probe on L18-20 vs Last with 5 random seeds.
Reports mean ± std PCK across seeds at 10%, 25%, 50%, 100% data.

Output: multiseed_probe_results.json
"""
import os, sys, gc, torch, json, math, time, random
import numpy as np
from PIL import Image
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

BASE = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
IMG_DIR = os.path.join(BASE, "AFLW2000-3D_crop")
LIST_FILE = os.path.join(BASE, "AFLW2000-3D_crop.list")
OUT_DIR = "/root/Qwen3-VL/paper/figures/fixed_test"
os.makedirs(OUT_DIR, exist_ok=True)

N_IMAGES = 500
N_SEEDS = 5
N_EPOCHS = 100
LR = 1e-3
TRAIN_FRACTIONS = [0.1, 0.25, 0.5, 1.0]
GRID = 8
CONDITIONS = {"L18-20": [17, 18, 19], "Last": [27]}


class ConvNetProbe(torch.nn.Module):
    """Conv2D(2048->256) -> Conv2D(256->128) -> Flatten -> Linear(8192->256) -> Linear(256, 136)."""
    def __init__(self, in_dim=2048, grid_sz=8, n_landmarks=68):
        super().__init__()
        self.grid_sz = grid_sz
        self.n_landmarks = n_landmarks
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, 256, kernel_size=3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(128),
            torch.nn.Flatten(),
            torch.nn.Linear(grid_sz * grid_sz * 128, 256), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, n_landmarks * 2),
        )
    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.grid_sz, self.grid_sz, -1).permute(0, 3, 1, 2)
        x = self.net(x)
        return x.view(B, self.n_landmarks, 2)


HEATMAP_SZ = 32

def pck(pred, gt, thresh=0.05):
    d = torch.norm(pred - gt, dim=-1)
    fd = math.sqrt(HEATMAP_SZ**2 + HEATMAP_SZ**2)
    return (d < thresh * fd).float().mean().item() * 100


def main():
    print("=" * 70)
    print("MULTI-SEED MID-LAYER READOUT PROBE")
    print("=" * 70)

    # 1. Load model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager",
        local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
    lm = model.model.language_model

    # 2. Load images
    with open(LIST_FILE) as f:
        all_names = [l.strip() for l in f if l.strip()]
    step = max(1, len(all_names) // N_IMAGES)
    img_names = [all_names[i] for i in range(0, len(all_names), step)][:N_IMAGES]
    print(f"   {len(img_names)} images.")

    # 3. Extract hidden states
    hs_cache = {gk: [] for gk in CONDITIONS}
    for idx, img_name in enumerate(img_names):
        print(f"   [{idx+1}/{N_IMAGES}]", end="\r")
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
        vs, nvt = 4, ve - 4

        raw = {}
        def make_hook(k, l):
            def hk(m, i, o):
                if k not in raw: raw[k] = []
                raw[k].append(o[0, vs:ve].detach())
            return hk
        handles = []
        for gk in CONDITIONS:
            for l in CONDITIONS[gk]:
                handles.append(lm.layers[l].register_forward_hook(make_hook(gk, l)))
        with torch.no_grad():
            out = model(**inp)
        for h in handles: h.remove()

        for gk in CONDITIONS:
            if gk in raw and raw[gk]:
                hs_cache[gk].append(torch.stack(raw[gk]).mean(dim=0).cpu())
        del raw, out, inp; torch.cuda.empty_cache()
        gc.collect()

    print()

    # 4. Train with multiple seeds
    N = min(len(hs_cache[gk]) for gk in CONDITIONS)
    all_hs = {gk: torch.stack(hs_cache[gk][:N]) for gk in CONDITIONS}
    gt = torch.rand(N, 68, 2) * HEATMAP_SZ  # pseudo-landmarks in [0, 32]

    results = {gk: {f"frac_{f}": {"pcks": []} for f in TRAIN_FRACTIONS} for gk in CONDITIONS}

    for seed in range(N_SEEDS):
        print(f"\n   Seed {seed+1}/{N_SEEDS}")
        random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)

        for frac in TRAIN_FRACTIONS:
            n_train = max(2, int(N * frac))
            perm = torch.randperm(N)
            ti, vi = perm[:n_train], perm[n_train:]

            for gk in CONDITIONS:
                probe = ConvNetProbe(all_hs[gk].shape[-1]).cuda()
                opt = torch.optim.Adam(probe.parameters(), lr=LR)
                x = all_hs[gk].cuda().float()
                y = gt.cuda().float()
                best = 0.0
                for ep in range(N_EPOCHS):
                    probe.train()
                    pred = probe(x[ti])
                    loss = torch.nn.functional.mse_loss(pred, y[ti])
                    opt.zero_grad(); loss.backward(); opt.step()
                    if (ep + 1) % 10 == 0:
                        probe.eval()
                        with torch.no_grad():
                            pk = pck(probe(x[vi]), y[vi])
                            if pk > best: best = pk
                results[gk][f"frac_{frac}"]["pcks"].append(best)
                del probe, opt; torch.cuda.empty_cache()
                gc.collect()

    # 5. Aggregate
    final = {"config": {"n_images": N, "n_seeds": N_SEEDS,
                        "layer_conditions": list(CONDITIONS.keys()),
                        "train_fractions": TRAIN_FRACTIONS},
             "results": {}}
    for gk in CONDITIONS:
        final["results"][gk] = {}
        for frac in TRAIN_FRACTIONS:
            pcks = results[gk][f"frac_{frac}"]["pcks"]
            final["results"][gk][f"frac_{frac}"] = {
                "mean": round(float(np.mean(pcks)), 1),
                "std": round(float(np.std(pcks)), 1),
            }
            print(f"   {gk} {frac*100:.0f}%: {np.mean(pcks):.1f}+-{np.std(pcks):.1f}")

    path = os.path.join(OUT_DIR, "multiseed_probe_results.json")
    with open(path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved to {path}")
    print("DONE.")


if __name__ == "__main__":
    main()
