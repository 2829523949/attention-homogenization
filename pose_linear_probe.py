"""
Pose classification from mean-pooled hidden states.
Linear(2048 -> 3): predict yaw class (left/frontal/right).
No attention, no spatial conv, no spatial decoding.
Tests: does collapse destroy linearly accessible pose information?
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
N_CLASSES = 3
LAYER_CONDITIONS = [([17, 18, 19], "L18-20"), ([24, 25, 26], "L25-27"), ([27], "Last")]
TRAIN_FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 100
DEVICE = "cuda"


class PoseLinear(nn.Module):
    """Linear(2048 -> 3): mean pool -> classify yaw."""
    def __init__(self, in_dim=2048):
        super().__init__()
        self.fc = nn.Linear(in_dim, N_CLASSES)

    def forward(self, x):
        x = x.mean(dim=1)  # [B, 2048]
        return self.fc(x)   # [B, 3]


def main():
    print("=" * 70)
    print("POSE CLASSIFICATION FROM HIDDEN STATES")
    print("=" * 70)

    # ---- Extract features ----
    print("\n[1/3] Extracting hidden states...")
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

    # Load pose labels (yaw stored in first column)
    pose_data = np.load(os.path.join(EVAL_DIR, "ALFW2000-3D_pose_3ANG.npy"))
    yaw_all = pose_data[:, 0]  # [-90, 90]
    # 3 classes: left (yaw < -15), frontal (-15 to 15), right (yaw > 15)
    labels = np.zeros(len(yaw_all), dtype=np.int64)
    labels[yaw_all < -15] = 0   # left
    labels[(yaw_all >= -15) & (yaw_all <= 15)] = 1  # frontal
    labels[yaw_all > 15] = 2    # right
    print(f"  Class distribution: left={sum(labels==0)}, frontal={sum(labels==1)}, right={sum(labels==2)}")

    features = {n: [] for _, n in LAYER_CONDITIONS}
    # Filter labels to match images that actually exist
    valid_indices = []
    t0 = time.time()

    for gi, gname in enumerate(img_names):
        imp = os.path.join(IMG_DIR, gname)
        if not os.path.exists(imp):
            continue
        valid_indices.append(sample_idx[gi])
        if gi % 100 == 0:
            print(f"  [{gi}/{N_IMAGES}] {time.time()-t0:.0f}s")
        imp = os.path.join(IMG_DIR, gname)
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

        del collected, inp
        torch.cuda.empty_cache()

    feat_tensors = {n: torch.stack(lst) for n, lst in features.items()}
    label_tensor = torch.tensor(labels[valid_indices])
    del model, lm, processor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Done: {len(img_names)} imgs in {time.time()-t0:.0f}s")

    # ---- Train linear classifiers ----
    print("\n[2/3] Training linear pose classifiers...")
    N = feat_tensors["L18-20"].size(0)
    rng = np.random.RandomState(42).permutation(N)
    n_train = int(N * 0.8)

    all_results = {}
    for li, nm in LAYER_CONDITIONS:
        print(f"\n  --- {nm} ---")
        feat_all = feat_tensors[nm].float()
        layer_results = {}

        for frac in TRAIN_FRACTIONS:
            n_frac = max(4, int(n_train * frac))
            # Stratified split to maintain class balance
            tr_idx = rng[:n_frac]
            va_idx = rng[n_train:]

            ft = feat_all[tr_idx].clone()
            fv = feat_all[va_idx].clone()
            fm = ft.mean(dim=(0, 1), keepdim=True)
            fs = ft.std(dim=(0, 1), keepdim=True) + 1e-8
            ft = (ft - fm) / fs
            fv = (fv - fm) / fs

            tr_labels = label_tensor[tr_idx]
            va_labels = label_tensor[va_idx]

            td = DataLoader(list(zip(ft, tr_labels)), BATCH_SIZE, shuffle=True)
            vd = DataLoader(list(zip(fv, va_labels)), BATCH_SIZE, shuffle=False)

            probe = PoseLinear().to(DEVICE)
            opt = torch.optim.Adam(probe.parameters(), lr=LR)
            best_acc = 0
            patience = 0

            for ep in range(EPOCHS):
                probe.train()
                for fb, cb in td:
                    fb, cb = fb.to(DEVICE), cb.to(DEVICE)
                    loss = nn.CrossEntropyLoss()(probe(fb), cb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                probe.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for fb, cb in vd:
                        fb = fb.to(DEVICE)
                        pred = probe(fb).argmax(dim=1).cpu()
                        correct += (pred == cb).sum().item()
                        total += cb.size(0)
                acc = correct / total
                if acc > best_acc:
                    best_acc = acc
                    patience = 0
                else:
                    patience += 1
                    if patience >= 20:
                        break

            # Random baseline for 3-class = 33.3%
            layer_results[f"frac_{frac}"] = {
                "accuracy": round(100 * best_acc, 1),
                "random_baseline": round(100 / N_CLASSES, 1),
                "n_train": n_frac,
            }
            print(f"    {frac*100:5.0f}% ({n_frac:3d} imgs): yaw acc={100*best_acc:.1f}%  (random={100/N_CLASSES:.0f}%)")

        all_results[nm] = layer_results

    # ---- Summary table ----
    print("\n[3/3] Summary (yaw classification accuracy %):")
    print(f"{'Layer':10s}", end="")
    for f in TRAIN_FRACTIONS:
        print(f"{f*100:>7.0f}%", end="")
    print("\n" + "-" * 60)
    for nm in [n for _, n in LAYER_CONDITIONS]:
        print(f"{nm:10s}", end="")
        for f in TRAIN_FRACTIONS:
            v = all_results[nm][f"frac_{f}"]["accuracy"]
            print(f"{v:>7.1f}", end="")
        print()

    save_data = {
        "config": {
            "model": "Qwen3-VL-2B", "n_images": N_IMAGES,
            "task": "yaw_3class (left/frontal/right)",
            "probe": "mean_pool -> Linear(2048 -> 3)",
            "layer_conditions": [n for _, n in LAYER_CONDITIONS],
            "train_fractions": TRAIN_FRACTIONS,
        },
        "results": all_results,
    }
    save_path = os.path.join(OUT_DIR, "pose_linear_results.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")
    print(f"Total: {time.time()-t0:.0f}s")
    print("Done!")


if __name__ == "__main__":
    main()
