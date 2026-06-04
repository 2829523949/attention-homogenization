"""
Mid-layer Readout Probe for Landmark Localization (AFLW2000)
v2: Direct coordinate regression (instead of heatmaps)
Architecture:
  [B, 64, 2048] -> reshape [B, 8, 8, 2048]
  -> 3x3 Conv(2048->256, padding=1) + ReLU  (spatial mixing)
  -> 3x3 Conv(256->128, padding=1) + ReLU
  -> Flatten -> Linear(8*8*128, 256) -> ReLU -> Linear(256, 68*2) -> [B, 136]
  -> Reshape [B, 68, 2]   (x, y in heatmap space [0, 32])

Loss: MSE on normalized (x, y) coordinates
Layer conditions: L18-20 (avg), L25-27 (avg), L28 (last)
Training fractions: 10%, 25%, 50%, 100%
"""

import os, sys, gc, math, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/root/Qwen3-VL')
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

BASE = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
IMG_DIR = os.path.join(BASE, "AFLW2000-3D_crop")
EVAL_DIR = os.path.join(BASE, "eval")
LIST_FILE = os.path.join(BASE, "AFLW2000-3D_crop.list")
OUT_DIR = "/root/Qwen3-VL/paper/figures/fixed_test"
os.makedirs(OUT_DIR, exist_ok=True)

N_IMAGES = 500
TRAIN_RATIO = 0.8
TRAIN_FRACTIONS = [0.1, 0.25, 0.5, 1.0]
LAYER_CONDITIONS = [
    ([17, 18, 19], "L18-20"),
    ([24, 25, 26], "L25-27"),
    ([27], "Last"),
]
HIDDEN_DIM = 2048
GRID_SZ = 8
HEATMAP_SZ = 32    # Normalization space for landmark coordinates
N_LANDMARKS = 68
BATCH_SIZE = 32
LR = 5e-4
EPOCHS = 100
PATIENCE = 15
DEVICE = "cuda"

QUERY_LANDMARKS = {
    "nose":     [30],
    "eyes":     list(range(36, 48)),
    "mouth":    list(range(48, 68)),
    "chin":     [8],
    "forehead": list(range(17, 27)),
    "beard":    [7, 8, 9],
}

def get_landmark_positions(pts, roi_box, img_idx):
    """Return [68, 2] landmark positions in heatmap space [0, 32]."""
    x_orig = pts[img_idx, 0, :]
    y_orig = pts[img_idx, 1, :]
    left, top, right, bottom = roi_box[img_idx]
    crop_w = right - left
    crop_h = bottom - top
    if crop_w <= 0 or crop_h <= 0:
        return np.zeros((N_LANDMARKS, 2), dtype=np.float32)
    x_crop = (x_orig - left) * 120.0 / crop_w
    y_crop = (y_orig - top) * 120.0 / crop_h
    x_heat = np.clip(x_crop * HEATMAP_SZ / 120.0, 0, HEATMAP_SZ - 1)
    y_heat = np.clip(y_crop * HEATMAP_SZ / 120.0, 0, HEATMAP_SZ - 1)
    return np.stack([x_heat, y_heat], axis=-1).astype(np.float32)

class FeatureDataset(Dataset):
    def __init__(self, features, coords):
        self.features = features.cpu()
        self.coords = coords.cpu()
    def __len__(self):
        return self.features.size(0)
    def __getitem__(self, idx):
        return self.features[idx], self.coords[idx]

class CoordProbe(nn.Module):
    """Predict [68, 2] landmark coordinates from [64, 2048] hidden states."""
    def __init__(self, in_dim=HIDDEN_DIM, grid_sz=GRID_SZ, n_landmarks=N_LANDMARKS):
        super().__init__()
        self.grid_sz = grid_sz
        self.n_landmarks = n_landmarks
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(grid_sz * grid_sz * 128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_landmarks * 2),
        )
    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.grid_sz, self.grid_sz, -1).permute(0, 3, 1, 2)
        x = self.net(x)  # [B, 136]
        return x.view(B, self.n_landmarks, 2)  # [B, 68, 2] (x, y)

def compute_pck(pred_coords, gt_coords, thresholds=[0.03, 0.05, 0.1, 0.2]):
    """
    pred_coords: [B, 68, 2] (x, y) in [0, 32]
    gt_coords: [B, 68, 2]
    """
    dist = torch.sqrt((pred_coords[:, :, 0] - gt_coords[:, :, 0])**2 +
                      (pred_coords[:, :, 1] - gt_coords[:, :, 1])**2)
    face_diag = math.sqrt(HEATMAP_SZ**2 + HEATMAP_SZ**2)
    results = {}
    for thresh in thresholds:
        correct = (dist < thresh * face_diag).float().mean().item()
        results[thresh] = round(100 * correct, 1)
    return results, dist

def extract_features():
    """Extract Qwen hidden states and landmark coords from AFLW2000."""
    print("=" * 60)
    print("STAGE 1: Extracting hidden states from Qwen")
    print("=" * 60)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        '/root/Qwen3-VL-2B-Instruct', torch_dtype=torch.bfloat16,
        device_map='cuda', attn_implementation='eager',
        local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        '/root/Qwen3-VL-2B-Instruct', local_files_only=True, trust_remote_code=True)
    lm = model.model.language_model

    with open(LIST_FILE) as f:
        all_names = [l.strip() for l in f if l.strip()]
    # Uniformly sample across full pose range (AFLW2000 sorted by yaw)
    step = max(1, len(all_names) // N_IMAGES)
    sample_indices = list(range(0, len(all_names), step))[:N_IMAGES]
    img_names = [all_names[i] for i in sample_indices]
    pts68 = np.load(os.path.join(EVAL_DIR, "AFLW2000-3D.pts68.npy"))
    roi = np.load(os.path.join(EVAL_DIR, "AFLW2000-3D_crop.roi_box.npy"))

    features = {name: [] for _, name in LAYER_CONDITIONS}
    coords_list = []
    t_start = time.time()

    for img_idx, img_name in enumerate(img_names):
        if img_idx % 50 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / max(1, img_idx) * (N_IMAGES - img_idx) if img_idx > 0 else 0
            print(f"  [{img_idx}/{N_IMAGES}] elapsed={elapsed:.0f}s eta={eta:.0f}s")
        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        msg = [{"role":"user","content":[{"type":"image","image":img},
                {"type":"text","text":"describe this image"}]}]
        prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = processor(text=prompt, images=img, return_tensors='pt').to('cuda')
        ids = inp['input_ids'][0]
        ve = (ids == 151653).nonzero(as_tuple=True)[0].item()
        nvt = ve - 4

        collected = {}
        hooks = []
        target_layers = set()
        for layer_indices, _ in LAYER_CONDITIONS:
            for l in layer_indices:
                target_layers.add(l)
        for l in sorted(target_layers):
            def make_hook(lid):
                def hook(m, i, o):
                    out = o[0] if isinstance(o, tuple) else o
                    collected[lid] = out[:, 4:4+nvt].detach().cpu().float()
                return hook
            hooks.append(lm.layers[l].register_forward_hook(make_hook(l)))
        with torch.no_grad():
            model(**inp)
        for h in hooks:
            h.remove()

        for layer_indices, name in LAYER_CONDITIONS:
            if len(layer_indices) == 1:
                feat = collected[layer_indices[0]]
            else:
                feat = torch.stack([collected[l] for l in layer_indices]).mean(dim=0)
            features[name].append(feat.squeeze(0))

        coords = get_landmark_positions(pts68, roi, sample_indices[img_idx])
        coords_list.append(torch.from_numpy(coords))

        del collected, inp
        torch.cuda.empty_cache()

    feature_tensors = {}
    for name, lst in features.items():
        feature_tensors[name] = torch.stack(lst)
    coords = torch.stack(coords_list)
    del model, lm, processor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Extracted {len(img_names)} images. Time: {time.time()-t_start:.0f}s")
    return feature_tensors, coords, img_names

def train_probe(train_loader, val_loader, device=DEVICE):
    model = CoordProbe().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for feats, coords_gt in train_loader:
            feats, coords_gt = feats.to(device), coords_gt.to(device)
            pred = model(feats)
            loss = F.mse_loss(pred, coords_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * feats.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_pred = []
        all_gt = []
        with torch.no_grad():
            for feats, coords_gt in val_loader:
                feats, coords_gt = feats.to(device), coords_gt.to(device)
                pred = model(feats)
                loss = F.mse_loss(pred, coords_gt)
                val_loss += loss.item() * feats.size(0)
                all_pred.append(pred.cpu())
                all_gt.append(coords_gt.cpu())
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        all_pred = torch.cat(all_pred, dim=0)
        all_gt = torch.cat(all_gt, dim=0)
        pck_results, dist_epoch = compute_pck(all_pred, all_gt)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}: tr_l={train_loss:.6f} val_l={val_loss:.6f} PCK@0.05={pck_results[0.05]:.1f}% dist={dist_epoch.mean():.2f}px")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    # Restore best model and final eval
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    all_pred = []
    all_gt = []
    with torch.no_grad():
        for feats, coords_gt in val_loader:
            feats, coords_gt = feats.to(device), coords_gt.to(device)
            pred = model(feats)
            all_pred.append(pred.cpu())
            all_gt.append(coords_gt.cpu())
    all_pred = torch.cat(all_pred, dim=0)
    all_gt = torch.cat(all_gt, dim=0)

    pck_results, dist_full = compute_pck(all_pred, all_gt)

    query_pck = {}
    for qname, l_indices in QUERY_LANDMARKS.items():
        li = torch.tensor(l_indices)
        dist = torch.sqrt((all_pred[:, li, 0] - all_gt[:, li, 0])**2 +
                          (all_pred[:, li, 1] - all_gt[:, li, 1])**2)
        face_diag = math.sqrt(HEATMAP_SZ**2 + HEATMAP_SZ**2)
        pck_02 = (dist < 0.2 * face_diag).float().mean().item()
        query_pck[qname] = round(100 * pck_02, 1)

    # Mean/median distance (full 68 landmarks)
    mean_dist = dist_full.mean().item()
    median_dist = dist_full.median().item()
    
    return {
        "best_val_loss": round(best_val_loss, 6),
        "best_epoch": epoch + 1,
        "pck": {str(k): v for k, v in pck_results.items()},
        "mean_distance_heatmap_px": round(mean_dist, 2),
        "median_distance_heatmap_px": round(median_dist, 2),
        "query_pck": query_pck,
    }

def main():
    print("=" * 60)
    print("MID-LAYER READOUT PROBE v2 (Coordinate Regression)")
    print("=" * 60)
    feature_tensors, coords_all, _ = extract_features()
    N = feature_tensors[LAYER_CONDITIONS[0][1]].size(0)
    indices = np.random.RandomState(42).permutation(N)
    n_train = int(N * TRAIN_RATIO)
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    print(f"\nTrain: {n_train}, Val: {N - n_train}")

    print("\n" + "=" * 60)
    print("STAGE 2: Training probes")
    print("=" * 60)
    all_results = {}

    for layer_indices, name in LAYER_CONDITIONS:
        print(f"\n--- Layer: {name} ---")
        features = feature_tensors[name]
        layer_results = {}
        for frac in TRAIN_FRACTIONS:
            n_frac = max(1, int(n_train * frac))
            frac_train_idx = train_idx[:n_frac]

            # Normalize features: z-score using training set stats
            feats_train = features[frac_train_idx].float().clone()
            feats_val = features[val_idx].float().clone()
            # Compute mean/std across (N, spatial) dims, per channel
            feat_mean = feats_train.mean(dim=(0, 1), keepdim=True)  # [1, 1, 2048]
            feat_std = feats_train.std(dim=(0, 1), keepdim=True) + 1e-8
            feats_train = (feats_train - feat_mean) / feat_std
            feats_val = (feats_val - feat_mean) / feat_std

            train_ds = FeatureDataset(feats_train, coords_all[frac_train_idx])
            val_ds = FeatureDataset(feats_val, coords_all[val_idx])
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

            print(f"\n  Train frac={frac*100:.0f}% ({n_frac} imgs)")
            result = train_probe(train_loader, val_loader)
            layer_results[f"frac_{frac}"] = result
            print(f"  -> PCK@0.05={result['pck']['0.05']}% dist={result['mean_distance_heatmap_px']}px")
        all_results[name] = layer_results

    print("\n" + "=" * 80)
    print("SUMMARY: PCK@0.05 (strict) x Sample Efficiency")
    print("=" * 80)
    header = f"{'Layer':<10s}" + "".join([f"{f'Frac {int(f*100)}%':>14s}" for f in TRAIN_FRACTIONS])
    print(header)
    print("-" * len(header))
    for name in [n for _, n in LAYER_CONDITIONS]:
        row = f"{name:<10s}"
        for frac in TRAIN_FRACTIONS:
            pck = all_results[name][f"frac_{frac}"]["pck"]["0.05"]
            row += f"{pck:>13.1f}%"
        print(row)
    
    print("\n" + "=" * 80)
    print("SUMMARY: Mean Distance (heatmap pixels) x Sample Efficiency")
    print("=" * 80)
    header = f"{'Layer':<10s}" + "".join([f"{f'Frac {int(f*100)}%':>14s}" for f in TRAIN_FRACTIONS])
    print(header)
    print("-" * len(header))
    for name in [n for _, n in LAYER_CONDITIONS]:
        row = f"{name:<10s}"
        for frac in TRAIN_FRACTIONS:
            d = all_results[name][f"frac_{frac}"]["mean_distance_heatmap_px"]
            row += f"{d:>13.2f}"
        print(row)

    print("\n" + "-" * 70)
    print("PER-QUERY PCK@0.2 (100% training)")
    print("-" * 70)
    q_header = f"{'Query':<12s}" + "".join([f"{n:>12s}" for _, n in LAYER_CONDITIONS])
    print(q_header)
    for qname in QUERY_LANDMARKS.keys():
        row = f"{qname:<12s}"
        for name in [n for _, n in LAYER_CONDITIONS]:
            val = all_results[name]["frac_1.0"]["query_pck"].get(qname, 0)
            row += f"{val:>11.1f}%"
        print(row)

    save_path = os.path.join(OUT_DIR, "midlayer_probe_results.json")
    save_data = {
        "config": {
            "n_images": N_IMAGES,
            "n_train": n_train,
            "n_val": N - n_train,
            "train_fractions": TRAIN_FRACTIONS,
            "layer_conditions": [n for _, n in LAYER_CONDITIONS],
        },
        "results": all_results,
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")
    print("\nDone!")

if __name__ == "__main__":
    main()
