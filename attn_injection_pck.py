"""
Attention Weight Injection - PCK验证
Captures L18 attention weights, injects into L28,
trains ConvNet probe on injected hidden states, compares PCK.
"""

import os, sys, gc, torch, json, math, time
import numpy as np
from PIL import Image
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    eager_attention_forward, apply_rotary_pos_emb, repeat_kv
)

BASE = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
IMG_DIR = os.path.join(BASE, "AFLW2000-3D_crop")
LIST_FILE = os.path.join(BASE, "AFLW2000-3D_crop.list")
EVAL_DIR = os.path.join(BASE, "eval")
OUT_DIR = "/root/Qwen3-VL/paper/figures/fixed_test"
os.makedirs(OUT_DIR, exist_ok=True)

N_IMAGES = 200
CAPTURE = 17  # L18
INJECT = 27   # L28
SEED = 42


def load_landmarks():
    pts = np.load(os.path.join(EVAL_DIR, "AFLW2000-3D.pts68.npy"))
    roi = np.load(os.path.join(EVAL_DIR, "AFLW2000-3D_crop.roi_box.npy"))
    lms = []
    for i in range(len(pts)):
        x1, y1, x2, y2 = roi[i]
        w, h = x2 - x1, y2 - y1
        lx = (pts[i, 0] - x1) * 32 / w
        ly = (pts[i, 1] - y1) * 32 / h
        lms.append(np.stack([lx, ly], axis=-1))
    return np.array(lms)


def load_split():
    with open(LIST_FILE) as f:
        names = [l.strip() for l in f if l.strip()]
    np.random.seed(SEED)
    idx = np.random.permutation(len(names))
    return names, idx[40:], idx[:40]


class ConvNetProbe(torch.nn.Module):
    def __init__(self, in_dim=2048, grid_sz=8):
        super().__init__()
        self.grid_sz = grid_sz
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, 256, 3, padding=1),
            torch.nn.ReLU(True), torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 128, 3, padding=1),
            torch.nn.ReLU(True), torch.nn.BatchNorm2d(128),
            torch.nn.Flatten(),
            torch.nn.Linear(grid_sz*grid_sz*128, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 136),
        )
    def forward(self, x):
        x = x.view(x.shape[0], self.grid_sz, self.grid_sz, -1).permute(0, 3, 1, 2)
        return self.net(x).view(x.shape[0], 68, 2)


def pck(pred, target, thresh=0.05):
    d = torch.norm(pred - target, dim=-1)
    return (d < thresh * math.sqrt(32**2+32**2)).float().mean().item()


def capture_attn(model, lm, inp, layer):
    """Returns full attn_weights [1,16,seq,seq] from layer."""
    saved = {}
    orig = lm.layers[layer].self_attn.forward
    def cap(hidden_states, position_embeddings, attention_mask=None, **kw):
        a = lm.layers[layer].self_attn
        sh = (*hidden_states.shape[:-1], -1, a.head_dim)
        q = a.q_norm(a.q_proj(hidden_states).view(sh)).transpose(1, 2)
        k = a.k_norm(a.k_proj(hidden_states).view(sh)).transpose(1, 2)
        v = a.v_proj(hidden_states).view(sh).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        _, w = eager_attention_forward(a, q, k, v, attention_mask, dropout=0.0, scaling=a.scaling)
        saved["w"] = w.detach().cpu()
        # Also compute real output to keep model state valid
        v_state = a.v_proj(hidden_states).view(sh).transpose(1, 2)
        v_state = repeat_kv(v_state, a.num_key_value_groups)
        real_out = torch.matmul(w.to(q.device, q.dtype), v_state)
        real_out = real_out.transpose(1, 2).contiguous().reshape(*hidden_states.shape[:-1], -1)
        real_out = a.o_proj(real_out)
        return real_out, w
    lm.layers[layer].self_attn.forward = cap
    with torch.no_grad(): model(**inp)
    lm.layers[layer].self_attn.forward = orig
    gc.collect(); torch.cuda.empty_cache()
    return saved.get("w")


def extract_hs(model, lm, inp, vs, ve, inject_w=None):
    """Forward with optional L28 attention injection. Returns image token hidden states [nvt, 2048]."""
    orig = lm.layers[INJECT].self_attn.forward
    if inject_w is not None:
        iw = inject_w
        def inj(hidden_states, position_embeddings, attention_mask=None, **kw):
            a = lm.layers[INJECT].self_attn
            sh = (*hidden_states.shape[:-1], -1, a.head_dim)
            q = a.q_norm(a.q_proj(hidden_states).view(sh)).transpose(1, 2)
            k = a.k_norm(a.k_proj(hidden_states).view(sh)).transpose(1, 2)
            v = a.v_proj(hidden_states).view(sh).transpose(1, 2)
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = repeat_kv(k, a.num_key_value_groups)
            v = repeat_kv(v, a.num_key_value_groups)
            out = torch.matmul(iw.to(q.device, q.dtype), v)
            out = out.transpose(1, 2).contiguous().reshape(*hidden_states.shape[:-1], -1)
            out = a.o_proj(out)
            return out, iw.to(q.device, q.dtype)
        lm.layers[INJECT].self_attn.forward = inj

    hs_list = []
    h = lm.layers[INJECT].register_forward_hook(lambda m,i,o: hs_list.append(o.detach().cpu()))
    with torch.no_grad(): model(**inp)
    h.remove()
    lm.layers[INJECT].self_attn.forward = orig
    gc.collect(); torch.cuda.empty_cache()
    if hs_list:
        return hs_list[0][0, vs:ve, :].float()
    return None


def run_split(name, indices, lm_data, model, processor, lm_model, hs_l28_all, hs_inj_all):
    for count, idx in enumerate(indices):
        img_name = load_names()[0][idx]  # need to return names
        p = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("RGB")
        msg = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"describe this face"}]}]
        prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
        ids = inp["input_ids"][0]
        ve = (ids==151653).nonzero(as_tuple=True)[0].item()
        vs = 4

        w18 = capture_attn(model, lm_model, inp, CAPTURE)
        h28 = extract_hs(model, lm_model, inp, vs, ve, inject_w=None)
        h_inj = extract_hs(model, lm_model, inp, vs, ve, inject_w=w18) if w18 is not None else None

        del inp; gc.collect(); torch.cuda.empty_cache()
        if h28 is not None: hs_l28_all.append(h28)
        if h_inj is not None: hs_inj_all.append(h_inj)

        if (count+1) % 20 == 0:
            print(f"   {name} [{count+1}/{len(indices)}]", flush=True)


def main():
    print("="*60)
    print("ATTN INJECTION PCK")
    print("="*60)

    print("\n[1] Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="eager",
        local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
    lm_model = model.model.language_model

    print("\n[2] Loading data...")
    with open(LIST_FILE) as f:
        all_names = [l.strip() for l in f if l.strip()]
    # Sample N_IMAGES and corresponding landmarks
    step = max(1, len(all_names) // N_IMAGES)
    sample_idx = list(range(0, len(all_names), step))[:N_IMAGES]
    all_names = [all_names[i] for i in sample_idx]
    landmarks = load_landmarks()[sample_idx]
    np.random.seed(SEED)
    perm = np.random.permutation(N_IMAGES)
    train_idx, test_idx = perm[40:], perm[:40]
    lm_train = torch.tensor(landmarks[train_idx]).float()
    lm_test = torch.tensor(landmarks[test_idx]).float()
    print(f"   {len(all_names)} images, {len(train_idx)} train, {len(test_idx)} test")

    print("\n[3] Extracting hidden states...")
    t0 = time.time()

    def extract_set(indices, label):
        hs28, hsinj = [], []
        for cnt, ii in enumerate(indices):
            p = os.path.join(IMG_DIR, all_names[ii])
            if not os.path.exists(p): continue
            img = Image.open(p).convert("RGB")
            msg = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"describe this face"}]}]
            prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inp = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
            ids = inp["input_ids"][0]
            ve = (ids==151653).nonzero(as_tuple=True)[0].item()
            vs = 4

            w18 = capture_attn(model, lm_model, inp, CAPTURE)
            h28 = extract_hs(model, lm_model, inp, vs, ve)
            h_inj = extract_hs(model, lm_model, inp, vs, ve, inject_w=w18) if w18 is not None else None

            del inp; gc.collect(); torch.cuda.empty_cache()
            if h28 is not None: hs28.append(h28)
            if h_inj is not None: hsinj.append(h_inj)

            if (cnt+1) % 20 == 0 or cnt == 0:
                print(f"   {label} [{cnt+1}/{len(indices)}]  elap={time.time()-t0:.0f}s", flush=True)

        return torch.stack(hs28) if hs28 else None, torch.stack(hsinj) if hsinj else None

    tr28, tr_inj = extract_set(train_idx, "train")
    te28, te_inj = extract_set(test_idx, "test")

    print(f"\n   Train: L28={tr28.shape if tr28 is not None else 'None'}, Inj={tr_inj.shape if tr_inj is not None else 'None'}")
    print(f"   Test:  L28={te28.shape if te28 is not None else 'None'}, Inj={te_inj.shape if te_inj is not None else 'None'}")

    print("\n[4] Training probes...")
    results = []
    for name, tr, te in [("L28_baseline", tr28, te28), ("L28_injected", tr_inj, te_inj)]:
        if tr is None or te is None:
            print(f"   {name}: SKIP")
            continue
        mean = tr.mean(dim=(0,1), keepdim=True)
        std = tr.std(dim=(0,1), keepdim=True) + 1e-8
        tr_n = (tr - mean) / std
        te_n = (te - mean) / std
        n_train = len(tr_n)

        row = {"name": name}
        for frac in [0.1, 0.25, 0.5, 1.0]:
            n_use = max(1, int(n_train * frac))
            ridx = np.random.RandomState(SEED).permutation(n_train)[:n_use]
            probe = ConvNetProbe().cuda()
            opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
            xt = tr_n[ridx].cuda()
            yt = lm_train[ridx].cuda()
            xte = te_n.cuda()
            yte = lm_test.cuda()

            for ep in range(100):
                opt.zero_grad()
                F.mse_loss(probe(xt), yt).backward()
                opt.step()

            with torch.no_grad():
                pc = pck(probe(xte), yte)
            pct = round(pc*100, 1)
            row[f"{int(frac*100)}%"] = pct
            print(f"   {name} {frac*100:3.0f}%: {pct}%")

        results.append(row)

    print(f"\nTime: {time.time()-t0:.0f}s")

    import json
    with open(os.path.join(OUT_DIR, "attn_injection_pck_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: attn_injection_pck_results.json")


if __name__ == "__main__":
    import torch.nn.functional as F
    main()
