"""Layer-wise Embedding Sensitivity: Compare discriminability across layers.

Three VLM extraction methods + CLIP (pooler + raw CLS) for fair comparison.
"""
import os, sys, json, torch
import numpy as np
sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"; os.environ["HF_HUB_OFFLINE"] = "1"
from bridge import load_qwen

ANGLES = list(range(0, 91, 10))
PROMPTS = [f"A person looking to the right at {a} degrees" for a in ANGLES]
print(f"Testing {len(PROMPTS)} prompts")

LAYER_GROUPS = {
    "Early (L1-3)":  [0, 1, 2],
    "Mid (L8-10)":   [7, 8, 9],
    "Later (L18-20)":[17, 18, 19],
    "Last (L25-27)": [25, 26, 27],
}

print("Loading Qwen3-VL...")
qwen, processor = load_qwen()
qwen.eval()
tok = processor.tokenizer

SPECIAL_PREFIXES = ("<|im_", "<|end|>", "<|extra_")

def get_content_mask(input_ids):
    ids = input_ids.tolist()
    mask = []
    for tid in ids:
        t = tok.decode([tid])
        is_special = any(t.startswith(p) for p in SPECIAL_PREFIXES)
        mask.append(not is_special)
    return torch.tensor(mask, device="cuda")

def find_angle_token_positions(input_ids):
    ids = input_ids.tolist()
    deg_idx = None
    for i, tid in enumerate(ids):
        t = tok.decode([tid]).strip()
        if "degrees" in t.lower():
            deg_idx = i
            break
    angle_idxs = []
    if deg_idx is not None and deg_idx > 0:
        for j in range(deg_idx - 1, -1, -1):
            t = tok.decode([ids[j]]).strip()
            if any(c.isdigit() for c in t):
                angle_idxs.insert(0, j)
            else:
                break
    if angle_idxs:
        return angle_idxs
    for i, tid in enumerate(ids):
        t = tok.decode([tid]).strip()
        if any(c.isdigit() for c in t):
            return [i]
    return [-1]

# Verify
print("\n--- Verifying angle token positions ---")
for a in ANGLES[:3]:
    text = f"A person looking to the right at {a} degrees"
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    positions = find_angle_token_positions(inputs["input_ids"][0])
    decoded = [tok.decode([inputs["input_ids"][0, p].item()]).strip() for p in positions]
    print(f"  Angle {a:3d}: positions={positions}, tokens={decoded}, seq_len={inputs['input_ids'].shape[1]}")

# --- VLM hidden states ---
all_data = {"angle_token": [], "mean_pool": [], "last_token": []}

for a, text in zip(ANGLES, PROMPTS):
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to("cuda")

    angle_positions = find_angle_token_positions(inputs["input_ids"][0])
    angle_pos = angle_positions[-1]
    content_mask = get_content_mask(inputs["input_ids"][0])

    with torch.no_grad():
        out = qwen(**inputs, output_hidden_states=True)

    hs = torch.stack(list(out.hidden_states)).squeeze(1)

    if angle_pos >= 0:
        angle_hs = hs[:, angle_pos, :2048].cpu()
    else:
        angle_hs = hs[:, -1, :2048].cpu()

    cm = content_mask.unsqueeze(0).unsqueeze(-1).expand_as(hs)
    masked = hs * cm
    mean_hs = (masked.sum(dim=1) / cm.sum(dim=1).clamp(min=1))[:, :2048].cpu()
    last_hs = hs[:, -1, :2048].cpu()

    all_data["angle_token"].append(angle_hs)
    all_data["mean_pool"].append(mean_hs)
    all_data["last_token"].append(last_hs)
    torch.cuda.empty_cache()

for method in all_data:
    all_data[method] = torch.stack(all_data[method])
    print(f"VLM {method}: {list(all_data[method].shape)}")

# --- CLIP baselines ---
print("\nLoading CLIP...")
from transformers import CLIPTextModel, CLIPTokenizer
clip_model = CLIPTextModel.from_pretrained("/root/models/clip-vit-base-patch32", local_files_only=True)
clip_tokenizer = CLIPTokenizer.from_pretrained("/root/models/clip-vit-base-patch32", local_files_only=True)
clip_model.eval().cuda()
print(f"CLIP loaded: {sum(p.numel() for p in clip_model.parameters())/1e6:.0f}M")

# Also load bridge for pooler_output
from lightweight_bridge import LightweightBridge, get_embedding_for_training
clip_b = LightweightBridge(encoder_name="clip")
ckpt = torch.load("/root/Qwen3-VL/checkpoints/clip_en_epoch_080.pt", map_location="cuda")
clip_b.projection.load_state_dict(ckpt["projection"])
clip_b.pose.load_state_dict(ckpt["pose"])
clip_b.eval()

def pair_cos(mat):
    norm = mat / mat.norm(dim=1, keepdim=True)
    return norm @ norm.T

def off_diag_mean(mat):
    N = mat.shape[0]
    return mat[~torch.eye(N, dtype=bool)].mean().item()

far_mask = torch.triu(torch.ones(10, 10), diagonal=4).bool()

def compute_discrim(embeddings):
    sim = pair_cos(embeddings)
    adj = sim.diagonal(offset=1).mean().item()
    far = sim[far_mask].mean().item()
    return round(adj - far, 4), {"adj": round(adj, 4), "far": round(far, 4), "mean": round(off_diag_mean(sim), 4)}

results = {}

# Method 1: CLIP pooler_output (standard)
print("\n=== CLIP pooler_output ===")
e1 = []
for text in PROMPTS:
    emb = get_embedding_for_training(clip_b, text)
    e1.append(emb.squeeze(0))
e1 = torch.stack(e1)
d1, m1 = compute_discrim(e1)
print(f"  Discrim: {d1}  (adj={m1['adj']}, far={m1['far']}, mean={m1['mean']})")
results["CLIP_pooler"] = {**m1, "discrim": d1}

# Method 2: CLIP raw [CLS] (last_hidden_state[:, 0, :])
print("\n=== CLIP raw [CLS] ===")
e2 = []
for text in PROMPTS:
    inputs = clip_tokenizer([text], return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        out = clip_model(**inputs)
    emb = out.last_hidden_state[:, 0, :].float()
    e2.append(emb.squeeze(0))
e2 = torch.stack(e2)
d2, m2 = compute_discrim(e2)
print(f"  Discrim: {d2}  (adj={m2['adj']}, far={m2['far']}, mean={m2['mean']})")
results["CLIP_raw_CLS"] = {**m2, "discrim": d2}

# Method 3: CLIP angle token position
print("\n=== CLIP angle token ===")
def find_clip_angle_pos(input_ids, angle_str, tokenizer):
    ids = input_ids.tolist()
    deg_idx = None
    for i, tid in enumerate(ids):
        t = tokenizer.decode([tid]).strip()
        if "degrees" in t.lower():
            deg_idx = i
            break
    angle_idxs = []
    if deg_idx is not None and deg_idx > 0:
        for j in range(deg_idx - 1, -1, -1):
            t = tokenizer.decode([ids[j]]).strip()
            if any(c.isdigit() for c in t):
                angle_idxs.insert(0, j)
            else:
                break
    if angle_idxs:
        return angle_idxs[-1]
    for i, tid in enumerate(ids):
        t = tokenizer.decode([tid]).strip()
        if any(c.isdigit() for c in t):
            return i
    return 0

# Check CLIP tokenization
print("  CLIP token positions:")
for a in ANGLES[:3]:
    text = f"A person looking to the right at {a} degrees"
    inputs = clip_tokenizer([text], return_tensors="pt")
    pos = find_clip_angle_pos(inputs["input_ids"][0], str(a), clip_tokenizer)
    tok_text = clip_tokenizer.decode([inputs["input_ids"][0, pos].item()]).strip()
    print(f"    Angle {a:3d}: pos={pos}, tok=\"{tok_text}\"")

e3 = []
for text in PROMPTS:
    inputs = clip_tokenizer([text], return_tensors="pt", padding=True, truncation=True).to("cuda")
    pos = find_clip_angle_pos(inputs["input_ids"][0], "", clip_tokenizer)
    with torch.no_grad():
        out = clip_model(**inputs)
    emb = out.last_hidden_state[:, pos, :].float()
    e3.append(emb.squeeze(0))
e3 = torch.stack(e3)
d3, m3 = compute_discrim(e3)
print(f"  Discrim: {d3}  (adj={m3['adj']}, far={m3['far']}, mean={m3['mean']})")
results["CLIP_angle_token"] = {**m3, "discrim": d3}

# Method 4: CLIP mean pool
print("\n=== CLIP mean pool ===")
special_ids = {clip_tokenizer.cls_token_id, clip_tokenizer.sep_token_id, clip_tokenizer.pad_token_id}
e4 = []
for text in PROMPTS:
    inputs = clip_tokenizer([text], return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        out = clip_model(**inputs)
    hs = out.last_hidden_state.float()
    mask = torch.tensor([tid not in special_ids for tid in inputs["input_ids"][0]], device="cuda")
    mask = mask.unsqueeze(0).unsqueeze(-1)
    emb = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    e4.append(emb.squeeze(0))
e4 = torch.stack(e4)
d4, m4 = compute_discrim(e4)
print(f"  Discrim: {d4}  (adj={m4['adj']}, far={m4['far']}, mean={m4['mean']})")
results["CLIP_mean_pool"] = {**m4, "discrim": d4}

# --- VLM results ---
for method in ["angle_token", "mean_pool", "last_token"]:
    data = all_data[method]
    results[method] = {}
    print(f"\n{'='*60}")
    print(f"{f'VLM {method}':^60s}")
    print(f"{'='*60}")
    print(f"{'Group':<22s} {'Mean Sim':>10s} {'Adj':>10s} {'Far':>10s} {'Discrim':>10s}")
    print("  " + "-" * 62)
    for gk, layer_idxs in LAYER_GROUPS.items():
        h = data[:, layer_idxs, :].mean(dim=1)
        sim = pair_cos(h)
        adj = sim.diagonal(offset=1).mean().item()
        far = sim[far_mask].mean().item()
        discrim = round(adj - far, 4)
        print(f"  {gk:<22s} {off_diag_mean(sim):>10.4f} {adj:>10.4f} {far:>10.4f} {discrim:>10.4f}")
        results[method][gk] = {
            "mean_sim": round(off_diag_mean(sim), 4),
            "adj_sim": round(adj, 4),
            "far_sim": round(far, 4),
            "discrim": discrim,
        }

# Save all results
out_path = "/root/Qwen3-VL/paper/figures/fixed_test/embedding_layer_ablation.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")

# Print final comparison table
print(f"\n{'='*70}")
print(f"{'FINAL COMPARISON: All Methods vs CLIP':^70s}")
print(f"{'='*70}")
print(f"{'Method':<28s} {'Discrim':>10s}")
print("  " + "-" * 38)
for method in ["angle_token", "mean_pool", "last_token"]:
    for gk in LAYER_GROUPS:
        d = results[method][gk]["discrim"]
        print(f"  {'VLM ' + method + ' (' + gk + ')':<28s} {d:>10.4f}")
for clip_method in ["CLIP_pooler", "CLIP_raw_CLS", "CLIP_angle_token", "CLIP_mean_pool"]:
    d = results[clip_method]["discrim"]
    print(f"  {clip_method:<28s} {d:>10.4f}")

print("\nDONE!")
