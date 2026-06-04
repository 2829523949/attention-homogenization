"""Save the corrected attention analysis data."""
import sys, os, json, torch
import numpy as np
from scipy.stats import pearsonr
sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

model = AutoModelForImageTextToText.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", dtype=torch.bfloat16,
    device_map="cuda", attn_implementation="eager",
    local_files_only=True, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)

QUERIES = ["nose", "eyes", "mouth", "chin", "forehead", "beard",
           "whole image", "bottom-left area"]
T_VISION_END, T_IM_END = 151653, 151645
IMG_DIR = "/root/output_phase1_test"

all_results = {}
for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.endswith(".png") or fname == "grid_9views.png":
        continue
    img_path = os.path.join(IMG_DIR, fname)
    img = Image.open(img_path).convert("RGB")
    print(f"Processing {fname}...")
    
    results = {}
    for query in QUERIES:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": f"describe the {query}"}
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
        ids = inputs["input_ids"][0]
        
        vis_end = (ids == T_VISION_END).nonzero(as_tuple=True)[0].item()
        im_end = (ids == T_IM_END).nonzero(as_tuple=True)[0].item()
        q_start, q_end = vis_end + 1, im_end
        
        attn_w = []
        def hook(m, i, o):
            if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
                attn_w.append(o[1].detach())
        handles = [layer.self_attn.register_forward_hook(hook) for layer in model.model.language_model.layers[-3:]]
        with torch.no_grad():
            out = model(**inputs, output_attentions=True)
        for h in handles: h.remove()
        
        attn = torch.stack(attn_w).mean(dim=0).mean(dim=[0, 1])
        vis = attn[q_start:q_end, vis_start:vis_end].mean(dim=0).cpu().float().numpy() if 'vis_start' in dir() else None
        
        # Get actual indices  
        nvt = vis_end - 4  # 4 = position after <|vision_start|>
        vis = attn[q_start:q_end, 4:vis_end].mean(dim=0).cpu().float().numpy()
        results[query] = vis.tolist()
    
    all_results[fname] = results

# Compute similarity matrices
all_similarities = {}
for fname in all_results:
    flat_maps = {q: np.array(all_results[fname][q]) for q in QUERIES}
    nq = len(QUERIES)
    sim = []
    for i in range(nq):
        row = []
        for j in range(nq):
            r, _ = pearsonr(flat_maps[QUERIES[i]], flat_maps[QUERIES[j]])
            row.append(r)
        sim.append(row)
    all_similarities[fname] = sim

# Save as JSON
out = {
    "queries": QUERIES,
    "heatmaps": {f: {q: all_results[f][q] for q in QUERIES} for f in all_results},
    "similarities": {f: all_similarities[f] for f in all_results},
    "summary": {}
}

# Summary stats
for fname in all_results:
    flat_maps = {q: np.array(all_results[fname][q]) for q in QUERIES}
    sims = []
    for i, q1 in enumerate(QUERIES):
        for j, q2 in enumerate(QUERIES):
            if i < j:
                r, _ = pearsonr(flat_maps[q1], flat_maps[q2])
                sims.append(r)
    out["summary"][fname] = {"min": float(np.min(sims)), "max": float(np.max(sims)), "mean": float(np.mean(sims)), "std": float(np.std(sims))}

with open("/root/Qwen3-VL/paper/figures/attention_results.json", "w") as f:
    json.dump(out, f, indent=2)

print("\n=== Summary ===")
for fname in sorted(out["summary"]):
    s = out["summary"][fname]
    print(f"  {fname}: min={s['min']:.4f}, max={s['max']:.4f}, mean={s['mean']:.4f}")

# Face groups analysis
face = QUERIES[:6]
control = QUERIES[6:]
face_sims, fc_sims, ctrl_sims = [], [], []
for fname in all_results:
    flat = {q: np.array(all_results[fname][q]) for q in QUERIES}
    for q1 in face:
        for q2 in face:
            if face.index(q1) < face.index(q2):
                r, _ = pearsonr(flat[q1], flat[q2])
                face_sims.append(r)
    for q1 in face:
        for q2 in control:
            r, _ = pearsonr(flat[q1], flat[q2])
            fc_sims.append(r)
    for q1 in control:
        for q2 in control:
            if control.index(q1) < control.index(q2):
                r, _ = pearsonr(flat[q1], flat[q2])
                ctrl_sims.append(r)

out["groups"] = {
    "within_face": {"mean": float(np.mean(face_sims)), "min": float(np.min(face_sims))},
    "face_vs_control": {"mean": float(np.mean(fc_sims)), "min": float(np.min(fc_sims))},
    "within_control": {"mean": float(np.mean(ctrl_sims)), "min": float(np.min(ctrl_sims))},
}
with open("/root/Qwen3-VL/paper/figures/attention_results.json", "w") as f:
    json.dump(out, f, indent=2)

print(f"\nGroup analysis (all images):")
print(f"  Within face features:  mean={np.mean(face_sims):.4f}, min={np.min(face_sims):.4f}")
print(f"  Face vs control:       mean={np.mean(fc_sims):.4f}, min={np.min(fc_sims):.4f}")
print(f"  Within control:        mean={np.mean(ctrl_sims):.4f}")
print("\nData saved to attention_results.json")
