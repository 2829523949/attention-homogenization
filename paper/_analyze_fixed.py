import sys, os, torch, numpy as np
from scipy.stats import pearsonr
sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

model = AutoModelForImageTextToText.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
    device_map="cuda", attn_implementation="eager",
    local_files_only=True, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)

QUERIES = ["nose", "eyes", "mouth", "chin", "forehead", "beard",
           "whole image", "bottom-left area"]

img = Image.open("/root/output_phase1_test/右侧15度平视.png").convert("RGB")
T_VISION_END = 151653
T_IM_END = 151645

# Extract attention for each query
all_flat = {}
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
    vis_start, vis_end_t = 4, vis_end
    q_start, q_end = vis_end + 1, im_end
    
    attn_w = []
    def hook(m, i, o):
        if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
            attn_w.append(o[1].detach())
    
    handles = [layer.self_attn.register_forward_hook(hook) 
               for layer in model.model.language_model.layers[-3:]]
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    for h in handles: h.remove()
    
    attn = torch.stack(attn_w).mean(dim=0).mean(dim=[0, 1])
    vis = attn[q_start:q_end, vis_start:vis_end_t].mean(dim=0).cpu().float().numpy()
    all_flat[query] = vis.flatten()

# Full similarity matrix
nq = len(QUERIES)
print("Full similarity matrix:\n")
print(f"{'':16s}", end="")
for q in QUERIES:
    print(f"{q:16s}", end="")
print()

for i, q1 in enumerate(QUERIES):
    print(f"{q1:16s}", end="")
    for j, q2 in enumerate(QUERIES):
        r, _ = pearsonr(all_flat[q1], all_flat[q2])
        print(f"{r:>12.4f}   ", end="")
    print()

# Analysis
print("\n=== Analysis ===")
# Group queries
face_upper = ["nose", "eyes", "forehead"]
face_lower = ["mouth", "chin", "beard"]
control = ["whole image", "bottom-left area"]

all_upper = []
all_lower = []
all_face_vs_control = []
all_within_control = []

for i, q1 in enumerate(QUERIES):
    for j, q2 in enumerate(QUERIES):
        if i >= j: continue
        r, _ = pearsonr(all_flat[q1], all_flat[q2])
        in_upper = q1 in face_upper and q2 in face_upper
        in_lower = q1 in face_lower and q2 in face_lower
        cross_face_control = (q1 in face_upper + face_lower and q2 in control) or \
                             (q2 in face_upper + face_lower and q1 in control)
        
        if in_upper: all_upper.append(r)
        if in_lower: all_lower.append(r)
        if cross_face_control: all_face_vs_control.append(r)
        if q1 in control and q2 in control: all_within_control.append(r)

print(f"Within face-upper features:  mean={np.mean(all_upper):.4f}, min={min(all_upper):.4f}")
print(f"Within face-lower features:  mean={np.mean(all_lower):.4f}, min={min(all_lower):.4f}")
print(f"Face vs control:             mean={np.mean(all_face_vs_control):.4f}, min={min(all_face_vs_control):.4f}")
print(f"Control vs control:          mean={np.mean(all_within_control):.4f}")

# Best and worst pairs
best_pair, best_r = "", 0
worst_pair, worst_r = "", 1
for i, q1 in enumerate(QUERIES):
    for j, q2 in enumerate(QUERIES):
        if i >= j: continue
        r, _ = pearsonr(all_flat[q1], all_flat[q2])
        if r > best_r:
            best_r = r
            best_pair = f"{q1} vs {q2}"
        if r < worst_r:
            worst_r = r
            worst_pair = f"{q1} vs {q2}"

print(f"\nMost similar:  {best_pair} (r={best_r:.4f})")
print(f"Most diff:     {worst_pair} (r={worst_r:.4f})")
