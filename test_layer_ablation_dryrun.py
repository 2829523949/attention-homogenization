"""
Dry-run: Validate token indexing and layer hooking WITHOUT running full experiment.
Loads 1 image, 1 query, checks:
  1. Token indices (vision_start, vision_end, query_start, query_end)
  2. Attention weights shape from each layer group
  3. Visual token count matches 16x16 grid
"""
import os, sys, torch
import numpy as np
from PIL import Image
sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

print("Loading Qwen3-VL...")
from transformers import AutoModelForImageTextToText, AutoProcessor
model = AutoModelForImageTextToText.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
    device_map="cuda", attn_implementation="eager",
    local_files_only=True, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
lm = model.model.language_model
print(f"Total LM layers: {len(lm.layers)}")

# Test image + query
img = Image.open("/root/output_phase1_test/右侧15度平视.png").convert("RGB")
query = "nose"

messages = [{"role": "user", "content": [
    {"type": "image", "image": img},
    {"type": "text", "text": f"describe the {query}"}
]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"][0]

print(f"\nInput IDs length: {len(input_ids)}")

# Token indexing
T_VISION_START = 151652
T_VISION_END = 151653
T_IM_END = 151645

vs_pos = (input_ids == T_VISION_START).nonzero(as_tuple=True)[0].item()
ve_pos = (input_ids == T_VISION_END).nonzero(as_tuple=True)[0].item()
ie_pos = (input_ids == T_IM_END).nonzero(as_tuple=True)[0].item()

print(f"\n-- Token indices --")
print(f"  <|vision_start|> @ {vs_pos}")
print(f"  <|vision_end|>   @ {ve_pos}")
print(f"  <|im_end|>       @ {ie_pos}")

vis_start = vs_pos + 1
vis_end = ve_pos
q_start = ve_pos + 1
q_end = ie_pos
nvt = vis_end - vis_start
nvt_dim = int(nvt ** 0.5)

print(f"\n  Visual tokens: [{vis_start}:{vis_end}) = {nvt} tokens")
print(f"  Query tokens:  [{q_start}:{q_end}) = {q_end - q_start} tokens")
print(f"  Grid: {nvt_dim}x{nvt_dim}")

assert nvt == 256, f"Expected 256 visual tokens, got {nvt}"
assert nvt_dim == 16, f"Expected 16x16 grid, got {nvt_dim}x{nvt_dim}"
print(f"  OK - visual token count and grid size CORRECT")

# Test each layer group
LAYER_GROUPS = {
    "A_Early (L1-3)":  [0, 1, 2],
    "B_Mid (L8-10)":   [7, 8, 9],
    "C_Later (L18-20)":[17, 18, 19],
    "D_Last (L25-27)": [25, 26, 27],
}

print(f"\n-- Testing layer hooking --")
for gk, layer_indices in LAYER_GROUPS.items():
    attn_weights = []

    def make_hook(weights_list):
        def hook(m, i, o):
            if isinstance(o, tuple) and len(o) > 1 and o[1] is not None:
                weights_list.append(o[1].detach())
        return hook

    handles = []
    for lidx in layer_indices:
        handles.append(lm.layers[lidx].self_attn.register_forward_hook(make_hook(attn_weights)))

    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    for h in handles:
        h.remove()

    print(f"  {gk}: {len(attn_weights)} weight tensors captured")
    if attn_weights:
        shapes = [list(w.shape) for w in attn_weights]
        print(f"         shapes: {shapes}")
        all_same = all(s == attn_weights[0].shape for s in shapes)
        print(f"         all same shape: {all_same}" + (" OK" if all_same else " FAIL"))

        attn = torch.stack(attn_weights).mean(dim=0)
        attn = attn.mean(dim=[0, 1])
        print(f"         averaged shape: {list(attn.shape)}")

        vis_attn = attn[q_start:q_end, vis_start:vis_end].mean(dim=0).cpu().float().numpy()
        print(f"         vis_attn shape: {vis_attn.shape}")
        vis_attn_2d = vis_attn.reshape(nvt_dim, nvt_dim)
        print(f"         heatmap shape: {vis_attn_2d.shape} OK")
        print(f"         heatmap range: [{vis_attn_2d.min():.6f}, {vis_attn_2d.max():.6f}]")
    else:
        print(f"         FAIL - NO ATTENTION CAPTURED")

print(f"\n-- All checks complete --")
