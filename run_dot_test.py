"""Dot test: extract attention from COLOR token only, not averaged across all query tokens"""
import os, sys, math, torch, numpy as np
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"; os.environ["HF_HUB_OFFLINE"] = "1"
OUT_DIR = "/root/Qwen3-VL/paper/figures/dot_test"; os.makedirs(OUT_DIR, exist_ok=True)

# ═══ 256x256 image with 16x16 color blocks ═══
SIZE = 256; CELL = 16
img = Image.new("RGB", (SIZE, SIZE), (0, 0, 0)); draw = ImageDraw.Draw(img)
blocks = {"red":(1,1), "green":(7,7), "blue":(14,14)}
colors = {"red":(255,0,0), "green":(0,255,0), "blue":(0,0,255)}
for n, (gx, gy) in blocks.items():
    x, y = gx*CELL, gy*CELL; draw.rectangle([x, y, x+CELL-1, y+CELL-1], fill=colors[n])
img.save(f"{OUT_DIR}/synthetic_dots.png"); print(f"Image: {SIZE}x{SIZE}")
LAYERS = [17, 18, 19]  # Best layer group

# ═══ Qwen3-VL ═══
print(f"\n{'='*60}\nQwen3-VL-2B\n{'='*60}")
from transformers import AutoModelForImageTextToText, AutoProcessor
qwen = AutoModelForImageTextToText.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct",torch_dtype=torch.bfloat16,
    device_map="cuda",attn_implementation="eager",
    local_files_only=True,trust_remote_code=True)
qproc = AutoProcessor.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct",local_files_only=True,trust_remote_code=True)
qlm = qwen.model.language_model
for l in LAYERS: qlm.layers[l].self_attn.output_attentions = True

# Test each color query + "image" control
QUERIES = {"red":"red","green":"green","blue":"blue","image":None}  # query_text:color_word
results = {}

for q_text, color_word in QUERIES.items():
    msg = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":f"describe the {q_text}"}]}]
    pr = qproc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
    inp = qproc(text=pr,images=img,return_tensors="pt").to("cuda")
    ids = inp["input_ids"][0]
    ve = (ids==151653).nonzero(as_tuple=True)[0].item()
    ie = (ids==151645).nonzero(as_tuple=True)[0].item()
    
    # Find which token index is the COLOR word
    txt_tokens = ids[ve+1:ie]  # query text tokens
    token_texts = qproc.tokenizer.convert_ids_to_tokens(txt_tokens.tolist())
    print(f"\n  Query '{q_text}': tokens = {token_texts}")
    
    # Find color token position
    color_pos = None
    if color_word:
        for j, t in enumerate(token_texts):
            if color_word in t.lower():
                color_pos = ve + 1 + j
                print(f"    -> Color token '{t}' at position {color_pos} (idx {j} in query)")
                break
    
    at = []; hs = [qlm.layers[l].self_attn.register_forward_hook(
        lambda m,i,o: at.append(o[1].detach()) if isinstance(o,tuple)and len(o)>1 and o[1] is not None else None)
        for l in LAYERS]
    with torch.no_grad(): out = qwen(**inp)
    for h in hs: h.remove()
    
    if at:
        a = torch.stack(at).mean(dim=0).mean(dim=[0,1])  # [q_len, k_len]
        nvt = ve - 4
        
        # Method 1: Extract from COLOR token only (if found)
        if color_pos is not None:
            color_attn = a[color_pos, 4:ve].cpu().float().numpy()
        else:
            color_attn = np.zeros(nvt)
        
        # Method 2: Average over all query tokens (original method)
        avg_attn = a[ve+1:ie, 4:ve].mean(dim=0).cpu().float().numpy()
        
        results[q_text] = {"color": color_attn, "avg": avg_attn, "nvt": nvt}
        
        # Print peaks for both methods
        s = int(math.isqrt(nvt))
        if s*s == nvt:
            hm_c = color_attn.reshape(s, s)
            hm_a = avg_attn.reshape(s, s)
            pk_c = np.unravel_index(hm_c.argmax(), hm_c.shape)
            pk_a = np.unravel_index(hm_a.argmax(), hm_a.shape)
            print(f"    COLOR-peak: {pk_c} val={hm_c[pk_c]:.6f}")
            print(f"    AVG-peak:   {pk_a} val={hm_a[pk_a]:.6f}")
    
    del at, a, out, inp; torch.cuda.empty_cache()

# Cross-query similarity (color token version)
print(f"\n  Cross-r (COLOR token version):")
color_keys = [k for k in results if results[k]["color"].sum() > 0]  # non-zero
for i, q1 in enumerate(color_keys):
    for q2 in color_keys[i+1:]:
        r,_ = pearsonr(results[q1]["color"], results[q2]["color"])
        print(f"    {q1:15s} vs {q2:15s}: r={r:.4f}")

print(f"\n  Cross-r (AVG version):")
for i, q1 in enumerate(QUERIES):
    for q2 in list(QUERIES.keys())[i+1:]:
        r,_ = pearsonr(results[q1]["avg"], results[q2]["avg"])
        print(f"    {q1:15s} vs {q2:15s}: r={r:.4f}")

# Heatmap figures with both methods side by side
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.patch.set_facecolor("white")
for i, q in enumerate(QUERIES):
    r = results[q]; s = int(math.isqrt(r["nvt"]))
    # Row 0: Color token
    hm_c = r["color"].reshape(s, s)
    axes[0, i].imshow(hm_c, cmap="hot", interpolation="nearest")
    axes[0, i].set_title(f'Color-tok: "{q}"', fontsize=9); axes[0, i].axis("off")
    if q in blocks:
        gy,gx = blocks[q]; axes[0, i].plot(gx, gy, 'c*', ms=12, mec='white', mew=1.5)
    # Row 1: Average
    hm_a = r["avg"].reshape(s, s)
    axes[1, i].imshow(hm_a, cmap="hot", interpolation="nearest")
    axes[1, i].set_title(f'Avg-tok: "{q}"', fontsize=9); axes[1, i].axis("off")
    if q in blocks:
        gy,gx = blocks[q]; axes[1, i].plot(gx, gy, 'c*', ms=12, mec='white', mew=1.5)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/qwen_dot.png", dpi=150); plt.close()
del qwen, qlm; torch.cuda.empty_cache()

# ═══ LLaVA-NeXT ═══
print(f"\n{'='*60}\nLLaVA-NeXT-7B\n{'='*60}")
from transformers import LlavaNextForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4")
lmod = LlavaNextForConditionalGeneration.from_pretrained(
    "/root/Qwen3-VL",quantization_config=bnb,device_map="auto",
    attn_implementation="eager",local_files_only=True,trust_remote_code=True,
    torch_dtype=torch.bfloat16)
lmod.config.image_grid_pinpoints = [[336, 336]]
lproc = AutoProcessor.from_pretrained(
    "/root/Qwen3-VL",local_files_only=True,trust_remote_code=True)
lproc.chat_template = ("{{ bos_token }}{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0%2==0) %}{{ raise_exception('') }}{% endif %}"
    "{% if message['role'] == 'user' %}{{ '[INST] '+message['content']|trim+' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}{{ ' '+message['content']|trim+' '+eos_token }}{% endif %}{% endfor %}")
llm = lmod.model.language_model
for l in LAYERS: llm.layers[l].self_attn.output_attentions = True

l_results = {}
for q_text, color_word in QUERIES.items():
    msg = [{"role":"user","content":f"<image>\ndescribe the {q_text}"}]
    pr = lproc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
    inp = lproc(text=pr,images=img,return_tensors="pt").to("cuda")
    ids = inp["input_ids"][0]
    ip = (ids==32000).nonzero(as_tuple=True)[0]
    vs, ve = int(ip[0]), int(ip[-1])+1
    txt = ids[ve:]
    toks = lproc.tokenizer.convert_ids_to_tokens(txt.tolist())
    print(f"\n  Query '{q_text}': tokens = {toks}")
    
    color_pos = None
    if color_word:
        for j, t in enumerate(toks):
            if color_word in t.lower():
                color_pos = ve + j
                print(f"    -> Color token '{t}' at position {color_pos}")
                break
    
    at = []; hs = [llm.layers[l].self_attn.register_forward_hook(
        lambda m,i,o: at.append(o[1].detach()) if isinstance(o,tuple)and len(o)>1 and o[1] is not None else None)
        for l in LAYERS]
    with torch.no_grad(): out = lmod(**inp)
    for h in hs: h.remove()
    
    if at:
        a = torch.stack(at).mean(dim=0).mean(dim=[0,1])
        nvt = ve - vs
        if color_pos is not None:
            color_attn = a[color_pos, vs:ve].cpu().float().numpy()
        else:
            color_attn = np.zeros(nvt)
        avg_attn = a[ve:, vs:ve].mean(dim=0).cpu().float().numpy()
        l_results[q_text] = {"color": color_attn, "avg": avg_attn, "nvt": nvt}
        s = int(math.isqrt(nvt))
        if s*s == nvt:
            hm_c = color_attn.reshape(s, s); hm_a = avg_attn.reshape(s, s)
            pk_c = np.unravel_index(hm_c.argmax(), hm_c.shape)
            pk_a = np.unravel_index(hm_a.argmax(), hm_a.shape)
            print(f"    COLOR-peak: {pk_c} val={hm_c[pk_c]:.6f}")
            print(f"    AVG-peak:   {pk_a} val={hm_a[pk_a]:.6f}")
        else:
            print(f"    Non-square({nvt}), skip peak")
    del at, a, out, inp; torch.cuda.empty_cache()

# Cross-r
color_keys = [k for k in l_results if l_results[k]["color"].sum() > 0]
if len(color_keys) >= 2:
    print(f"\n  Cross-r (COLOR token version):")
    for i, q1 in enumerate(color_keys):
        for q2 in color_keys[i+1:]:
            r,_ = pearsonr(l_results[q1]["color"], l_results[q2]["color"])
            print(f"    {q1:15s} vs {q2:15s}: r={r:.4f}")

# LLaVA figure
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.patch.set_facecolor("white")
for i, q in enumerate(QUERIES):
    r = l_results.get(q); s = int(math.isqrt(r["nvt"])) if r else 1
    if r and s*s == r["nvt"]:
        axes[0,i].imshow(r["color"].reshape(s,s), cmap="hot", interpolation="nearest")
        axes[1,i].imshow(r["avg"].reshape(s,s), cmap="hot", interpolation="nearest")
    axes[0,i].set_title(f'Color-tok: "{q}"', fontsize=9); axes[0,i].axis("off")
    axes[1,i].set_title(f'Avg-tok: "{q}"', fontsize=9); axes[1,i].axis("off")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/llava_dot.png", dpi=150); plt.close()
del lmod, llm; torch.cuda.empty_cache()
print(f"\nDONE! {OUT_DIR}/")
