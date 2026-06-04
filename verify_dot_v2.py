"""Verify dot test correctness: check EVERY step"""
import os, sys, torch
from PIL import Image
sys.path.insert(0, "/root/Qwen3-VL")

img = Image.open("/root/Qwen3-VL/paper/figures/dot_test/synthetic_dots.png")
print(f"1. IMAGE: {img.size}x{img.size}")

# Qwen3-VL
from transformers import AutoModelForImageTextToText, AutoProcessor
qwen = AutoModelForImageTextToText.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct",torch_dtype=torch.bfloat16,
    device_map="cuda",attn_implementation="eager",
    local_files_only=True,trust_remote_code=True)
qproc = AutoProcessor.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct",local_files_only=True,trust_remote_code=True)
qlm = qwen.model.language_model

# Step 2: Check processor output
msg = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"describe the red area"}]}]
prompt = qproc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
inputs = qproc(text=prompt,images=img,return_tensors="pt")
ids = inputs["input_ids"][0]
ve = (ids==151653).nonzero(as_tuple=True)[0].item()
ie = (ids==151645).nonzero(as_tuple=True)[0].item()
nvt = ve - 4
print(f"2. TOKENS: total={len(ids)}, vision=[4:{ve})={nvt}tokens, query=[{ve+1}:{ie})={ie-ve-1}tokens")

# Check image_grid_thw
thw = inputs.get("image_grid_thw", None)
if thw is not None:
    t,h,w = thw[0].tolist()
    print(f"3. PROCESSOR GRID: image_grid_thw=[{t},{h},{w}] → patches={h}x{w} → merge(/2)={h//2}x{w//2}")
    assert h//2 == 8 and w//2 == 8, f"Expected 8x8 after merge, got {h//2}x{w//2}"

# Step 4: Run forward pass, extract attention
for l in [17,18,19]: qlm.layers[l].self_attn.output_attentions = True
at = []
hs = [qlm.layers[l].self_attn.register_forward_hook(
    lambda m,i,o: at.append(o[1].detach()) if isinstance(o,tuple)and len(o)>1 and o[1] is not None else None)
    for l in [17,18,19]]
with torch.no_grad(): out = qwen(**inputs)
for h in hs: h.remove()

if at:
    a = torch.stack(at).mean(dim=0).mean(dim=[0,1])
    print(f"4. ATTENTION SHAPE: {list(a.shape)} (q_len={a.shape[0]}, k_len={a.shape[1]})")
    
    # Extract query→visual attention
    q_attn = a[ve+1:ie, 4:ve]  # [query_tokens, visual_tokens]
    print(f"5. QUERY→VISUAL: {list(q_attn.shape)}")
    print(f"   Query tokens: {len(q_attn)} tokens attending to {nvt} visual positions")
    
    # Average over query tokens
    vis_attn = q_attn.mean(dim=0)  # [nvt]
    print(f"6. AVG QUERY ATTENTION: {len(vis_attn)} values")
    
    # Check sum (should be ~1.0 if softmax over visual tokens)
    print(f"   Sum of attention weights: {vis_attn.sum().item():.4f}")
    
    # Reshape
    s = int(len(vis_attn) ** 0.5)
    hm = vis_attn.reshape(s, s).cpu().float().numpy()
    pk = (int((hm.argmax() // s).item()), int((hm.argmax() % s).item()))
    print(f"7. HEATMAP: {s}x{s}, peak={pk}")
    
    # Now do the same for ALL queries to check if peaks differ
    print(f"\n8. ALL QUERIES:")
    for q in ["red area","green area","blue area","image"]:
        msg = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":f"describe the {q}"}]}]
        pr = qproc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
        inp = qproc(text=pr,images=img,return_tensors="pt").to("cuda")
        ids2 = inp["input_ids"][0]
        ve2 = (ids2==151653).nonzero(as_tuple=True)[0].item()
        ie2 = (ids2==151645).nonzero(as_tuple=True)[0].item()
        at2 = []; hs2 = [qlm.layers[l].self_attn.register_forward_hook(
            lambda m,i,o: at2.append(o[1].detach()) if isinstance(o,tuple)and len(o)>1 and o[1] is not None else None)
            for l in [17,18,19]]
        with torch.no_grad(): o2 = qwen(**inp)
        for h2 in hs2: h2.remove()
        if at2:
            a2 = torch.stack(at2).mean(dim=0).mean(dim=[0,1])
            va2 = a2[ve2+1:ie2, 4:ve2].mean(dim=0).cpu().float().numpy()
            s2 = int(len(va2)**0.5)
            hm2 = va2.reshape(s2, s2)
            pk2 = np.unravel_index(hm2.argmax(), hm2.shape) if 'np' in dir() else (0,0)
            print(f"   [{q:15s}] peak={pk2} val={hm2[pk2]:.4f}")
        del at2,a2,o2,inp; torch.cuda.empty_cache()
else:
    print("No attention captured!")
