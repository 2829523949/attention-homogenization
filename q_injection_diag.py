"""Diagnostic: verify Q injection works - FIXED signature"""
import os, torch
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, eager_attention_forward

model = AutoModelForImageTextToText.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16,
    device_map="cuda", attn_implementation="eager",
    local_files_only=True, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(
    "/root/Qwen3-VL-2B-Instruct", local_files_only=True, trust_remote_code=True)
lm = model.model.language_model

img = Image.open("/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data/AFLW2000-3D_crop/image00002.jpg").convert("RGB")
msg = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "describe the nose"}]}]
prompt = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
inp = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
ids = inp["input_ids"][0]
ve = (ids == 151653).nonzero(as_tuple=True)[0].item()
vs, nvt = 4, ve - 4
print(f"nvt={nvt}")

orig18 = lm.layers[17].self_attn.forward
q18_val = None
def cap18(hidden_states, position_embeddings, attention_mask=None, **kw):
    global q18_val
    attn = lm.layers[17].self_attn
    q = attn.q_norm(attn.q_proj(hidden_states).view((*hidden_states.shape[:-1], -1, attn.head_dim))).transpose(1, 2)
    c, s = position_embeddings
    q, _ = apply_rotary_pos_emb(q, q, c, s)
    q18_val = q[:, :, :nvt, :].detach().cpu()
    return orig18(hidden_states, position_embeddings, attention_mask, **kw)

lm.layers[17].self_attn.forward = cap18
with torch.no_grad(): model(**inp)
lm.layers[17].self_attn.forward = orig18
del inp; torch.cuda.empty_cache()
print(f"Q18 captured, shape={q18_val.shape}")

inp2 = processor(text=prompt, images=img, return_tensors="pt").to("cuda")
q28_val = None
orig28 = lm.layers[27].self_attn.forward
def cap28(hidden_states, position_embeddings, attention_mask=None, **kw):
    global q28_val
    attn = lm.layers[27].self_attn
    q = attn.q_norm(attn.q_proj(hidden_states).view((*hidden_states.shape[:-1], -1, attn.head_dim))).transpose(1, 2)
    c, s = position_embeddings
    q, _ = apply_rotary_pos_emb(q, q, c, s)
    q28_val = q[:, :, :nvt, :].detach().cpu()
    return orig28(hidden_states, position_embeddings, attention_mask, **kw)

lm.layers[27].self_attn.forward = cap28
with torch.no_grad(): model(**inp2)
lm.layers[27].self_attn.forward = orig28
del inp2; torch.cuda.empty_cache()

cos = torch.nn.functional.cosine_similarity(q18_val.flatten(), q28_val.flatten(), dim=0)
print(f"Q18 vs Q28 cos_sim: {cos.item():.6f}")
print(f"Q18 norm={q18_val.norm().item():.4f}, Q28 norm={q28_val.norm().item():.4f}")
print(f"Max abs diff: {(q18_val - q28_val).abs().max().item():.6f}")

# Baseline attn
lm.layers[27].self_attn.forward = orig28
lm.layers[27].self_attn.output_attentions = True

def get_attn(query_str):
    mn = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": f"describe the {query_str}"}]}]
    pp = processor.apply_chat_template(mn, tokenize=False, add_generation_prompt=True)
    ip = processor(text=pp, images=img, return_tensors="pt").to("cuda")
    iid = ip["input_ids"][0]
    ve2 = (iid == 151653).nonzero(as_tuple=True)[0].item()
    ie2 = (iid == 151645).nonzero(as_tuple=True)[0].item()
    vs2, qs2 = 4, ve2 + 1
    toks = [processor.decode([t]).strip() for t in iid.tolist()]
    kw = [qs2 + i for i, t in enumerate(toks[qs2:ie2]) if query_str.split()[0].lower() in t.lower()]
    qp = kw[0] if kw else qs2
    aw = []
    hh = lm.layers[27].self_attn.register_forward_hook(lambda m,i,o: aw.append(o[1].detach()) if len(o)>1 and o[1] is not None else None)
    with torch.no_grad(): model(**ip)
    hh.remove()
    del ip; torch.cuda.empty_cache()
    if aw: return aw[0][0, :, qp, vs2:ve2].mean(dim=0).cpu().float().numpy()
    return None

from scipy.stats import pearsonr
import numpy as np
m_nose = get_attn("nose")
m_eyes = get_attn("eyes")
r_base = pearsonr(m_nose, m_eyes)[0]
print(f"\nBaseline within-face r: {r_base:.4f}")

# Injection
call_count = [0]
def inject_fwd(hidden_states, position_embeddings, attention_mask=None, **kw):
    call_count[0] += 1
    attn = lm.layers[27].self_attn
    q = attn.q_norm(attn.q_proj(hidden_states).view((*hidden_states.shape[:-1], -1, attn.head_dim))).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(hidden_states).view((*hidden_states.shape[:-1], -1, attn.head_dim))).transpose(1, 2)
    v = attn.v_proj(hidden_states).view((*hidden_states.shape[:-1], -1, attn.head_dim)).transpose(1, 2)
    c, s = position_embeddings
    q_r, k = apply_rotary_pos_emb(q, k, c, s)
    q = q.clone()
    q[:, :, :nvt, :] = q18_val.to(q.device, q.dtype)
    attn_o, attn_w = eager_attention_forward(attn, q, k, v, attention_mask, dropout=0.0, scaling=attn.scaling)
    attn_o = attn_o.reshape(*hidden_states.shape[:-1], -1).contiguous()
    attn_o = attn.o_proj(attn_o)
    return attn_o, attn_w

lm.layers[27].self_attn.forward = inject_fwd
m_nose_i = get_attn("nose")
m_eyes_i = get_attn("eyes")
lm.layers[27].self_attn.forward = orig28

r_inj = pearsonr(m_nose_i, m_eyes_i)[0]
print(f"Injected within-face r:             {r_inj:.4f}")
print(f"Delta:                              {r_inj - r_base:+.4f}")
print(f"Max diff nose attn map:            {np.abs(m_nose_i - m_nose).max():.6f}")
print(f"Monkey-patch called: {call_count[0]} times")
print(f"Q18 vs Q28 identical: {(q18_val - q28_val).abs().max().item() < 1e-6}")
