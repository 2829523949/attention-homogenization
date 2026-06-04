"""Full LLaVA-NeXT layer ablation - with anyres disabled to fit 8GB VRAM"""
import os, sys, math, json, torch, numpy as np
from PIL import Image
from scipy.stats import pearsonr
sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import LlavaNextForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

OUT_DIR = "/root/Qwen3-VL/paper/figures/llava"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load model (4-bit) ──
print("Loading LLaVA-NeXT (4-bit)...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "/root/Qwen3-VL", quantization_config=bnb, device_map="auto",
    attn_implementation="eager", local_files_only=True, trust_remote_code=True,
    torch_dtype=torch.bfloat16)
proc = AutoProcessor.from_pretrained(
    "/root/Qwen3-VL", local_files_only=True, trust_remote_code=True)
# Disable anyres: only use base 336x336 resolution
model.config.image_grid_pinpoints = []
print(f"Loaded. {len(model.model.language_model.model.layers)} LM layers. anyres=OFF")

lm = model.model.language_model.model

# ── Config ──
LAYER_GROUPS = {"Early (L1-3)": [0,1,2], "Mid (L8-10)": [7,8,9],
                "Later (L18-20)": [17,18,19], "Last (L29-31)": [29,30,31]}
QUERIES = ["nose","eyes","mouth","chin","forehead","beard",
           "whole image","bottom-left area"]
IMG_DIR = "/root/output_phase1_test"
images = sorted([f for f in os.listdir(IMG_DIR)
                if f.endswith((".png",".jpg",".jpeg")) and f != "grid_9views.png"])[:3]

# Mistral chat template
proc.chat_template = ("{{ bos_token }}{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('') }}{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ '[INST] ' + message['content']|trim + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' ' + message['content']|trim + ' ' + eos_token }}"
    "{% endif %}{% endfor %}")

def get_token_ranges(input_ids):
    ip = (input_ids == 32000).nonzero(as_tuple=True)[0]
    if len(ip) == 0:
        return {"vis_start": 0, "vis_end": 0, "q_start": 1, "q_end": len(input_ids)}
    return {"vis_start": int(ip[0]), "vis_end": int(ip[-1]) + 1,
            "q_start": int(ip[-1]) + 1, "q_end": len(input_ids)}

def build_messages(image, query):
    return [{"role": "user", "content": f"<image>\ndescribe the {query}"}]

results = {}
for fname in images:
    img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
    print(f"\n{'='*50}\n[Image] {fname}")
    for gk, layers in LAYER_GROUPS.items():
        print(f"  [{gk}] layers={layers}  ", end="", flush=True)
        img_res = {}
        for q in QUERIES:
            msg = build_messages(img, q)
            prompt = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=prompt, images=img, return_tensors="pt").to("cuda")
            tr = get_token_ranges(inputs["input_ids"][0])

            for l in layers:
                lm.layers[l].self_attn.output_attentions = True

            attn = []
            handles = [lm.layers[l].self_attn.register_forward_hook(
                lambda m, i, o: attn.append(o[1].detach())
                if isinstance(o, tuple) and len(o) > 1 and o[1] is not None else None)
                for l in layers]

            with torch.no_grad():
                out = model(**inputs)
            for h in handles: h.remove()

            if attn:
                a = torch.stack(attn).mean(dim=0).mean(dim=[0, 1])
                va = a[tr["q_start"]:tr["q_end"],
                       tr["vis_start"]:tr["vis_end"]].mean(dim=0).cpu().float().numpy()
            else:
                va = np.zeros(tr["vis_end"] - tr["vis_start"])
            img_res[q] = va
            del attn, a, out, inputs
            torch.cuda.empty_cache()
            print(".", end="", flush=True)
        results.setdefault(gk, {})[fname] = img_res
        print(f" vram={torch.cuda.mem_get_info()[0]/1e9:.2f}GB")

# ── Stats ──
print(f"\n{'='*50}")
print(f"{'Group':<22s}  Within-Face    Face-vs-Ctrl    Ctrl-Ctrl")
print("  " + "-"*58)
stats = {}
for gk in LAYER_GROUPS:
    all_f, all_fc, all_cc = [], [], []
    for fname in images:
        hm = {q: results[gk][fname][q] for q in QUERIES}; nq = len(QUERIES)
        sim = np.zeros((nq,nq))
        for i in range(nq):
            for j in range(nq):
                r,_ = pearsonr(hm[QUERIES[i]], hm[QUERIES[j]]); sim[i,j] = r
        all_f.extend([sim[i,j] for i in range(6) for j in range(i+1,6)])
        all_fc.extend([sim[i,j] for i in range(6) for j in range(6,8)])
        all_cc.append(sim[6,7])
    fm, fs = float(np.mean(all_f)), float(np.std(all_f))
    fcm, fcs = float(np.mean(all_fc)), float(np.std(all_fc))
    ccm = float(np.mean(all_cc))
    print(f"  {gk:<22s}  {fm:.4f}+/-{fs:.4f}  {fcm:.4f}+/-{fcs:.4f}  {ccm:.4f}")
    stats[gk] = {"within_face_mean":round(fm,4),"within_face_std":round(fs,4),
                 "face_vs_ctrl_mean":round(fcm,4),"face_vs_ctrl_std":round(fcs,4),
                 "ctrl_ctrl_mean":round(ccm,4)}
with open(os.path.join(OUT_DIR,"attention_stats.json"),"w") as f: json.dump(stats,f,indent=2)

# ── Figures ──
gks = list(LAYER_GROUPS.keys())
for fname in images:
    base = fname.replace(".png","")
    for gk in gks:
        hm = {q: results[gk][fname][q] for q in QUERIES}; nq = len(QUERIES)
        sim = np.zeros((nq,nq))
        for i in range(nq):
            for j in range(nq):
                r,_ = pearsonr(hm[QUERIES[i]], hm[QUERIES[j]]); sim[i,j] = r
        fig,ax = plt.subplots(figsize=(9,7)); fig.patch.set_facecolor("white")
        im = ax.imshow(sim, cmap="viridis",vmin=0.5,vmax=1.0)
        ax.set_xticks(range(nq)); ax.set_yticks(range(nq))
        ax.set_xticklabels(QUERIES,rotation=45,ha="right",fontsize=9)
        ax.set_yticklabels(QUERIES,fontsize=9)
        for i in range(nq):
            for j in range(nq):
                ax.text(j,i,f"{sim[i,j]:.4f}",ha="center",va="center",fontsize=6.5)
        plt.colorbar(im,ax=ax,label="Pearson r")
        ax.set_title(f"[{gk}]",fontsize=13,fontweight="bold")
        plt.tight_layout()
        short = gk.split(" (")[0].replace(" ","_")
        plt.savefig(os.path.join(OUT_DIR,f"sim_{base}_{short}.png"),dpi=150); plt.close()

# 4-panel
fig,axes = plt.subplots(2,2,figsize=(18,14)); fig.patch.set_facecolor("white")
for idx,gk in enumerate(gks):
    ax = axes.flatten()[idx]; sim_sum=None; cnt=0
    for fname in images:
        hm = {q: results[gk][fname][q] for q in QUERIES}; nq=len(QUERIES)
        sim = np.zeros((nq,nq))
        for i in range(nq):
            for j in range(nq):
                r,_ = pearsonr(hm[QUERIES[i]],hm[QUERIES[j]]); sim[i,j]=r
        sim_sum = sim if sim_sum is None else sim_sum+sim; cnt+=1
    sim_avg = sim_sum/cnt
    im = ax.imshow(sim_avg,cmap="viridis",vmin=0.5,vmax=1.0)
    ax.set_xticks(range(nq)); ax.set_yticks(range(nq))
    ax.set_xticklabels(QUERIES,rotation=45,ha="right",fontsize=8)
    ax.set_yticklabels(QUERIES,fontsize=8)
    for i in range(nq):
        for j in range(nq):
            ax.text(j,i,f"{sim_avg[i,j]:.3f}",ha="center",va="center",fontsize=6)
    rect = plt.Rectangle((-0.5,-0.5),6,6,fill=False,edgecolor="cyan",linewidth=2,linestyle="--")
    ax.add_patch(rect); ax.set_title(gk,fontsize=13,fontweight="bold")
    plt.colorbar(im,ax=ax,fraction=0.046,label="Pearson r")
plt.suptitle("LLaVA-NeXT: Layer-wise Attention Similarity",fontsize=15,fontweight="bold",y=0.98)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"layer_ablation_4panel.png"),dpi=150); plt.close()

# Bar chart
fm = [stats[gk]["within_face_mean"] for gk in gks]
fs_ = [stats[gk]["within_face_std"] for gk in gks]
fcm = [stats[gk]["face_vs_ctrl_mean"] for gk in gks]
fcs_ = [stats[gk]["face_vs_ctrl_std"] for gk in gks]
fig,ax = plt.subplots(figsize=(10,5.5)); fig.patch.set_facecolor("white")
x = np.arange(len(gks)); w = 0.35
b1 = ax.bar(x-w/2,fm,w,yerr=fs_,capsize=4,label="Within-face",color="#4A6FA5")
b2 = ax.bar(x+w/2,fcm,w,yerr=fcs_,capsize=4,label="Face vs Control",color="#D16B6B")
ax.set_xticks(x); ax.set_xticklabels(gks,fontsize=10); ax.set_ylim(0.5,1.0)
ax.set_ylabel("Pearson r",fontsize=12); ax.legend(fontsize=11)
ax.set_title("LLaVA-NeXT: Cross-Query Similarity",fontsize=14,fontweight="bold")
ax.grid(axis="y",alpha=0.3); ax.set_axisbelow(True)
for b,v in zip(b1,fm): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.003,f"{v:.4f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
for b,v in zip(b2,fcm): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.003,f"{v:.4f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"layer_ablation_bars.png"),dpi=150); plt.close()

print(f"\nAll saved to {OUT_DIR}/")
print("DONE!")
