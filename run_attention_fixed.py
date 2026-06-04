"""
Fixed: Extract attention from SPECIFIC feature-word token (not last, not avg).
Also compares with last-token and avg methods. Saves all results as JSON.
Runs: Qwen3-VL face features + Qwen3-VL dot test
"""
import os, sys, math, json, torch, numpy as np
from PIL import Image, ImageDraw
from scipy.stats import pearsonr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/root/Qwen3-VL")
os.environ["TRANSFORMERS_OFFLINE"] = "1"; os.environ["HF_HUB_OFFLINE"] = "1"
OUT = "/root/Qwen3-VL/paper/figures/fixed_test"; os.makedirs(OUT, exist_ok=True)
LAYERS = [17, 18, 19]

# ═══════════════════════════════════════════════
#  Face Feature Experiment
# ═══════════════════════════════════════════════

def run_face():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct",torch_dtype=torch.bfloat16,
        device_map="cuda",attn_implementation="eager",
        local_files_only=True,trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct",local_files_only=True,trust_remote_code=True)
    lm = model.model.language_model
    for l in LAYERS: lm.layers[l].self_attn.output_attentions = True

    QUERIES = [
        ("nose", "nose"), ("eyes", "eye"), ("mouth", "mouth"),
        ("chin", "chin"), ("forehead", "forehead"), ("beard", "beard"),
        ("whole image", "image"), ("bottom-left area", "area"),
    ]
    IMG_DIR = "/root/output_phase1_test"
    images = sorted([f for f in os.listdir(IMG_DIR)
                    if f.endswith((".png",".jpg",".jpeg")) and f!="grid_9views.png"])[:3]

    results = {"spec": {}, "last": {}, "avg": {}}

    for fname in images:
        img = Image.open(os.path.join(IMG_DIR,fname)).convert("RGB")
        print(f"\n[Face] {fname}")
        for q_text, keyword in QUERIES:
            msg = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":f"describe the {q_text}"}]}]
            pr = proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
            inp = proc(text=pr,images=img,return_tensors="pt").to("cuda")
            ids = inp["input_ids"][0]
            ve = (ids==151653).nonzero(as_tuple=True)[0].item()
            ie = (ids==151645).nonzero(as_tuple=True)[0].item()
            q_toks = ids[ve+1:ie]
            tok_strs = proc.tokenizer.convert_ids_to_tokens(q_toks.tolist())
            
            # Find keyword token(s)
            spec_positions = []
            for j, t in enumerate(tok_strs):
                if keyword.lower() in t.lower().replace("Ġ","").replace("▁",""):
                    spec_positions.append(ve+1+j)
            
            if fname == images[0]:
                print(f"  [{q_text:20s}] toks={tok_strs}  keyword='{keyword}'@{spec_positions}")
            
            at = []; hs = [lm.layers[l].self_attn.register_forward_hook(
                lambda m,i,o: at.append(o[1].detach()) if isinstance(o,tuple)and len(o)>1 and o[1] is not None else None)
                for l in LAYERS]
            with torch.no_grad(): out = model(**inp)
            for h in hs: h.remove()
            
            if at:
                a = torch.stack(at).mean(dim=0).mean(dim=[0,1])
                nvt = ve - 4
                # SPECIFIC token(s)
                if spec_positions:
                    spec = a[spec_positions, 4:ve].mean(dim=0).cpu().float().numpy()
                else:
                    spec = np.zeros(nvt)
                    print(f"    WARNING: keyword '{keyword}' not found in tokens!")
                # LAST token
                last = a[ie-1, 4:ve].cpu().float().numpy()
                # AVG
                avg = a[ve+1:ie, 4:ve].mean(dim=0).cpu().float().numpy()
                
                for mk, mv in [("spec",spec),("last",last),("avg",avg)]:
                    results[mk].setdefault(fname, {})[q_text] = mv
            del at, a, out, inp; torch.cuda.empty_cache()
    
    del model, lm; torch.cuda.empty_cache()
    return results, [q[0] for q in QUERIES], images


# ═══════════════════════════════════════════════
#  Dot Test
# ═══════════════════════════════════════════════

def run_dot():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct",torch_dtype=torch.bfloat16,
        device_map="cuda",attn_implementation="eager",
        local_files_only=True,trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(
        "/root/Qwen3-VL-2B-Instruct",local_files_only=True,trust_remote_code=True)
    lm = model.model.language_model
    for l in LAYERS: lm.layers[l].self_attn.output_attentions = True

    SIZE, CELL = 256, 16
    img = Image.new("RGB", (SIZE,SIZE), (0,0,0)); draw = ImageDraw.Draw(img)
    blocks = {"red":(1,1),"green":(7,7),"blue":(14,14)}
    for n,(gx,gy) in blocks.items():
        x,y=gx*CELL,gy*CELL; draw.rectangle([x,y,x+CELL-1,y+CELL-1],fill={"red":(255,0,0),"green":(0,255,0),"blue":(0,0,255)}[n])
    
    QUERIES = [("red area","red"),("green area","green"),("blue area","blue"),("image",None)]
    results = {"spec": {}, "last": {}, "avg": {}}
    
    for q_text, keyword in QUERIES:
        msg = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":f"describe the {q_text}"}]}]
        pr = proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
        inp = proc(text=pr,images=img,return_tensors="pt").to("cuda")
        ids = inp["input_ids"][0]
        ve = (ids==151653).nonzero(as_tuple=True)[0].item()
        ie = (ids==151645).nonzero(as_tuple=True)[0].item()
        q_toks = ids[ve+1:ie]; tok_strs = proc.tokenizer.convert_ids_to_tokens(q_toks.tolist())
        
        spec_positions = []
        if keyword:
            for j,t in enumerate(tok_strs):
                if keyword.lower() in t.lower().replace("Ġ","").replace("▁",""):
                    spec_positions.append(ve+1+j)
        print(f"\n[Dot] '{q_text}': toks={tok_strs}  keyword='{keyword}'@{spec_positions}")
        
        at = []; hs = [lm.layers[l].self_attn.register_forward_hook(
            lambda m,i,o: at.append(o[1].detach()) if isinstance(o,tuple)and len(o)>1 and o[1] is not None else None)
            for l in LAYERS]
        with torch.no_grad(): out = model(**inp)
        for h in hs: h.remove()
        
        if at:
            a = torch.stack(at).mean(dim=0).mean(dim=[0,1])
            nvt = ve - 4
            spec = a[spec_positions, 4:ve].mean(dim=0).cpu().float().numpy() if spec_positions else np.zeros(nvt)
            last = a[ie-1, 4:ve].cpu().float().numpy()
            avg = a[ve+1:ie, 4:ve].mean(dim=0).cpu().float().numpy()
            for mk, mv in [("spec",spec),("last",last),("avg",avg)]:
                results[mk][q_text] = mv
            results.setdefault("nvt", nvt)
        del at, a, out, inp; torch.cuda.empty_cache()
    
    del model, lm; torch.cuda.empty_cache()
    return results, [q[0] for q in QUERIES], blocks


# ═══════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════

print("="*60)
print("FACE FEATURE EXPERIMENT")
print("="*60)
face_r, face_q, face_imgs = run_face()

# Stats
all_stats = {}
for method in ["spec", "last", "avg"]:
    all_f, all_fc = [], []
    for fname in face_imgs:
        hm = face_r[method][fname]; nq = len(face_q)
        sim = np.zeros((nq,nq))
        for i in range(nq):
            for j in range(nq):
                r,_ = pearsonr(hm[face_q[i]], hm[face_q[j]]); sim[i,j] = r
        all_f.extend([sim[i,j] for i in range(6) for j in range(i+1,6)])
        all_fc.extend([sim[i,j] for i in range(6) for j in range(6,8)])
    fm, fs = round(float(np.mean(all_f)),4), round(float(np.std(all_f)),4)
    fcm, fcs = round(float(np.mean(all_fc)),4), round(float(np.std(all_fc)),4)
    all_stats[method] = {"within_face_mean":fm,"within_face_std":fs,
                         "face_vs_ctrl_mean":fcm,"face_vs_ctrl_std":fcs}
    print(f"\n{method}:")
    print(f"  Within-face: {fm} +/- {fs}")
    print(f"  Face-vs-ctrl: {fcm} +/- {fcs}")

# Save face JSON
face_data = {m: {fname: {q: face_r[m][fname][q].tolist() for q in face_q} for fname in face_imgs}
             for m in ["spec","last","avg"]}
face_data["stats"] = all_stats
with open(f"{OUT}/face_results.json","w") as f: json.dump(face_data, f, indent=2)
print(f"\nSaved: face_results.json")

# Face figure
fig,axes = plt.subplots(3,8,figsize=(20,7)); fig.patch.set_facecolor("white")
for qi,q in enumerate(face_q):
    for mi,m in enumerate(["spec","last","avg"]):
        va = face_r[m][face_imgs[0]][q]; s = int(math.isqrt(len(va)))
        hm = va.reshape(s,s) if s*s==len(va) else va
        axes[mi,qi].imshow(hm, cmap="hot",interpolation="nearest")
        axes[mi,qi].axis("off")
        if mi==0: axes[mi,qi].set_title(q,fontsize=8)
for mi,m in enumerate(["spec","last","avg"]):
    axes[mi,0].set_ylabel(m,fontsize=11,fontweight="bold")
plt.suptitle("Face: spec vs last vs avg",fontsize=13); plt.tight_layout()
plt.savefig(f"{OUT}/face_comparison.png",dpi=150); plt.close()

# ═══ DOT TEST ═══
print(f"\n{'='*60}")
print("DOT TEST")
print(f"{'='*60}")
dot_r, dot_q, dot_blocks = run_dot()

print(f"\n--- Peak positions ---")
for method in ["spec","last","avg"]:
    print(f"  {method}:")
    for q in dot_q:
        r = dot_r[method].get(q); nvt = dot_r.get("nvt",0)
        if r is not None:
            s = int(math.isqrt(len(r)))
            if s*s==len(r):
                hm = r.reshape(s,s); pk = np.unravel_index(hm.argmax(),hm.shape)
                print(f"    [{q:15s}] peak={pk} val={hm[pk]:.4f}")

print(f"\n--- Cross-r ---")
for method in ["spec","last","avg"]:
    keys = [q for q in dot_q if dot_r[method].get(q) is not None and dot_r[method][q].sum()>0]
    if len(keys)>=2:
        print(f"  {method}:")
        for i,q1 in enumerate(keys):
            for q2 in keys[i+1:]:
                r,_ = pearsonr(dot_r[method][q1], dot_r[method][q2])
                print(f"    {q1:15s} vs {q2:15s}: r={r:.4f}")

# Save dot JSON
dot_data = {m: {q: dot_r[m][q].tolist() for q in dot_q if dot_r[m].get(q) is not None}
            for m in ["spec","last","avg"]}
dot_data["nvt"] = dot_r.get("nvt", 0)
with open(f"{OUT}/dot_results.json","w") as f: json.dump(dot_data, f, indent=2)
print(f"Saved: dot_results.json")

# Dot figure
fig,axes = plt.subplots(3,4,figsize=(14,9)); fig.patch.set_facecolor("white")
for qi,q in enumerate(dot_q):
    for mi,m in enumerate(["spec","last","avg"]):
        r = dot_r[m].get(q); nvt = dot_r.get("nvt",0)
        if r is not None:
            s = int(math.isqrt(len(r)))
            hm = r.reshape(s,s) if s*s==len(r) else r
            axes[mi,qi].imshow(hm, cmap="hot",interpolation="nearest")
            if q in dot_blocks:
                gy,gx = dot_blocks[q]
                axes[mi,qi].plot(gx,gy,'c*',ms=12,mec='white',mew=1.5)
        axes[mi,qi].axis("off")
        if mi==0: axes[mi,qi].set_title(q,fontsize=9)
for mi,m in enumerate(["spec","last","avg"]):
    axes[mi,0].set_ylabel(m,fontsize=11,fontweight="bold")
plt.suptitle("Dot: spec(last) vs last vs avg",fontsize=13); plt.tight_layout()
plt.savefig(f"{OUT}/dot_comparison.png",dpi=150); plt.close()

print(f"\n{'='*60}\nDONE! All results in {OUT}/")
