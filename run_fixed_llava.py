"""LLaVA-NeXT fixed: specific token extraction. Disable anyres properly."""
import os,sys,math,json,torch,numpy as np
from PIL import Image,ImageDraw
from scipy.stats import pearsonr
import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
sys.path.insert(0,"/root/Qwen3-VL");os.environ["TRANSFORMERS_OFFLINE"]="1";os.environ["HF_HUB_OFFLINE"]="1"
OUT="/root/Qwen3-VL/paper/figures/fixed_test";os.makedirs(OUT,exist_ok=True)
LAYERS=[17,18,19]; GRID_DIM=24

from transformers import LlavaNextForConditionalGeneration,AutoProcessor,BitsAndBytesConfig
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4")
model=LlavaNextForConditionalGeneration.from_pretrained("/root/Qwen3-VL",
    quantization_config=bnb,device_map="auto",attn_implementation="eager",
    local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16)
# Disable anyres: single base resolution
model.config.image_grid_pinpoints = [[336, 336]]
proc=AutoProcessor.from_pretrained("/root/Qwen3-VL",local_files_only=True,trust_remote_code=True)
proc.chat_template=("{{ bos_token }}{% for message in messages %}"
    "{% if (message['role']=='user')!=(loop.index0%2==0)%}{{ raise_exception('')}}{% endif %}"
    "{% if message['role']=='user'%}{{'[INST] '+message['content']|trim+' [/INST]'}}"
    "{% elif message['role']=='assistant'%}{{' '+message['content']|trim+' '+eos_token}}{% endif %}{% endfor %}")

lm=model.model.language_model
for l in LAYERS:lm.layers[l].self_attn.output_attentions=True

def extract(img,query_text,keyword):
    """Extract attention from specific keyword token. Pre-resize to avoid anyres."""
    rimg=img.resize((336,336),Image.LANCZOS)
    msg=[{"role":"user","content":f"<image>\ndescribe the {query_text}"}]
    pr=proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
    inp=proc(text=pr,images=rimg,return_tensors="pt").to("cuda")
    ids=inp["input_ids"][0]
    ip=(ids==32000).nonzero(as_tuple=True)[0]
    vs,ve=int(ip[0]),int(ip[-1])+1
    nvt=ve-vs  # Actual visual token count
    q_toks=ids[ve:];tok_strs=proc.tokenizer.convert_ids_to_tokens(q_toks.tolist())
    
    # Find keyword token
    spec_pos=[]
    if keyword:
        for j,t in enumerate(tok_strs):
            if keyword.lower() in t.lower().replace("Ġ","").replace("▁",""):
                spec_pos.append(ve+j)
    
    if nvt < GRID_DIM*GRID_DIM:
        print(f"  WARNING: nvt={nvt} < {GRID_DIM*GRID_DIM}, cannot form {GRID_DIM}x{GRID_DIM} grid!")
    
    at=[];hs=[lm.layers[l].self_attn.register_forward_hook(
        lambda m,i,o:at.append(o[1].detach())if isinstance(o,tuple)and len(o)>1 and o[1] is not None else None)
        for l in LAYERS]
    with torch.no_grad():out=model(**inp)
    for h in hs:h.remove()
    
    result={"tokens":tok_strs,"spec_pos":spec_pos,"nvt":nvt}
    if at:
        a=torch.stack(at).mean(dim=0).mean(dim=[0,1])
        spec=a[spec_pos,vs:ve].mean(dim=0).cpu().float().numpy()if spec_pos else np.zeros(nvt)
        last=a[-1,vs:ve].cpu().float().numpy()
        avg=a[ve:,vs:ve].mean(dim=0).cpu().float().numpy()
        result.update({"spec":spec,"last":last,"avg":avg})
        # Print nvt for first query to verify
        if query_text=="nose" or query_text=="red area":
            print(f"  nvt={nvt}, expected={GRID_DIM*GRID_DIM}, using first {min(nvt,GRID_DIM*GRID_DIM)} tokens")
    del at,a,out,inp;torch.cuda.empty_cache()
    return result

def to_grid(va):
    """Take first 576 tokens for 24x24 grid."""
    n=min(len(va),GRID_DIM*GRID_DIM)
    return va[:n].reshape(GRID_DIM,GRID_DIM)

# ═══ FACE ═══
FACE_Q=[("nose","nose"),("eyes","eye"),("mouth","mouth"),("chin","chin"),
        ("forehead","forehead"),("beard","beard"),("whole image","image"),("bottom-left area","area")]
IMG_DIR="/root/output_phase1_test"
images=sorted([f for f in os.listdir(IMG_DIR)if f.endswith((".png",".jpg",".jpeg"))and f!="grid_9views.png"])[:3]

print("="*60+"\nLLaVA Face\n"+"="*60)
face_r={}
for fname in images:
    img=Image.open(os.path.join(IMG_DIR,fname)).convert("RGB")
    print(f"\n[Face]{fname}")
    face_r[fname]={}
    for q_text,keyword in FACE_Q:
        r=extract(img,q_text,keyword)
        face_r[fname][q_text]=r
        if fname==images[0]:
            print(f"  [{q_text:20s}] toks={r['tokens']} kw@{r['spec_pos']}")

face_ql=[q[0] for q in FACE_Q]
for method in ["spec","last","avg"]:
    af,afc=[],[]
    for fname in images:
        nq=len(face_ql);sim=np.zeros((nq,nq))
        for i in range(nq):
            for j in range(nq):
                v1=to_grid(face_r[fname][face_ql[i]][method]).flatten()
                v2=to_grid(face_r[fname][face_ql[j]][method]).flatten()
                r,_=pearsonr(v1,v2);sim[i,j]=r
        af.extend([sim[i,j] for i in range(6)for j in range(i+1,6)])
        afc.extend([sim[i,j] for i in range(6)for j in range(6,8)])
    fm,fcm=np.mean(af),np.mean(afc)
    print(f"  {method}: within-face={fm:.4f} face-ctrl={fcm:.4f}")
    if method=="spec":
        print("  Peaks (spec, 24x24):")
        for q in face_ql:
            hm=to_grid(face_r[images[0]][q]["spec"]);pk=np.unravel_index(hm.argmax(),hm.shape)
            print(f"    [{q:20s}] peak={pk} val={hm[pk]:.6f}")

# ═══ DOT ═══
print(f"\n{'='*60}\nLLaVA Dot\n{'='*60}")
SIZE,CELL=256,16
img=Image.new("RGB",(SIZE,SIZE),(0,0,0));draw=ImageDraw.Draw(img)
blocks={"red":(1,1),"green":(7,7),"blue":(14,14)}
for n,(gx,gy) in blocks.items():
    x,y=gx*CELL,gy*CELL;draw.rectangle([x,y,x+CELL-1,y+CELL-1],fill={"red":(255,0,0),"green":(0,255,0),"blue":(0,0,255)}[n])

DOT_Q=[("red area","red"),("green area","green"),("blue area","blue"),("image",None)]
dot_r={m:{} for m in ["spec","last","avg"]}
for q_text,keyword in DOT_Q:
    r=extract(img,q_text,keyword)
    for m in ["spec","last","avg"]:dot_r[m][q_text]=r[m]

print("\n--- Dot Peaks (24x24) ---")
for method in ["spec","last","avg"]:
    print(f"  {method}:")
    for q_text,_ in DOT_Q:
        hm=to_grid(dot_r[method][q_text])
        pk=np.unravel_index(hm.argmax(),hm.shape)
        print(f"    [{q_text:15s}] peak={pk} val={hm[pk]:.6f}")

print("\n--- Dot Cross-r (24x24) ---")
for method in ["spec","last","avg"]:
    keys=[k for k in dot_r[method] if np.sum(dot_r[method][k])>1e-10]
    print(f"  {method}:")
    for i,k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            v1=to_grid(dot_r[method][k1]).flatten()
            v2=to_grid(dot_r[method][k2]).flatten()
            r,_=pearsonr(v1,v2)
            print(f"    {k1:15s} vs {k2:15s}: r={r:.4f}")

print(f"\nDONE!")
