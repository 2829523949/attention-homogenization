"""
Gate stability: run gated skip-connection with 3 seeds (42, 123, 456).
Only tests 10% and 100% fractions (most/least data).
"""
from PIL import Image
import os, sys, gc, math, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, '/root/Qwen3-VL')
os.environ['TRANSFORMERS_OFFLINE'] = '1'; os.environ['HF_HUB_OFFLINE'] = '1'

BASE="/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
IMG_DIR=os.path.join(BASE,"AFLW2000-3D_crop"); EVAL_DIR=os.path.join(BASE,"eval")
LIST_FILE=os.path.join(BASE,"AFLW2000-3D_crop.list")
OUT_DIR="/root/Qwen3-VL/paper/figures/fixed_test"; os.makedirs(OUT_DIR,exist_ok=True)
N_IMAGES=500; TRAIN_RATIO=0.8; HIDDEN_DIM=2048; GRID_SZ=8; HEATMAP_SZ=32
N_LANDMARKS=68; BATCH_SIZE=32; LR=5e-4; EPOCHS=100; PATIENCE=15; DEVICE="cuda"
MID_LAYERS=[17,18,19]; LAST_LAYER=[27]

class DualDataset(Dataset):
    def __init__(self,fm,fl,c): self.fm=fm.cpu(); self.fl=fl.cpu(); self.c=c.cpu()
    def __len__(self): return self.fm.size(0)
    def __getitem__(self,i): return self.fm[i],self.fl[i],self.c[i]

class GatedProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_logit=nn.Parameter(torch.tensor(1.0))
        self.net=nn.Sequential(
            nn.Conv2d(HIDDEN_DIM,256,3,padding=1),nn.ReLU(True),nn.BatchNorm2d(256),
            nn.Conv2d(256,128,3,padding=1),nn.ReLU(True),nn.BatchNorm2d(128),
            nn.Flatten(),nn.Linear(GRID_SZ*GRID_SZ*128,256),nn.ReLU(True),
            nn.Linear(256,N_LANDMARKS*2))
    def forward(self,fm,fl):
        g=torch.sigmoid(self.gate_logit)
        fused=(1-g)*fl+g*fm
        B=fused.shape[0]
        x=fused.view(B,GRID_SZ,GRID_SZ,-1).permute(0,3,1,2)
        return self.net(x).view(B,N_LANDMARKS,2),g

def get_landmark_positions(pts,roi_box,img_idx):
    x_orig=pts[img_idx,0,:]; y_orig=pts[img_idx,1,:]
    l,t,r,b=roi_box[img_idx]; cw,ch=r-l,b-t
    if cw<=0 or ch<=0: return np.zeros((N_LANDMARKS,2),dtype=np.float32)
    xc=(x_orig-l)*120/cw; yc=(y_orig-t)*120/ch
    xh=np.clip(xc*HEATMAP_SZ/120,0,HEATMAP_SZ-1)
    yh=np.clip(yc*HEATMAP_SZ/120,0,HEATMAP_SZ-1)
    return np.stack([xh,yh],-1).astype(np.float32)

def extract():
    print("Extracting features...")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model=AutoModelForImageTextToText.from_pretrained('/root/Qwen3-VL-2B-Instruct',
        torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager',
        local_files_only=True,trust_remote_code=True)
    processor=AutoProcessor.from_pretrained('/root/Qwen3-VL-2B-Instruct',
        local_files_only=True,trust_remote_code=True)
    lm=model.model.language_model
    with open(LIST_FILE) as f: all_names=[l.strip() for l in f if l.strip()]
    step=max(1,len(all_names)//N_IMAGES); si=list(range(0,len(all_names),step))[:N_IMAGES]
    img_names=[all_names[i] for i in si]
    pts68=np.load(EVAL_DIR+"/AFLW2000-3D.pts68.npy")
    roi=np.load(EVAL_DIR+"/AFLW2000-3D_crop.roi_box.npy")
    fm_list,fl_list,coords_list=[],[],[]; t0=time.time()
    for idx,iname in enumerate(img_names):
        if idx%50==0:
            e=time.time()-t0
            eta=e/max(1,idx)*(N_IMAGES-idx) if idx>0 else 0
            print(f"  [{idx}/{N_IMAGES}] {e:.0f}s eta={eta:.0f}s")
        ip=os.path.join(IMG_DIR,iname)
        if not os.path.exists(ip): continue
        img=Image.open(ip).convert("RGB")
        msg=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"describe this image"}]}]
        pr=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
        inp=processor(text=pr,images=img,return_tensors='pt').to('cuda')
        ids=inp['input_ids'][0]
        ve=(ids==151653).nonzero(as_tuple=True)[0].item(); nvt=ve-4
        collected={}; hooks=[]
        for l in sorted([17,18,19,27]):
            def mk(lid):
                def hk(m,i,o):
                    out=o[0] if isinstance(o,tuple) else o
                    collected[lid]=out[:,4:4+nvt].detach().cpu().float()
                return hk
            hooks.append(lm.layers[l].register_forward_hook(mk(l)))
        with torch.no_grad(): model(**inp)
        for h in hooks: h.remove()
        mid=torch.stack([collected[l] for l in [17,18,19]]).mean(0).squeeze(0)
        last=collected[27].squeeze(0)
        fm_list.append(mid); fl_list.append(last)
        coords_list.append(torch.from_numpy(get_landmark_positions(pts68,roi,si[idx])))
        del collected,inp; torch.cuda.empty_cache()
    fmid=torch.stack(fm_list); flast=torch.stack(fl_list); coords=torch.stack(coords_list)
    del model,lm,processor; gc.collect(); torch.cuda.empty_cache()
    print(f"Done. {time.time()-t0:.0f}s"); return fmid,flast,coords

def train_gated(mid,last,coords,ti,vi,frac,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    nf=max(1,int(len(ti)*frac)); fti=ti[:nf]
    fmt=mid[fti].float().clone(); fmv=mid[vi].float().clone()
    flt=last[fti].float().clone(); flv=last[vi].float().clone()
    mm=fmt.mean((0,1),keepdim=True); sm=fmt.std((0,1),keepdim=True)+1e-8
    ml=flt.mean((0,1),keepdim=True); sl=flt.std((0,1),keepdim=True)+1e-8
    ds_tr=DualDataset((fmt-mm)/sm,(flt-ml)/sl,coords[fti])
    ds_va=DualDataset((fmv-mm)/sm,(flv-ml)/sl,coords[vi])
    tr_dl=DataLoader(ds_tr,BATCH_SIZE,shuffle=True)
    va_dl=DataLoader(ds_va,BATCH_SIZE,shuffle=False)
    model=GatedProbe().to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=8,factor=0.5)
    best_loss=float('inf'); best_state=None; best_gate=None; pc=0
    for ep in range(EPOCHS):
        model.train(); tl=0.0
        for fm,fl,cgt in tr_dl:
            fm,fl,cgt=fm.to(DEVICE),fl.to(DEVICE),cgt.to(DEVICE)
            p,g=model(fm,fl); l=F.mse_loss(p,cgt)
            opt.zero_grad(); l.backward(); opt.step(); tl+=l.item()*fm.size(0)
        tl/=len(tr_dl.dataset)
        model.eval(); vl=0.0; ap,ag=[],[]
        with torch.no_grad():
            for fm,fl,cgt in va_dl:
                fm,fl,cgt=fm.to(DEVICE),fl.to(DEVICE),cgt.to(DEVICE)
                p,g=model(fm,fl); l=F.mse_loss(p,cgt)
                vl+=l.item()*fm.size(0); ap.append(p.cpu()); ag.append(cgt.cpu())
        vl/=len(va_dl.dataset); sched.step(vl)
        ap=torch.cat(ap,0); ag=torch.cat(ag,0)
        cg=torch.sigmoid(model.gate_logit).item()
        if (ep+1)%10==0: print(f"    Ep {ep+1:3d}: gate={cg:.3f}")
        if vl<best_loss: best_loss=vl; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; best_gate=cg; pc=0
        else: pc+=1
        if pc>=PATIENCE: break
    model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})
    model.eval(); ap,ag=[],[]
    with torch.no_grad():
        for fm,fl,cgt in va_dl:
            fm,fl,cgt=fm.to(DEVICE),fl.to(DEVICE),cgt.to(DEVICE)
            ap.append(model(fm,fl)[0].cpu()); ag.append(cgt.cpu())
    ap=torch.cat(ap,0); ag=torch.cat(ag,0)
    d=torch.sqrt((ap[:,:,0]-ag[:,:,0])**2+(ap[:,:,1]-ag[:,:,1])**2)
    fd=math.sqrt(HEATMAP_SZ**2+HEATMAP_SZ**2)
    pck=round(100*(d<0.05*fd).float().mean().item(),1)
    return {"gate":round(best_gate,4),"pck_0.05":pck,"mean_dist":round(d.mean().item(),2)}

# Main
SEEDS=[42,123,456]; FRACS=[0.1,1.0]
print("="*60); print("GATE STABILITY: 3 SEEDS x 2 FRACTIONS"); print("="*60)
fmid,flast,coords=extract()
N=fmid.size(0)

results={s:{} for s in SEEDS}
for seed in SEEDS:
    print(f"\n{'='*50}\nSeed={seed}\n{'='*50}")
    rng=np.random.RandomState(seed); idxs=rng.permutation(N)
    n_tr=int(N*TRAIN_RATIO); ti,vi=idxs[:n_tr],idxs[n_tr:]
    for frac in FRACS:
        print(f"\n--- Frac {int(frac*100)}% ---")
        r=train_gated(fmid,flast,coords,ti,vi,frac,seed)
        results[seed][f"frac_{frac}"]=r
        print(f"  Seed {seed}: gate={r['gate']:.4f}, PCK@0.05={r['pck_0.05']:.1f}%, dist={r['mean_dist']}px")

# Summary
print("\n"+"="*70)
print("GATE STABILITY SUMMARY")
print("="*70)
for frac in FRACS:
    gates=[results[s][f"frac_{frac}"]["gate"] for s in SEEDS]
    pcks=[results[s][f"frac_{frac}"]["pck_0.05"] for s in SEEDS]
    dists=[results[s][f"frac_{frac}"]["mean_dist"] for s in SEEDS]
    print(f"\nFrac {int(frac*100)}%:")
    print(f"  Gate:    {np.mean(gates):.4f} ± {np.std(gates):.4f}  (range [{min(gates):.4f}, {max(gates):.4f}])")
    print(f"  PCK:     {np.mean(pcks):.1f} ± {np.std(pcks):.1f}%")
    print(f"  Dist:    {np.mean(dists):.2f} ± {np.std(dists):.2f}px")
    print(f"  Values:  gates={gates}")

save={"seeds":SEEDS,"fractions":FRACS,"results":results,
      "gate_summary":{"0.1":{"mean":round(np.mean([results[s]["frac_0.1"]["gate"] for s in SEEDS]),4),
                             "std":round(np.std([results[s]["frac_0.1"]["gate"] for s in SEEDS]),4)},
                      "1.0":{"mean":round(np.mean([results[s]["frac_1.0"]["gate"] for s in SEEDS]),4),
                             "std":round(np.std([results[s]["frac_1.0"]["gate"] for s in SEEDS]),4)}}}
with open(os.path.join(OUT_DIR,"gate_stability_results.json"),"w") as fp:
    json.dump(save,fp,indent=2)
print(f"\nSaved: {os.path.join(OUT_DIR,'gate_stability_results.json')}\nDone!")
