"""
Skip-connection Proof-of-Concept (Plan A)
Gated fusion: fused = L28 + g * (L18_20 - L28), g = sigmoid(learned)
Compares: L28 only | L25-27 only | L18-20 only | Gated(L28+L18-20)
"""
import os, sys, gc, math, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, '/root/Qwen3-VL')
os.environ['TRANSFORMERS_OFFLINE'] = '1'; os.environ['HF_HUB_OFFLINE'] = '1'

BASE = "/mnt/c/Users/张俊豪/OneDrive/Desktop/aflw2000_data/aflw2000_data"
IMG_DIR = os.path.join(BASE, "AFLW2000-3D_crop")
EVAL_DIR = os.path.join(BASE, "eval")
LIST_FILE = os.path.join(BASE, "AFLW2000-3D_crop.list")
OUT_DIR = "/root/Qwen3-VL/paper/figures/fixed_test"
os.makedirs(OUT_DIR, exist_ok=True)

N_IMAGES = 500; TRAIN_RATIO = 0.8
TRAIN_FRACTIONS = [0.1, 0.25, 0.5, 1.0]
HIDDEN_DIM = 2048; GRID_SZ = 8; HEATMAP_SZ = 32; N_LANDMARKS = 68
BATCH_SIZE = 32; LR = 5e-4; EPOCHS = 100; PATIENCE = 15; DEVICE = "cuda"
QUERY_LANDMARKS = {"nose":[30],"eyes":list(range(36,48)),"mouth":list(range(48,68)),
    "chin":[8],"forehead":list(range(17,27)),"beard":[7,8,9]}
MID_LAYERS = [17,18,19]; NEAR_LAST = [24,25,26]; LAST_LAYER = [27]

def get_landmark_positions(pts, roi_box, img_idx):
    x_orig=pts[img_idx,0,:]; y_orig=pts[img_idx,1,:]
    l,t,r,b=roi_box[img_idx]; cw,ch=r-l,b-t
    if cw<=0 or ch<=0: return np.zeros((N_LANDMARKS,2),dtype=np.float32)
    xc=(x_orig-l)*120/cw; yc=(y_orig-t)*120/ch
    xh=np.clip(xc*HEATMAP_SZ/120,0,HEATMAP_SZ-1)
    yh=np.clip(yc*HEATMAP_SZ/120,0,HEATMAP_SZ-1)
    return np.stack([xh,yh],-1).astype(np.float32)

class SingleDataset(Dataset):
    def __init__(self,f,c): self.f=f.cpu(); self.c=c.cpu()
    def __len__(self): return self.f.size(0)
    def __getitem__(self,i): return self.f[i],self.c[i]
class DualDataset(Dataset):
    def __init__(self,fm,fl,c): self.fm=fm.cpu(); self.fl=fl.cpu(); self.c=c.cpu()
    def __len__(self): return self.fm.size(0)
    def __getitem__(self,i): return self.fm[i],self.fl[i],self.c[i]

class CProbe(nn.Module):
    def __init__(self,idim=HIDDEN_DIM,gs=GRID_SZ,nl=N_LANDMARKS):
        super().__init__(); self.gs=gs; self.nl=nl
        self.net=nn.Sequential(
            nn.Conv2d(idim,256,3,padding=1),nn.ReLU(True),nn.BatchNorm2d(256),
            nn.Conv2d(256,128,3,padding=1),nn.ReLU(True),nn.BatchNorm2d(128),
            nn.Flatten(),nn.Linear(gs*gs*128,256),nn.ReLU(True),
            nn.Linear(256,nl*2))
    def forward(self,x):
        B=x.shape[0]; x=x.view(B,self.gs,self.gs,-1).permute(0,3,1,2)
        return self.net(x).view(B,self.nl,2)

class GatedProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_logit=nn.Parameter(torch.tensor(1.0))
        self.probe=CProbe()
    def forward(self,fm,fl):
        g=torch.sigmoid(self.gate_logit)
        return self.probe((1-g)*fl+g*fm),g

def compute_pck(pred,gt,ths=[0.03,0.05,0.1,0.2]):
    d=torch.sqrt((pred[:,:,0]-gt[:,:,0])**2+(pred[:,:,1]-gt[:,:,1])**2)
    fd=math.sqrt(HEATMAP_SZ**2+HEATMAP_SZ**2)
    return {str(t):round(100*(d<t*fd).float().mean().item(),1) for t in ths},d

def extract_features():
    print("="*60); print("STAGE 1: Extracting hidden states"); print("="*60)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model=AutoModelForImageTextToText.from_pretrained('/root/Qwen3-VL-2B-Instruct',
        torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager',
        local_files_only=True,trust_remote_code=True)
    processor=AutoProcessor.from_pretrained('/root/Qwen3-VL-2B-Instruct',
        local_files_only=True,trust_remote_code=True)
    lm=model.model.language_model
    with open(LIST_FILE) as f: all_names=[l.strip() for l in f if l.strip()]
    step=max(1,len(all_names)//N_IMAGES)
    si=list(range(0,len(all_names),step))[:N_IMAGES]
    img_names=[all_names[i] for i in si]
    pts68=np.load(EVAL_DIR+"/AFLW2000-3D.pts68.npy")
    roi=np.load(EVAL_DIR+"/AFLW2000-3D_crop.roi_box.npy")
    fm_list,fn_list,fl_list,coords_list=[],[],[],[]
    t0=time.time(); target_set=set(MID_LAYERS+NEAR_LAST+LAST_LAYER)
    for idx,iname in enumerate(img_names):
        if idx%50==0:
            e=time.time()-t0; eta=e/max(1,idx)*(N_IMAGES-idx) if idx>0 else 0
            print(f"  [{idx}/{N_IMAGES}] elapsed={e:.0f}s eta={eta:.0f}s")
        ip=os.path.join(IMG_DIR,iname)
        if not os.path.exists(ip): continue
        img=Image.open(ip).convert("RGB")
        msg=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"describe this image"}]}]
        pr=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
        inp=processor(text=pr,images=img,return_tensors='pt').to('cuda')
        ids=inp['input_ids'][0]; ve=(ids==151653).nonzero(as_tuple=True)[0].item(); nvt=ve-4
        collected={}; hooks=[]
        for l in sorted(target_set):
            def mk(lid):
                def hk(m,i,o):
                    out=o[0] if isinstance(o,tuple) else o
                    collected[lid]=out[:,4:4+nvt].detach().cpu().float()
                return hk
            hooks.append(lm.layers[l].register_forward_hook(mk(l)))
        with torch.no_grad(): model(**inp)
        for h in hooks: h.remove()
        mid=torch.stack([collected[l] for l in MID_LAYERS]).mean(0).squeeze(0)
        near=torch.stack([collected[l] for l in NEAR_LAST]).mean(0).squeeze(0)
        last=collected[LAST_LAYER[0]].squeeze(0)
        fm_list.append(mid); fn_list.append(near); fl_list.append(last)
        coords_list.append(torch.from_numpy(get_landmark_positions(pts68,roi,si[idx])))
        del collected,inp; torch.cuda.empty_cache()
    fmid=torch.stack(fm_list); fnear=torch.stack(fn_list); flast=torch.stack(fl_list)
    coords=torch.stack(coords_list)
    del model,lm,processor; gc.collect(); torch.cuda.empty_cache()
    print(f"  Done. Time: {time.time()-t0:.0f}s")
    return fmid,fnear,flast,coords

def train_probe(tr_ld,va_ld,gated=False,device=DEVICE):
    model=GatedProbe().to(device) if gated else CProbe().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=8,factor=0.5)
    best_loss=float('inf'); best_state=None; best_gate=None; pc=0
    for ep in range(EPOCHS):
        model.train(); tl=0.0
        if gated:
            for fm,fl,cgt in tr_ld:
                fm,fl,cgt=fm.to(device),fl.to(device),cgt.to(device)
                p,g=model(fm,fl); l=F.mse_loss(p,cgt)
                opt.zero_grad(); l.backward(); opt.step(); tl+=l.item()*fm.size(0)
        else:
            for f,cgt in tr_ld:
                f,cgt=f.to(device),cgt.to(device); p=model(f)
                l=F.mse_loss(p,cgt); opt.zero_grad(); l.backward(); opt.step()
                tl+=l.item()*f.size(0)
        tl/=len(tr_ld.dataset)
        model.eval(); vl=0.0; ap,ag=[],[]
        with torch.no_grad():
            if gated:
                for fm,fl,cgt in va_ld:
                    fm,fl,cgt=fm.to(device),fl.to(device),cgt.to(device)
                    p,g=model(fm,fl); l=F.mse_loss(p,cgt); vl+=l.item()*fm.size(0)
                    ap.append(p.cpu()); ag.append(cgt.cpu())
            else:
                for f,cgt in va_ld:
                    f,cgt=f.to(device),cgt.to(device); p=model(f)
                    l=F.mse_loss(p,cgt); vl+=l.item()*f.size(0)
                    ap.append(p.cpu()); ag.append(cgt.cpu())
        vl/=len(va_ld.dataset); sched.step(vl)
        ap=torch.cat(ap,0); ag=torch.cat(ag,0)
        pck,_=compute_pck(ap,ag)
        cg=torch.sigmoid(model.gate_logit).item() if gated else None
        if (ep+1)%10==0:
            gs=f" gate={cg:.3f}" if gated else ""
            print(f"    Ep {ep+1:3d}: tr_l={tl:.6f} val_l={vl:.6f} PCK@0.05={pck['0.05']:.1f}%{gs}")
        if vl<best_loss:
            best_loss=vl; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; pc=0
            if gated: best_gate=cg
        else:
            pc+=1
            if pc>=PATIENCE: print(f"    Early stop at ep {ep+1}"); break
    model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
    model.eval(); ap,ag=[],[]
    with torch.no_grad():
        if gated:
            for fm,fl,cgt in va_ld:
                fm,fl,cgt=fm.to(device),fl.to(device),cgt.to(device)
                ap.append(model(fm,fl)[0].cpu()); ag.append(cgt.cpu())
        else:
            for f,cgt in va_ld:
                f,cgt=f.to(device),cgt.to(device)
                ap.append(model(f).cpu()); ag.append(cgt.cpu())
    ap=torch.cat(ap,0); ag=torch.cat(ag,0)
    pck,df=compute_pck(ap,ag)
    qp={}
    for qn,li in QUERY_LANDMARKS.items():
        li_t=torch.tensor(li)
        d=torch.sqrt((ap[:,li_t,0]-ag[:,li_t,0])**2+(ap[:,li_t,1]-ag[:,li_t,1])**2)
        qp[qn]=round(100*(d<0.2*math.sqrt(HEATMAP_SZ**2+HEATMAP_SZ**2)).float().mean().item(),1)
    r={"best_val_loss":round(best_loss,6),"best_epoch":ep+1,
       "pck":{str(k):v for k,v in pck.items()},"mean_distance_heatmap_px":round(df.mean().item(),2),
       "query_pck":qp}
    if gated: r["learned_gate"]=round(best_gate,4) if best_gate is not None else None
    return r

def main():
    print("="*60); print("SKIP-CONNECTION PROOF-OF-CONCEPT"); print("="*60)
    fmid,fnear,flast,coords=extract_features()
    N=fmid.size(0); idxs=np.random.RandomState(42).permutation(N)
    n_tr=int(N*TRAIN_RATIO); ti,vi=idxs[:n_tr],idxs[n_tr:]
    print(f"Train: {n_tr}, Val: {N-n_tr}")
    conds=[("L28 (baseline)",flast,False),("L25-27",fnear,False),
           ("L18-20 (upper bound)",fmid,False),("Gated: L28+g*(L18-20-L28)",None,True)]
    all_r={}
    for name,feat,mode in conds:
        print(f"\n--- {name} ---")
        lr={}
        for frac in TRAIN_FRACTIONS:
            nf=max(1,int(n_tr*frac)); fti=ti[:nf]
            if mode:
                fmt=fmid[fti].float().clone(); fmv=fmid[vi].float().clone()
                flt=flast[fti].float().clone(); flv=flast[vi].float().clone()
                mm=fmt.mean((0,1),keepdim=True); sm=fmt.std((0,1),keepdim=True)+1e-8
                ml=flt.mean((0,1),keepdim=True); sl=flt.std((0,1),keepdim=True)+1e-8
                ds_tr=DualDataset((fmt-mm)/sm,(flt-ml)/sl,coords[fti])
                ds_va=DualDataset((fmv-mm)/sm,(flv-ml)/sl,coords[vi])
            else:
                ft=feat[fti].float().clone(); fv=feat[vi].float().clone()
                m=ft.mean((0,1),keepdim=True); s=ft.std((0,1),keepdim=True)+1e-8
                ds_tr=SingleDataset((ft-m)/s,coords[fti])
                ds_va=SingleDataset((fv-m)/s,coords[vi])
            tr_dl=DataLoader(ds_tr,BATCH_SIZE,shuffle=True)
            va_dl=DataLoader(ds_va,BATCH_SIZE,shuffle=False)
            print(f"  Frac {frac*100:.0f}% ({nf} imgs)")
            r=train_probe(tr_dl,va_dl,gated=mode)
            lr[f"frac_{frac}"]=r
            gs=f", gate={r.get('learned_gate','N/A')}" if mode else ""
            print(f"  -> PCK@0.05={r['pck']['0.05']}% dist={r['mean_distance_heatmap_px']}px{gs}")
        all_r[name]=lr
    print("\n"+"="*80)
    print("PCK@0.05 x Sample Efficiency"); print("="*80)
    h=f"{'Condition':<40s}"+"".join([f"{f'Frac {int(f*100)}%':>10s}" for f in TRAIN_FRACTIONS])
    print(h); print("-"*len(h))
    for name,_,m in conds:
        r=f"{name:<40s}"
        for f in TRAIN_FRACTIONS: r+=f"{all_r[name][f'frac_{f}']['pck']['0.05']:>9.1f}%"
        gs=f" [g={all_r[name]['frac_1.0'].get('learned_gate','N/A')}]" if m else ""
        print(f"{r}{gs}")
    print("\n"+"="*80)
    print("Mean Distance (px) x Sample Efficiency"); print("="*80)
    print(h); print("-"*len(h))
    for name,_,_ in conds:
        r=f"{name:<40s}"
        for f in TRAIN_FRACTIONS: r+=f"{all_r[name][f'frac_{f}']['mean_distance_heatmap_px']:>9.2f}"
        print(r)
    sp=os.path.join(OUT_DIR,"skip_connection_results.json")
    with open(sp,"w") as f: json.dump({"results":all_r},f,indent=2)
    print(f"\nSaved: {sp}\nDone!")
if __name__=="__main__": main()
