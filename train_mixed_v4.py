"""
混合训练 v4: 真正多样化的文本 + 对比学习
=====================================================================
改进:
  1. gen_text_v2: 7种完全不同语法结构的句式 (非模板风)
  2. 对比学习: 同角度不同文本 → hidden state 余弦相似度 loss
  3. 模板权重 5→2, Pitch权重 2.5→1.3, 多样文本 800→1500
  4. 自动续训 + 优化器状态保存
"""
import os,sys,math,random,json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict
sys.path.insert(0,"/root/Qwen3-VL")
from bridge import QwenEG3DBridge,get_llm_hidden,pose_6d_to_camera_params

PITCH_WEIGHT=1.3       # ↓ 2.5→1.3 释放yaw梯度
CONTRASTIVE_WEIGHT=0.1  # 对比学习权重

def circular_mse(pred,target):
    d=(pred-target)%(2*math.pi);d=torch.min(d,2*math.pi-d);return(d**2).mean()

# ============================================================
# 多样化文本生成器 v2: 7种完全不同语法结构的句式
# ============================================================
def gen_text_v2(yaw, pitch_offset):
    """为任意角度生成7种完全不同语法结构的自然语言描述"""
    ay=abs(yaw)
    # 注意: yaw<0=左侧, yaw>0=右侧, yaw≈0=正面
    if yaw < -0.65: s_left="左侧45度"; s_rt="左边";   dr="向左";   sf="左"
    elif yaw < -0.35: s_left="左侧30度"; s_rt="左边";  dr="向左";   sf="左"
    elif yaw < -0.08: s_left="左侧"; s_rt="左边";      dr="向左";   sf="左"
    elif yaw < 0.08: s_left="正面"; s_rt="前方";       dr="";       sf="正"
    elif yaw < 0.35: s_left="右侧"; s_rt="右边";       dr="向右";   sf="右"
    elif yaw < 0.65: s_left="右侧30度"; s_rt="右边";   dr="向右";   sf="右"
    else: s_left="右侧45度"; s_rt="右边";               dr="向右";   sf="右"
    
    deg = round(ay*180/math.pi) if ay > 0.05 else 0
    if deg==0: deg_s=""
    else: deg_s=f"{deg}度"
    
    # 垂直方向
    if pitch_offset<-0.4: pt="大俯视";   pt_v="低头俯视";     pt_s="从上往下俯看";    pt_sc="俯视角度"
    elif pitch_offset<-0.15: pt="俯视";  pt_v="略低头俯视";   pt_s="从上往下";        pt_sc="俯视"
    elif pitch_offset<0.15: pt="平视";   pt_v="平视前方";     pt_s="水平看";          pt_sc="水平角度"
    elif pitch_offset<0.4: pt="仰视";    pt_v="略仰头看";     pt_s="从下往上看";      pt_sc="仰视"
    else: pt="大仰视";                   pt_v="抬头仰望";     pt_s="从下往上仰望";    pt_sc="仰视角度"

    addr = f"{s_left}{deg_s}" if deg_s else s_left
    
    patterns = [
        # A: 标准模板风 (控制组)
        lambda: f"{addr}{pt}",
        # B: 动宾结构
        lambda: f"脸{dr}转{deg_s}，{pt_v}",
        # C: 方位短语
        lambda: f"从{s_left}{deg_s}的方向{pt_sc}",
        # D: 摄影用语
        lambda: f"用{s_left}{deg_s}{pt}的镜头拍摄",
        # E: 名词短语
        lambda: f"一张{s_left}{deg_s}、{pt}的人脸照片",
        # F: 场景描述
        lambda: f"一个人脸朝{sf}偏{deg_s}，{pt_v}",
        # G: 简洁命令
        lambda: f"头偏{sf}{deg_s}，{pt_s}",
    ]
    return random.choice(patterns)()

# 正面对比对: 同一角度生成2种不同说法
def gen_text_pair(yaw, pitch_offset):
    t1=gen_text_v2(yaw,pitch_offset)
    t2=gen_text_v2(yaw,pitch_offset)
    # 确保不是一模一样的文本(概率极低但以防万一)
    while t2==t1: t2=gen_text_v2(yaw,pitch_offset)
    return t1,t2


# ============================================================
# 混合数据集 v4
# ============================================================
class MixedPoseDatasetV4:
    YAWS=[round(y,2) for y in [-0.79,-0.52,-0.26,0.0,0.26,0.52,0.79]]
    PITCHES=[round(p,2) for p in [-0.5,-0.25,0.0,0.25,0.5]]
    DIR_N={-0.79:"左侧45度",-0.52:"左侧30度",-0.26:"左侧15度",0.0:"正面",0.26:"右侧15度",0.52:"右侧30度",0.79:"右侧45度"}
    PITCH_N={-0.5:"大俯视",-0.25:"俯视",0.0:"平视",0.25:"仰视",0.5:"大仰视"}

    def __init__(self,val_ratio=0.15,tw=2,ed=1500):
        """tw: 模板权重(默认2), ed: 实时多样化文本数(默认1500)"""
        self.tw=tw;self.ed=ed
        self.T=[(y,p,f"{self.DIR_N[y]}{self.PITCH_N[p]}") for y in self.YAWS for p in self.PITCHES]
        self.TT={t[2] for t in self.T}
        print(f"模板: {len(self.T)} (7yaw x 5pitch, weight={tw}x)")
        vp="/root/Qwen3-VL/text_variants.json"
        if os.path.exists(vp):
            with open(vp) as f:av=json.load(f)
            fv=[d for d in av if d['text'] not in self.TT and abs(d['yaw'])<=0.79]
            se=set();uv=[]
            for d in fv:
                k=(d['text'],round(d['yaw'],2),round(d['pitch_offset'],2))
                if k not in se:se.add(k);uv.append(d)
            print(f"多样化文件: {len(uv)} 条")
        else:uv=[]
        tg=defaultdict(list)
        for d in uv:tg[d['text']].append(d)
        at=list(tg.keys());random.shuffle(at)
        nv=max(1,int(len(at)*val_ratio))
        vt=set(at[:nv]);trt=set(at[nv:])
        self.dtr=[d for d in uv if d['text'] in trt]
        self.dva=[d for d in uv if d['text'] in vt]
        print(f"多样切分: 训练{len(self.dtr)} 验证{len(self.dva)}")
        self._rebuild()
        # 验证集
        self.val_data=[]
        for item in self.dva:
            p=math.pi/2+item['pitch_offset']
            r6=torch.tensor([[item['yaw'],p,2.7,18.8,0.0,0.2]])
            self.val_data.append(dict(text=item['text'],r6=r6,cam=pose_6d_to_camera_params(r6),yg=item['yaw'],pg=p,po=item['pitch_offset']))
        self.val_t=[]
        for y,po,text in self.T:
            p=math.pi/2+po
            r6=torch.tensor([[y,p,2.7,18.8,0.0,0.2]])
            self.val_t.append(dict(text=text,r6=r6,cam=pose_6d_to_camera_params(r6),yg=y,pg=p,po=po))
        print(f"验证: 多样{len(self.val_data)}+模板{len(self.val_t)}")

    def _rebuild(self):
        self.pool=[]
        self.contrast_pool=[]
        # 模板(×tw)
        for _ in range(self.tw):
            for y,po,text in self.T:
                self.pool.append(dict(text=text,yaw=y,po=po,is_t=True))
        # 多样化文件
        for d in self.dtr:
            self.pool.append(dict(text=d['text'],yaw=d['yaw'],po=d['pitch_offset'],is_t=False))
        # 实时生成多样文本 + 对比对
        for _ in range(self.ed):
            y=random.uniform(-0.79,0.79)
            po=random.uniform(-0.6,0.6)
            t=gen_text_v2(y,po)
            self.pool.append(dict(text=t,yaw=round(y,2),po=round(po,2),is_t=False))
            # 对比对: 同一角度另一种说法
            t2=gen_text_v2(y,po)
            self.contrast_pool.append((y,po,t,t2))
        random.shuffle(self.pool)
        self.n=len(self.pool)
        tc=sum(1 for p in self.pool if p['is_t'])
        print(f"池: {self.n} (模板{tc}+多样{self.n-tc}) 对比对: {len(self.contrast_pool)}")

    def get_batch(self,n=1):
        items=random.choices(self.pool,k=n)
        ts,r6s,cs=[],[],[]
        for item in items:
            y=item['yaw']+random.uniform(-0.06,0.06)
            po=item['po']+random.uniform(-0.06,0.06)
            pitch=math.pi/2+po
            r=random.uniform(2.5,3.0);f=random.uniform(16,22)
            lx=random.uniform(-0.05,0.05);lz=random.uniform(0.15,0.25)
            r6=torch.tensor([[y,pitch,r,f,lx,lz]],dtype=torch.float32)
            ts.append(item['text']);r6s.append(r6);cs.append(pose_6d_to_camera_params(r6))
        return ts,torch.cat(r6s).cuda(),torch.cat(cs).cuda()

    def get_contrast_batch(self):
        """返回一对同角度不同文本的对比样本"""
        y,po,t1,t2=random.choice(self.contrast_pool)
        y+=random.uniform(-0.06,0.06)
        po+=random.uniform(-0.06,0.06)
        pitch=math.pi/2+po
        r=random.uniform(2.5,3.0);f=random.uniform(16,22)
        lx=random.uniform(-0.05,0.05);lz=random.uniform(0.15,0.25)
        r6=torch.tensor([[y,pitch,r,f,lx,lz]],dtype=torch.float32)
        return t1,t2,r6.cuda(),pose_6d_to_camera_params(r6).cuda()


# ============================================================
# 训练
# ============================================================
def train():
    print("="*60)
    print(f"混合v4: 7句式多样文本 + 对比学习")
    print(f"  模板×{2}  Pitch×{PITCH_WEIGHT}  对比×{CONTRASTIVE_WEIGHT}  多样{1500}/轮")
    print("="*60)
    bridge=QwenEG3DBridge()
    ckd="/root/Qwen3-VL/checkpoints";os.makedirs(ckd,exist_ok=True)
    ck=sorted([f for f in os.listdir(ckd) if f.startswith('mix_v2_') and f.endswith('.pt')],
              key=lambda x:int(x.split('_')[-1].split('.')[0]))
    se=0;opt_s=None
    if ck:
        cp=torch.load(os.path.join(ckd,ck[-1]),map_location='cuda')
        bridge.pose.load_state_dict(cp['pose']);se=cp.get('epoch',0);opt_s=cp.get('optimizer')
        print(f"加载: {ck[-1]} (epoch {se})")
    for p in bridge.qwen.parameters():p.requires_grad=False
    for p in bridge.pose.parameters():p.requires_grad=True
    params=list(bridge.pose.parameters())
    print(f"参数: {sum(p.numel() for p in params):,}")
    ne=100  # 更多epoch, 因为数据更丰富了
    opt=torch.optim.AdamW(params,lr=3e-5,weight_decay=1e-5)
    if opt_s is not None:
        try:opt.load_state_dict(opt_s);print("优化器恢复")
        except:print("优化器重新初始化")
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=ne,eta_min=1e-6)
    for _ in range(min(se,ne)):sch.step()
    for ep in range(se,ne):
        bridge.train()
        ds=MixedPoseDatasetV4(tw=2,ed=1500)
        tl=tyl=tpl=tcl=0.0;sm=0.0
        pb=tqdm(range(ds.n),desc=f"Epoch {ep+1}/{ne}")
        for st in pb:
            opt.zero_grad()
            # 主loss
            ts,r6,cam=ds.get_batch(n=1)
            with torch.no_grad():h=get_llm_hidden(bridge.qwen,bridge.processor,ts[0])
            p6d=bridge.pose(h.unsqueeze(0));pcam=pose_6d_to_camera_params(p6d)
            ly=circular_mse(p6d[:,0],r6[:,0])
            lp=F.mse_loss(p6d[:,1],r6[:,1])
            lo=F.mse_loss(p6d[:,2:],r6[:,2:])
            l6=ly+lp*PITCH_WEIGHT+lo;lc=F.mse_loss(pcam,cam);loss=l6+lc
            
            # 对比学习 (每3步)
            if st%3==0:
                t1,t2,cr6,ccam=ds.get_contrast_batch()
                with torch.no_grad():
                    h1=get_llm_hidden(bridge.qwen,bridge.processor,t1)
                    h2=get_llm_hidden(bridge.qwen,bridge.processor,t2)
                # 同角度文本的hidden state应该相似
                l_contra=CONTRASTIVE_WEIGHT*(1-F.cosine_similarity(h1.unsqueeze(0),h2.unsqueeze(0)))
                loss=loss+l_contra
                tcl+=l_contra.item()
            
            loss.backward();torch.nn.utils.clip_grad_norm_(params,0.5);opt.step()
            tl+=loss.item();tyl+=ly.item();tpl+=lp.item()
            sm=0.95*sm+0.05*loss.item() if sm else loss.item()
            pb.set_postfix({'l':f"{sm:.4f}",'y':f"{ly.item():.4f}",'p':f"{lp.item():.4f}"})
        sch.step()
        if (ep+1)%5==0:
            bridge.eval()
            ye,pe=[],[]
            with torch.no_grad():
                for v in ds.val_data:
                    h=get_llm_hidden(bridge.qwen,bridge.processor,v['text'])
                    p6d=bridge.pose(h.unsqueeze(0))
                    e=abs(p6d[0,0].item()-v['yg']);ye.append(min(e,2*math.pi-e))
                    pe.append(abs(p6d[0,1].item()-v['pg']))
            ty,tp=[],[]
            with torch.no_grad():
                for v in ds.val_t:
                    h=get_llm_hidden(bridge.qwen,bridge.processor,v['text'])
                    p6d=bridge.pose(h.unsqueeze(0))
                    e=abs(p6d[0,0].item()-v['yg']);ty.append(min(e,2*math.pi-e))
                    tp.append(abs(p6d[0,1].item()-v['pg']))
            pth=os.path.join(ckd,f"mix_v4_epoch_{ep+1:03d}.pt")
            torch.save({'epoch':ep+1,'pose':bridge.pose.state_dict(),'optimizer':opt.state_dict()},pth)
            print(f"\n  [Ep{ep+1}] loss={tl/ds.n:.4f} lr={sch.get_last_lr()[0]:.2e}")
            print(f"  多样: Dy={np.mean(ye)*180/math.pi:.2f} Dp={np.mean(pe)*180/math.pi:.2f}")
            print(f"  模板: Dy={np.mean(ty)*180/math.pi:.2f} Dp={np.mean(tp)*180/math.pi:.2f}")
            for i,v in enumerate(ds.val_data[:3]):
                with torch.no_grad():
                    p6d=bridge.pose(get_llm_hidden(bridge.qwen,bridge.processor,v['text']).unsqueeze(0))
                print(f"  [{i+1}] '{v['text'][:35]:35s}' -> y={p6d[0,0]:+.2f} p={p6d[0,1]:+.2f} gt=({v['yg']:.2f},{v['pg']:.2f})")
            print(f"  保存: {pth}\n")
            bridge.train()
    print("\n完成!")
if __name__=="__main__":train()
