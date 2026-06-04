import os,sys,torch,math
import numpy as np
from PIL import Image
sys.path.insert(0,'/root/Qwen3-VL')
os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1'
from transformers import AutoModelForImageTextToText,AutoProcessor

print("加载模型(eager)...")
model=AutoModelForImageTextToText.from_pretrained('/root/Qwen3-VL-2B-Instruct',torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager',local_files_only=True,trust_remote_code=True)
processor=AutoProcessor.from_pretrained('/root/Qwen3-VL-2B-Instruct',local_files_only=True,trust_remote_code=True)
print("完成")

# 探测模型结构
lm=model.model.language_model
print(f"LM type: {type(lm).__name__}")
for n,_ in lm.named_children():
    print(f"  child: {n}")
    c=getattr(lm,n)
    if hasattr(c,'layers'):
        print(f"    has layers: {len(c.layers)}")
        print(f"    layer0 type: {type(c.layers[0]).__name__}")
        l0=c.layers[0]
        for n2,_ in l0.named_children():
            print(f"      sub: {n2}")
            if 'attn' in n2.lower():
                s=getattr(l0,n2)
                print(f"        type: {type(s).__name__}")

img=Image.open('/root/output_phase1_test/正面平视.png').convert('RGB')
img=img.resize((224,224))

queries=["鼻子","嘴巴","眼睛","下巴","额头","胡须","左眼","右眼"]
for q in queries[:3]:
    msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":f"请关注图中的{q}"}]}]
    prompt=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=prompt,images=img,return_tensors='pt').to('cuda')
    
    nvt=256
    attn_weights=[]
    def hook(m,i,o):
        if isinstance(o,tuple) and len(o)>1 and o[1] is not None:
            attn_weights.append(o[1].detach().float())
    
    handles=[]
    # 根据上面的探测结果找到正确的层路径
    # 尝试不同路径
    try:
        layers=lm.model.layers[-4:]
        for layer in layers:
            handles.append(layer.self_attn.register_forward_hook(hook))
    except:
        for n,_ in lm.named_children():
            c=getattr(lm,n)
            if hasattr(c,'layers'):
                for layer in c.layers[-4:]:
                    handles.append(layer.self_attn.register_forward_hook(hook))
                break
    
    with torch.no_grad():
        out=model(**inputs,output_attentions=True)
    for h in handles:h.remove()
    
    if attn_weights:
        attn=torch.stack(attn_weights).float().mean(dim=[0,1,2])
        n_total=attn.shape[0]
        vis_attn=attn[-3:-1,:nvt].mean(dim=0)
        h=w=16
        hm=vis_attn.reshape(h,w).cpu().numpy()
        max_pos=np.unravel_index(hm.argmax(),hm.shape)
        print(f"'{q:6s}': peak=({max_pos[0]:2d},{max_pos[1]:2d}) val={hm.max():.3f} range=[{hm.min():.3f},{hm.max():.3f}]")
    else:
        print(f"'{q:6s}': no attention")
