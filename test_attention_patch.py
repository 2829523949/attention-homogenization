import os,sys,torch,math
import numpy as np
from PIL import Image
sys.path.insert(0,'/root/Qwen3-VL')
os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1'
from transformers import AutoModelForImageTextToText,AutoProcessor

print("加载模型(eager mode)...")
model=AutoModelForImageTextToText.from_pretrained('/root/Qwen3-VL-2B-Instruct',torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager',local_files_only=True,trust_remote_code=True)
processor=AutoProcessor.from_pretrained('/root/Qwen3-VL-2B-Instruct',local_files_only=True,trust_remote_code=True)
print("完成")

# 用EG3D人脸图
img=Image.open('/root/output_phase1_test/正面平视.png').convert('RGB')
img=img.resize((224,224))

# Qwen3-VL ViT: 先看处理后的图像尺寸
inputs=processor(text=[processor.apply_chat_template([{"role":"user","content":[{"type":"text","text":"test"}]}],tokenize=False,add_generation_prompt=True)],images=img,return_tensors='pt',padding=True)
print(f"image_grid_thw: {inputs.get('image_grid_thw')}")
print(f"input_ids shape: {inputs['input_ids'].shape}")

# 计算visual tokens
if 'image_grid_thw' in inputs:
    gt=inputs['image_grid_thw'][0]
    nvt=int(gt[0]*gt[1]*gt[2])
    print(f"Visual tokens: {nvt} (grid {gt[0]}x{gt[1]}x{gt[2]})")
    
    queries=["鼻子","眼睛","嘴巴","下巴","额头","胡须","左眼","右眼"]
    for q in queries:
        msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":f"请描述{q}"}]}]
        prompt=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inputs=processor(text=prompt,images=img,return_tensors='pt').to('cuda')
        
        attn_weights=[]
        def hook(m,i,o):
            if isinstance(o,tuple) and len(o)>1 and o[1] is not None:
                attn_weights.append(o[1].detach().float())
        
        handles=[]
        for layer in model.model.language_model.model.layers[-4:]:
            handles.append(layer.self_attn.register_forward_hook(hook))
        
        with torch.no_grad():
            out=model(**inputs,output_attentions=True)
        for h in handles:h.remove()
        
        if attn_weights:
            # 多层的attention平均
            attn=torch.stack(attn_weights).float().mean(dim=0)  # [layers, batch, heads, q_len, k_len]
            attn=attn.mean(dim=[0,1])  # [q_len, k_len]
            
            # 取最后一个user文本token对visual token的注意力
            # 文本token数量 = total - nvt
            n_total=attn.shape[0]
            vis_attn=attn[-3:-1,:nvt].mean(dim=0)  # [nvt]
            
            # reshape为2D
            h=int(math.sqrt(nvt))
            w=nvt//h
            if h*w==nvt:
                hm=vis_attn.reshape(h,w).cpu().numpy()
                max_pos=np.unravel_index(hm.argmax(),hm.shape)
                print(f"'{q:6s}': peak=({max_pos[0]:2d},{max_pos[1]:2d}) val={hm.max():.3f} range=[{hm.min():.3f},{hm.max():.3f}]")
            else:
                print(f"'{q:6s}': nvt={nvt} not square ({h}x{w})")
        else:
            print(f"'{q:6s}': no attention captured")
else:
    print("No image_grid_thw found")
