import os,sys,torch,math
import numpy as np
from PIL import Image
sys.path.insert(0,'/root/Qwen3-VL')
os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1'
from transformers import AutoModelForImageTextToText,AutoProcessor

model=AutoModelForImageTextToText.from_pretrained('/root/Qwen3-VL-2B-Instruct',torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager',local_files_only=True,trust_remote_code=True)
processor=AutoProcessor.from_pretrained('/root/Qwen3-VL-2B-Instruct',local_files_only=True,trust_remote_code=True)

img=Image.open('/root/output_phase1_test/正面平视.png').convert('RGB')
img=img.resize((224,224))

# 确定visual tokens的精确数量 = 66, 排除前后特殊标记后约64
N_PATCH=64
H=W=8

queries=["鼻子","嘴巴","眼睛","下巴","额头","胡须","左眼","右眼","正脸"]
with torch.no_grad():
    for q in queries:
        msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":f"图中{q}在哪里"}]}]
        prompt=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inputs=processor(text=prompt,images=img,return_tensors='pt').to('cuda')
        
        lm_out=model.model.language_model(input_ids=inputs['input_ids'],
                                          attention_mask=inputs.get('attention_mask'),
                                          output_attentions=True)
        # 取最后3层的attention平均
        all_a=torch.stack([a[0] for a in lm_out.attentions[-3:]]).mean(dim=0)  # [heads, q_len, k_len]
        
        # 取最后2个文本token对所有visual token (跳过首尾特殊标记)的注意力
        vis_a=all_a[:,-3:-1,1:1+N_PATCH].mean(dim=[0,1]).cpu().float().numpy()  # [64]
        
        hm=vis_a.reshape(H,W)
        max_pos=np.unravel_index(hm.argmax(),hm.shape)
        
        # 计算热力图的中心/集中度
        ys,xs=np.mgrid[0:H,0:W]
        cy=(hm*ys).sum()/hm.sum()
        cx=(hm*xs).sum()/hm.sum()
        # 熵 (越低越集中)
        p=hm/hm.sum()+1e-10
        entropy=-np.sum(p*np.log2(p))
        max_entropy=np.log2(H*W)
        concentration=1-entropy/max_entropy
        
        print(f"'{q:6s}': peak=({max_pos[0]:d},{max_pos[1]:d}) cx={cx:.1f} cy={cy:.1f} conc={concentration:.2f}")
