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

# 先跑一个纯文本看视觉token数 vs 总token数
msgs_vis=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"描述人脸"}]}]
prompt_vis=processor.apply_chat_template(msgs_vis,tokenize=False,add_generation_prompt=True)
inputs_vis=processor(text=prompt_vis,images=img,return_tensors='pt').to('cuda')

msgs_txt=[{"role":"user","content":[{"type":"text","text":"描述人脸"}]}]
prompt_txt=processor.apply_chat_template(msgs_txt,tokenize=False,add_generation_prompt=True)
inputs_txt=processor.tokenizer(prompt_txt,return_tensors='pt').to('cuda')

n_vis_total=inputs_vis['input_ids'].shape[1]
n_txt_only=inputs_txt['input_ids'].shape[1]
n_vis_tokens=n_vis_total-n_txt_only
print(f"图文输入: {n_vis_total} tokens")
print(f"纯文本输入: {n_txt_only} tokens")
print(f"Visual tokens: {n_vis_tokens}")
h=int(math.sqrt(n_vis_tokens))
print(f"Grid: {h}x{h}")

# 测试多种查询
queries=["鼻子","嘴巴","眼睛","下巴","额头","胡须","左眼","右眼"]
with torch.no_grad():
    for q in queries:
        msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":f"图中{q}在哪里"}]}]
        prompt=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        inputs=processor(text=prompt,images=img,return_tensors='pt').to('cuda')
        
        lm_out=model.model.language_model(input_ids=inputs['input_ids'],
                                          attention_mask=inputs.get('attention_mask'),
                                          output_attentions=True)
        attns=lm_out.attentions
        a=attns[-1][0]  # [heads, q_len, k_len]
        
        # 取最后2个文本token对visual token的注意力
        q_len=a.shape[1]
        vis_a=a[:,-3:-1,:n_vis_tokens].mean(dim=[0,1]).cpu().float().numpy()
        
        hm=vis_a.reshape(h,h)
        max_pos=np.unravel_index(hm.argmax(),hm.shape)
        # 计算分布集中度
        top5=np.sort(hm.flatten())[-5:].sum()
        total=hm.sum()
        concentration=top5/total
        print(f"'{q:6s}': peak=({max_pos[0]:2d},{max_pos[1]:2d}) val={hm.max():.3f} conc={concentration:.2f}")
