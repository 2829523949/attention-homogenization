import os,sys,torch,math
import numpy as np
from PIL import Image
sys.path.insert(0,'/root/Qwen3-VL')
os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1'
from transformers import AutoModelForImageTextToText,AutoProcessor

print("加载模型(eager)...")
model=AutoModelForImageTextToText.from_pretrained('/root/Qwen3-VL-2B-Instruct',torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager',local_files_only=True,trust_remote_code=True)
processor=AutoProcessor.from_pretrained('/root/Qwen3-VL-2B-Instruct',local_files_only=True,trust_remote_code=True)

img=Image.open('/root/output_phase1_test/正面平视.png').convert('RGB')
img=img.resize((224,224))
nvt=256

queries=["鼻子","嘴巴","眼睛"]
for q in queries:
    msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":f"这张人脸图片中{q}在哪里？请用坐标回答。"}]}]
    prompt=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=prompt,images=img,return_tensors='pt').to('cuda')
    
    # 方法1: 直接用model的输出拿attentions
    with torch.no_grad():
        out=model(**inputs,output_attentions=True)
    
    # output_attentions=True返回每层的attention (past_key_values格式?)
    # 试试看lm的输出
    lm=model.model.language_model
    with torch.no_grad():
        lm_out=lm(input_ids=inputs['input_ids'],
                  attention_mask=inputs.get('attention_mask'),
                  output_attentions=True)
    
    attns=lm_out.attentions
    if attns is not None and len(attns)>0:
        # attns是list of [batch, heads, q_len, k_len]
        a=attns[-1][0]  # 最后一层, batch0
        print(f"'{q}': attn shape={a.shape}")
        # a: [heads, q_len, k_len]
        # 取最后2个文本token对visual token的注意力
        vis_a=a[:,-3:-1,:nvt].mean(dim=[0,1])  # [nvt]
        hm=vis_a.reshape(16,16).cpu().float().numpy()
        max_pos=np.unravel_index(hm.argmax(),hm.shape)
        print(f"  peak=({max_pos[0]:2d},{max_pos[1]:2d}) val={hm.max():.3f}")
    else:
        print(f"'{q}': no attentions returned")
