import os,sys,torch,math
import numpy as np
from PIL import Image
sys.path.insert(0,'/root/Qwen3-VL')
os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1'
from transformers import AutoModelForImageTextToText,AutoProcessor

print("加载模型...")
model=AutoModelForImageTextToText.from_pretrained('/root/Qwen3-VL-2B-Instruct',torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='eager',local_files_only=True,trust_remote_code=True)
processor=AutoProcessor.from_pretrained('/root/Qwen3-VL-2B-Instruct',local_files_only=True,trust_remote_code=True)
print("完成")

def get_visual_token_count(image,processor):
    """计算图片会生成多少个visual token"""
    from transformers.models.qwen3_vl.image_processing_qwen3_vl import smart_resize
    w,h=image.size
    grid_thw=smart_resize(h,w,factor=28)
    t=grid_thw[0]*grid_thw[1]*grid_thw[2]
    return t,grid_thw

# 用EG3D生成的人脸测试
img=Image.open('/root/output_phase1_test/正面平视.png').convert('RGB')
# 可以试试调整大小让visual token数量可控
img=img.resize((224,224))

nvt,_=get_visual_token_count(img,processor)
print(f"Visual tokens: {nvt}")

messages=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"请描述鼻子"}]}]
# 改为多种查询
queries=["鼻子","眼睛","嘴巴","下巴","额头","胡须"]
for q in queries:
    messages=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":f"请描述{q}"}]}]
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=prompt,images=img,return_tensors='pt').to('cuda')
    
    # 注册hook拿attention
    attn_weights=[]
    def hook(m,i,o):
        if isinstance(o,tuple) and len(o)>1 and o[1] is not None:
            attn_weights.append(o[1].detach())
    
    lm=model.model.language_model
    # 取最后3层的attention平均
    handles=[]
    for layer in lm.model.layers[-3:]:
        handles.append(layer.self_attn.register_forward_hook(hook))
    
    with torch.no_grad():
        out=model(**inputs,output_attentions=True)
    
    for h in handles:h.remove()
    
    if attn_weights:
        # 平均所有head和层, 取最后几个text token对visual token的注意力
        attn=torch.stack(attn_weights).mean(dim=0)  # [n_layers, batch, heads, q_len, k_len]
        attn=attn.mean(dim=[0,1])  # [q_len, k_len]
        # 最后几个token是对生成文本的注意, 取最后一个是 </s> 或生成token
        # 取最后一个user text token对visual区域的注意
        vis_attn=attn[-3:-1,:nvt].mean(dim=0)  # [nvt]
        h=w=int(math.sqrt(nvt))
        if h*w==nvt:
            heatmap=vis_attn.reshape(h,w).cpu().float().numpy()
            print(f"  '{q}': heatmap [{heatmap.min():.3f}, {heatmap.max():.3f}], peak at ({heatmap.argmax()//w},{heatmap.argmax()%w})")
        else:
            print(f"  '{q}': nvt={nvt} not square")
    else:
        print(f"  '{q}': no attention")
