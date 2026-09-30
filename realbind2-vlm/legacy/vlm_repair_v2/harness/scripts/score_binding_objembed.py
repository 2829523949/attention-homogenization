#!/usr/bin/env python3
"""ObjEmbed official REC path with frozen externally supplied union proposals."""
from __future__ import annotations
import copy,json,sys
from pathlib import Path
ROOT=Path('/mnt/d/vlm_repair_v2');sys.path.insert(0,str(ROOT/'external/objembed_pydeps'));sys.path.insert(0,str(ROOT/'external/ObjEmbed'))
OUT=ROOT/'outputs/high_recall_binding_repair/p1_objembed';POOL=ROOT/'outputs/high_recall_binding_repair/p0_qwen_independent/candidate_pool_manifest.json'
CHECKPOINT=Path('/root/vlm_objembed_2b') if (Path('/root/vlm_objembed_2b')/'model-00001-of-00002.safetensors').exists() else ROOT/'models/objembed_2b'
QUERY_TEMPLATE='Find an object that matches the given caption. {referent} <local_text>'

def validate_boxes(boxes,size):
    w,h=size
    out=[]
    for b in boxes:
        assert len(b)==4
        c=[max(0.,min(float(w),b[0])),max(0.,min(float(h),b[1])),max(0.,min(float(w),b[2])),max(0.,min(float(h),b[3]))]
        assert 0<=c[0]<c[2]<=w and 0<=c[1]<c[3]<=h,(b,c)
        out.append(c)
    return out

def main():
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoProcessor
    from models.qwen3vl_objembed import ObjectEmbed
    from models.vision_process import process_vision_info
    OUT.mkdir(parents=True,exist_ok=True)
    pool=json.loads(POOL.read_text())['referents']
    processor=AutoProcessor.from_pretrained(CHECKPOINT)
    model=ObjectEmbed.from_pretrained(CHECKPOINT,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map='auto').eval()
    obj_id=processor.tokenizer.convert_tokens_to_ids('<object>')
    text_id=processor.tokenizer.convert_tokens_to_ids('<local_text>')
    global_id=processor.tokenizer.convert_tokens_to_ids('<global>') if model.use_global_caption else None
    global_text_id=processor.tokenizer.convert_tokens_to_ids('<global_text>') if model.use_global_caption else None
    model.model.object_token_id=obj_id
    done={(r['family_id'],r['logical_ref']) for r in [json.loads(x) for x in (OUT/'objembed_scores.jsonl').read_text().splitlines()]} if (OUT/'objembed_scores.jsonl').exists() else set()
    for row in pool:
        key=(row['family_id'],row['logical_ref'])
        if key in done:continue
        im=Image.open(row['image_path']).convert('RGB')
        boxes=validate_boxes([c['bbox_xyxy'] for c in row['candidates']],im.size)
        msg=[{'role':'user','content':[{'type':'text','text':QUERY_TEMPLATE.format(referent=row['referent_phrase'])}]}]
        texts=[processor.apply_chat_template(msg,tokenize=False).strip()]
        tinputs=processor(text=texts,return_tensors='pt',padding=True,do_resize=False).to(model.device)
        with torch.inference_mode():
            q=model(text_processed=tinputs,global_id=global_id,local_text_id=text_id,global_text_id=global_text_id)['local_text_embeddings']
        obj_str=''.join(f'Object {j}: <object><object>. ' for j in range(len(boxes)))
        if model.use_two_tokens==0:obj_str+='The global image is <global>'
        elif model.use_two_tokens==1:obj_str='The global image is <global>. '+obj_str+'The global image is <global>'
        else:obj_str='The coarse global image is <global>. '+obj_str+' The detailed global image is <global>. '
        msg=[{'role':'user','content':[{'type':'image','image':copy.deepcopy(im)},{'type':'text','text':'Locate the specific object being described by analyzing its unique instance-level attributes, its spatial position, and its relationship with surrounding objects. '+obj_str}]}]
        images,videos=process_vision_info(msg,image_patch_size=16)
        texts=[processor.apply_chat_template(msg,tokenize=False).strip()]
        inputs=processor(text=texts,images=images,videos=videos,return_tensors='pt',padding=True,do_resize=False).to(model.device)
        with torch.inference_mode():
            pred=model(**inputs,bboxes=[torch.tensor(boxes,device=model.device,dtype=model.dtype)],ori_shapes=[im.size],bboxes_id=obj_id,global_id=global_id,local_text_id=text_id,global_text_id=global_text_id)
        emb=pred['object_embeddings'];loc=pred['objness'].sigmoid().float().flatten()
        semantic=(F.normalize(q,dim=-1)@F.normalize(emb,dim=-1).T*model.logit_log_scale.exp()+model.logit_bias).float().sigmoid().flatten()
        score=(semantic*loc).detach().cpu().tolist()
        assert len(score)==len(boxes)
        rec={'family_id':key[0],'logical_ref':key[1],'referent_phrase':row['referent_phrase'],'query_template':QUERY_TEMPLATE,'candidate_ids':[c['candidate_id'] for c in row['candidates']],'scores':score,'semantic_scores':semantic.detach().cpu().tolist(),'localization_quality':loc.detach().cpu().tolist(),'status':'OK'}
        with (OUT/'objembed_scores.jsonl').open('a') as h:h.write(json.dumps(rec)+'\n')
        print('objembed',len(done)+1,key,len(boxes),flush=True);done.add(key)
if __name__=='__main__':main()
