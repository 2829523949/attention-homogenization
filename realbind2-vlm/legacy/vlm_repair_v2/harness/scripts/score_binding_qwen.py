#!/usr/bin/env python3
"""Independent, answer-blind Qwen YES/NO logit scorer for frozen union candidates."""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from PIL import Image

ROOT=Path('/mnt/d/vlm_repair_v2');sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/high_recall_binding_repair/p0_qwen_independent'
POOL=OUT/'candidate_pool_manifest.json'
PROMPT='Does the candidate shown correspond to the entity described as "{referent}"? Answer exactly YES or NO.'

def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def score_one(q,phrase,crop_path):
    import torch
    from experiments import qwen35_kv_gdn_factorial_pilot as runtime
    prompt=PROMPT.format(referent=phrase)
    rendered=runtime.render_chat(q.processor,str(crop_path),prompt)
    inputs=runtime.encode_single(q.processor,rendered,str(crop_path),q.device)
    with torch.inference_mode():
        output=q.model(**inputs,use_cache=False,logits_to_keep=1)
    logits=output.logits[0,-1].float().log_softmax(-1)
    tok=q.processor.tokenizer
    y=tok.encode('YES',add_special_tokens=False);n=tok.encode('NO',add_special_tokens=False)
    assert len(y)==len(n)==1,(y,n)
    return float(logits[y[0]].item()),float(logits[n[0]].item()), sorted(inputs.keys())
def main(limit):
    pool=json.loads(POOL.read_text())['referents'];OUT.mkdir(parents=True,exist_ok=True)
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained('/root/Qwen3-VL/models/Qwen3.5-4B',trust_remote_code=True)
    audit={x:t.encode(x,add_special_tokens=False) for x in ('YES','NO',' YES',' NO')}
    (OUT/'qwen_tokenization_audit.json').write_text(json.dumps({'tokenization':audit,'response_strings':['YES','NO'],'score':'next-token log P(YES)-log P(NO)','prompt_template':PROMPT},indent=2)+'\n')
    from harness.reasoners.qwen35 import Qwen35Reasoner
    q=Qwen35Reasoner();done={(r['family_id'],r['logical_ref'],r['candidate_id']) for r in rows(OUT/'qwen_scores.jsonl')};count=0
    cropdir=OUT/'crops';cropdir.mkdir(exist_ok=True)
    for row in pool:
        im=Image.open(row['image_path']).convert('RGB')
        for c in row['candidates']:
            key=(row['family_id'],row['logical_ref'],c['candidate_id'])
            if key in done:continue
            b=c['bbox_xyxy'];box=(max(0,math.floor(b[0])),max(0,math.floor(b[1])),min(im.width,math.ceil(b[2])),min(im.height,math.ceil(b[3])))
            p=cropdir/f'{row["family_id"]}__{row["logical_ref"]}__{c["candidate_id"]}.png';im.crop(box).save(p)
            rec={'family_id':key[0],'logical_ref':key[1],'candidate_id':key[2],'referent_phrase':row['referent_phrase'],'bbox_xyxy':b,'provenance':c['provenance'],'mask_available':False,'prompt':PROMPT.format(referent=row['referent_phrase']),'crop_path':str(p)}
            try:
                y,n,keys=score_one(q,row['referent_phrase'],p)
                rec.update({'yes_logprob':y,'no_logprob':n,'compatibility_score':y-n,'input_keys':keys,'status':'OK'})
            except Exception as e:
                rec.update({'yes_logprob':None,'no_logprob':None,'compatibility_score':None,'status':'MODEL_EXECUTION_FAILURE','error':repr(e)})
            with (OUT/'qwen_scores.jsonl').open('a') as h:h.write(json.dumps(rec,ensure_ascii=False)+'\n')
            count+=1
            if count%10==0:print('scored',len(done)+count,'/',sum(len(x['candidates']) for x in pool),flush=True)
            if limit and count>=limit:return
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);a=p.parse_args();main(a.limit)
