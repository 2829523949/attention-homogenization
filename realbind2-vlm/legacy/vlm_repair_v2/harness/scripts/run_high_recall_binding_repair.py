#!/usr/bin/env python3
"""Frozen union proposal construction and evaluator-only recall audit.

The inference manifest is written before evaluator annotations are opened.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/d/vlm_repair_v2')
OUT = ROOT/'outputs/high_recall_binding_repair'
SAM = ROOT/'tmp/sam3_dev_candidates.json'
R4 = ROOT/'outputs/region_bottleneck_joint_repair/candidate_results.jsonl'
GT = ROOT/'outputs/grounding_dino_acquisition_pilot/entity_backend_comparison.jsonl'
K = 64
DEDUP_IOU = .7

def readl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
def iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0]));h=max(0,min(a[3],b[3])-max(a[1],b[1]));z=w*h
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return z/(aa+bb-z) if aa+bb>z else 0.
def valid(b,size):
    w,h=size
    return len(b)==4 and all(isinstance(x,(int,float)) for x in b) and b[2]>b[0] and b[3]>b[1] and b[0]<w and b[1]<h and b[2]>0 and b[3]>0
def canonical(items):
    out=[]
    for src in items:
        b=src['bbox_xyxy']
        match=next((x for x in out if iou(x['bbox_xyxy'],b)>DEDUP_IOU),None)
        if match:
            match['provenance'].append(src['source'])
            match['source_entries'].append(src['source_id'])
        else:
            out.append({'candidate_id':f'c{len(out):03d}','bbox_xyxy':b,'provenance':[src['source']], 'source_entries':[src['source_id']], 'sam3_rank':src.get('sam3_rank'),'native_score':src.get('native_score'),'mask_available':False})
    return out
def build():
    sam=json.loads(SAM.read_text());r4={(r['family_id'],r['logical_ref']):r for r in readl(R4) if r['generator']=='strong'}
    assert len(sam)==len(r4)==40
    rows=[]
    for key in sorted(sam):
        s=sam[key];f,ref=key.rsplit('|',1);rr=r4[f,ref];size=s['image_size'];items=[]
        for i,c in enumerate(rr['candidates']):
            b=c['bbox_xyxy']
            if valid(b,size): items.append({'source':'R4/MM-GDINO-L','source_id':f'r4:{i}','bbox_xyxy':b,'native_score':c.get('native_score')})
        for i,b in enumerate(s['boxes'][:K]):
            if valid(b,size): items.append({'source':'SAM3-LiteText-L','source_id':f'sam3:{i}','bbox_xyxy':b,'native_score':s['scores'][i],'sam3_rank':i+1})
        rows.append({'family_id':f,'logical_ref':ref,'referent_phrase':s['referent_phrase'],'image_path':s['image_path'],'image_size':size,'candidates':canonical(items)})
    assert len(rows)==40
    assert len({r['family_id'] for r in rows})==20
    write(OUT/'p0_qwen_independent/candidate_pool_manifest.json',{'schema':'binding_union_v1','primary_sam3_k':K,'dedup_bbox_iou_strictly_greater_than':DEDUP_IOU,'candidate_order':'R4 native order then SAM3 native order','mask_policy':'bbox crop used for matcher; mask view unavailable in frozen proposal dump','source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SAM,R4)},'referents':rows})
    return rows,sam,r4
def evaluate(pool,sam,r4):
    # This function is the first and only point where annotations are opened.
    gt={}
    for row in readl(GT):
        inf=row['inference']
        if inf['backend']=='grounding_dino':gt[inf['family_id'],inf['logical_ref']]=row['EVALUATOR_ONLY']['gt_bbox_xyxy_pixels']
    assert len(gt)==40
    by={(r['family_id'],r['logical_ref']):r for r in pool}
    depths={}; entity={}
    for label,k in [('R4',None),('SAM3 top16',16),('SAM3 top32',32),('SAM3 top64',64),('SAM3 top128',128),('SAM3 top200',200),('Union top64',None)]:
        hit=[]
        for key,box in gt.items():
            f,ref=key
            boxes=([c['bbox_xyxy'] for c in r4[key]['candidates']] if label=='R4' else sam[f+'|'+ref]['boxes'][:k] if k else [c['bbox_xyxy'] for c in by[key]['candidates']])
            hit.append(max((iou(b,box) for b in boxes),default=0)>=.5)
        depths[label]={'entity':sum(hit),'entity_denominator':40,'both_entities':sum(all(hit[2*i:2*i+2]) for i in range(20)),'both_denominator':20}
    for key,box in gt.items():
        p=by[key];entity['|'.join(key)]={'candidate_max_iou':max((iou(c['bbox_xyxy'],box) for c in p['candidates']),default=0),'candidate_available':any(iou(c['bbox_xyxy'],box)>=.5 for c in p['candidates'])}
    write(OUT/'final_comparison/candidate_oracle.json',{'depths':depths,'entity':entity,'evaluator_only':True})
    print(json.dumps(depths,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--skip-eval',action='store_true');a=p.parse_args()
    pool,sam,r4=build()
    if not a.skip_eval:evaluate(pool,sam,r4)
