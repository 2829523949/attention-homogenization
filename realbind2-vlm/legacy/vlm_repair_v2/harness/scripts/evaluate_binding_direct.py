#!/usr/bin/env python3
"""Evaluator-only direct EGM region and target metrics; no proposal routing."""
import json,sys
from pathlib import Path
ROOT=Path('/mnt/d/vlm_repair_v2');sys.path.insert(0,str(ROOT))
from harness.scripts.run_high_recall_binding_repair import iou
OUT=ROOT/'outputs/high_recall_binding_repair/p3_direct_grounding'
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
    sel={(r['family_id'],r['logical_ref']):r for r in rows(OUT/'direct_regions.jsonl')};assert len(sel)==40
    gt={}
    for r in rows(ROOT/'outputs/grounding_dino_acquisition_pilot/entity_backend_comparison.jsonl'):
        x=r['inference']
        if x['backend']=='grounding_dino':gt[x['family_id'],x['logical_ref']]=r['EVALUATOR_ONLY']['gt_bbox_xyxy_pixels']
    assert gt.keys()==sel.keys()
    entity=[]
    for k in sorted(sel):
        b=sel[k]['selected_bbox_xyxy'];v=iou(b,gt[k]) if b else 0.
        entity.append({'family_id':k[0],'logical_ref':k[1],'selected_bbox_xyxy':b,'selected_iou':v,'selected_correct':v>=.5,'invalid':b is None})
    em={(r['family_id'],r['logical_ref']):r for r in entity};prior={r['qa_id']:r for r in rows(ROOT/'outputs/minimal_decomposition_route_pilot/route_results.jsonl')};qas=[]
    for q in rows(ROOT/'outputs/minimal_decomposition_route_pilot/prepared.jsonl'):
        f=q['family_id'];a=sel[f,'A']['selected_bbox_xyxy'];b=sel[f,'B']['selected_bbox_xyxy'];true=prior[q['qa_id']]['EVALUATOR_ONLY']['true_target']
        if a and b:
            ac=a[0]+a[2];bc=b[0]+b[2];side='left' if q['qa_type'].endswith('LEFT') else 'right';relation='left' if ac<bc else 'right' if ac>bc else None;target='A' if relation==side else 'B' if relation else None
        else:target=None
        logical=target==true;grounded=bool(logical and target and em[f,target]['selected_correct'])
        qas.append({'qa_id':q['qa_id'],'family_id':f,'qa_type':q['qa_type'],'target':target,'logical_target_correct':logical,'grounded_target_correct':grounded})
    metrics={'region_correct':sum(r['selected_correct'] for r in entity),'both_regions_correct':sum(em[f,'A']['selected_correct'] and em[f,'B']['selected_correct'] for f in {r['family_id'] for r in entity}),'logical_target_correct':sum(r['logical_target_correct'] for r in qas),'grounded_target_correct':sum(r['grounded_target_correct'] for r in qas),'invalid_bbox':sum(r['invalid'] for r in entity),'entity_denominator':40,'family_denominator':20,'qa_denominator':40}
    write(OUT/'binding_metrics.json',metrics)
    (OUT/'entity_evaluation.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in entity));(OUT/'qa_binding_evaluation.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in qas))
    print(metrics)
if __name__=='__main__':main()
