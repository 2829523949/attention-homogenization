#!/usr/bin/env python3
"""Evaluator-only native-rank provenance for binding rescues."""
import json,sys
from pathlib import Path
ROOT=Path('/mnt/d/vlm_repair_v2');sys.path.insert(0,str(ROOT))
from harness.scripts.run_high_recall_binding_repair import iou
BASE=ROOT/'outputs/high_recall_binding_repair'
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def analyze(route):
    d=BASE/route;pool={(r['family_id'],r['logical_ref']):r for r in json.loads((BASE/'p0_qwen_independent/candidate_pool_manifest.json').read_text())['referents']}
    selected={(r['family_id'],r['logical_ref']):r for r in rows(d/('joint_assignment_results.jsonl' if route=='p2_joint_assignment' else 'selected_regions.jsonl'))}
    historical={(r['family_id'],r['logical_ref']):r for r in rows(ROOT/'outputs/strong_integrated_repair/binding_results.jsonl')}
    sam=json.loads((ROOT/'tmp/sam3_dev_candidates.json').read_text());gt={}
    for r in rows(ROOT/'outputs/grounding_dino_acquisition_pilot/entity_backend_comparison.jsonl'):
        x=r['inference']
        if x['backend']=='grounding_dino':gt[x['family_id'],x['logical_ref']]=r['EVALUATOR_ONLY']['gt_bbox_xyxy_pixels']
    out=[]
    for k in sorted(gt):
        b=gt[k];old=historical[k]['selected_bbox_xyxy'];now=selected[k]['selected_bbox_xyxy']
        if not(now and iou(now,b)>=.5 and (not old or iou(old,b)<.5)):continue
        cs=pool[k]['candidates'];good=[(i,c) for i,c in enumerate(cs) if iou(c['bbox_xyxy'],b)>=.5]
        s=sam[k[0]+'|'+k[1]];sr=next((i+1 for i,box in enumerate(s['boxes']) if iou(box,b)>=.5),None)
        si=next((i+1 for i,c in enumerate(cs) if c['candidate_id']==selected[k]['selected_candidate_id']),None)
        out.append({'family_id':k[0],'logical_ref':k[1],'correct_candidate_sam3_native_rank':sr,'first_correct_union_rank':good[0][0]+1 if good else None,'selected_union_rank':si,'selected_provenance':selected[k].get('selected_provenance'),'selected_iou':iou(now,b),'historical_iou':iou(old,b) if old else 0})
    (d/'candidate_rank_rescues.json').write_text(json.dumps({'newly_rescued_entities':out,'count':len(out),'evaluator_only':True},indent=2)+'\n')
if __name__=='__main__':
    for a in sys.argv[1:]:analyze(a)
