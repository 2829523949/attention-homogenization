#!/usr/bin/env python3
"""Paired evaluator comparison after all route outputs are complete."""
import json
from pathlib import Path
BASE=Path('/mnt/d/vlm_repair_v2/outputs/high_recall_binding_repair')
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def transition(a,b,field,key):
    aa={r[key]:bool(r[field]) for r in a};bb={r[key]:bool(r[field]) for r in b};assert aa.keys()==bb.keys()
    return {'wrong_to_correct':sum(not aa[k] and bb[k] for k in aa),'correct_to_wrong':sum(aa[k] and not bb[k] for k in aa),'both_correct':sum(aa[k] and bb[k] for k in aa),'both_wrong':sum(not aa[k] and not bb[k] for k in aa),'n':len(aa)}
def compare(da,db):
    ae=rows(da/'entity_evaluation.jsonl');be=rows(db/'entity_evaluation.jsonl')
    aq=rows(da/'qa_binding_evaluation.jsonl');bq=rows(db/'qa_binding_evaluation.jsonl')
    for x in ae:x['key']=x['family_id']+'|'+x['logical_ref']
    for x in be:x['key']=x['family_id']+'|'+x['logical_ref']
    out={'entity_region':transition(ae,be,'selected_correct','key'),'grounded_target':transition(aq,bq,'grounded_target_correct','qa_id')}
    if (da/'downstream_results.jsonl').exists() and (db/'downstream_results.jsonl').exists():out['minimal_state']=transition(rows(da/'downstream_results.jsonl'),rows(db/'downstream_results.jsonl'),'minimal_state_sufficient','qa_id')
    return out
if __name__=='__main__':
    p0=BASE/'p0_qwen_independent';p1=BASE/'p1_objembed';p2=BASE/'p2_joint_assignment'
    write(p1/'paired_vs_qwen.json',compare(p0,p1))
    best=p0 if json.loads((p0/'binding_metrics.json').read_text())['grounded_target_correct']>=json.loads((p1/'binding_metrics.json').read_text())['grounded_target_correct'] else p1
    write(p2/'paired_vs_best_independent.json',compare(best,p2))
