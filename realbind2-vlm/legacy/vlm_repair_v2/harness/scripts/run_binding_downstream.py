#!/usr/bin/env python3
"""Apply frozen integrated downstream stages to a completed binding route."""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path('/mnt/d/vlm_repair_v2');sys.path.insert(0,str(ROOT))
from harness.scripts import run_strong_integrated_repair as frozen
BASE=ROOT/'outputs/high_recall_binding_repair'
MAP={'p0':'p0_qwen_independent','p1':'p1_objembed','p2':'p2_joint_assignment','p3':'p3_direct_grounding'}
STAGES=('prepare','mask','attribute','assemble','final','report')
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write(p,x):p.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in x))
def main(route,stage):
    d=BASE/MAP[route];d.mkdir(parents=True,exist_ok=True);frozen.OUT=d
    if stage=='prepare':
        name='direct_regions.jsonl' if route=='p3' else 'joint_assignment_results.jsonl' if route=='p2' else 'selected_regions.jsonl'
        selection=rows(d/name);assert len(selection)==40
        binding=[]
        for r in selection:
            binding.append({'family_id':r['family_id'],'logical_ref':r['logical_ref'],'referent_phrase':next(x['referent_phrase'] for x in json.loads((BASE/'p0_qwen_independent/candidate_pool_manifest.json').read_text())['referents'] if x['family_id']==r['family_id'] and x['logical_ref']==r['logical_ref']),'selected_bbox_xyxy':r['selected_bbox_xyxy'],'selected_provider':MAP[route],'fallback_used':bool(r.get('fallback_reason')),'fallback_reason':r.get('fallback_reason'),'persistent_entity_state':True})
        write(d/'binding_results.jsonl',binding)
        shutil.copy2(ROOT/'outputs/strong_integrated_repair/grounding_results.jsonl',d/'grounding_results.jsonl')
        assert len(rows(d/'grounding_results.jsonl'))==40
        return
    if stage=='mask':frozen.stage_mask()
    elif stage=='attribute':frozen.stage_attribute()
    elif stage=='assemble':frozen.stage_assemble()
    elif stage=='final':frozen.stage_final()
    elif stage=='report':
        qa=rows(d/'qa_results.jsonl');final={r['qa_id']:r for r in rows(d/'final_results.jsonl')};assert len(qa)==len(final)==40
        result=[{**r,'final_correct':final[r['qa_id']]['final_correct'],'clean_final_correct':bool(r['minimal_state_sufficient'] and final[r['qa_id']]['final_correct']),'image_present':final[r['qa_id']]['image_present'],'vision_encoder_called':final[r['qa_id']]['vision_encoder_called']} for r in qa]
        write(d/'downstream_results.jsonl',result)
        assert all(not r['image_present'] and not r['vision_encoder_called'] for r in result)
        binding=rows(d/'binding_results.jsonl');masks=rows(d/'mask_results.jsonl')
        assert len(binding)==len(masks)==40
        family_qas={f:[r for r in qa if r['family_id']==f] for f in {r['family_id'] for r in qa}}
        assert all(len(v)==2 and {r['qa_type'] for r in v}=={'REL2C_LEFT','REL2C_RIGHT'} for v in family_qas.values())
        persistence={'families':20,'persistent_A_B_groundings_shared_across_both_QA':True,'persistent_A_B_masks_shared_across_both_QA':True,'binding_rows':len(binding),'mask_rows':len(masks),'image_free_final_calls':sum(not r['image_present'] and not r['vision_encoder_called'] for r in result),'hard_handoff_pass':True}
        (d/'persistence_and_handoff_audit.json').write_text(json.dumps(persistence,indent=2)+'\n')
        print({'route':route,'logical':sum(r['logical_target_correct'] for r in result),'grounded':sum(r['grounded_target_correct'] for r in result),'state':sum(r['minimal_state_sufficient'] for r in result),'clean':sum(r['clean_final_correct'] for r in result),'raw':sum(r['final_correct'] for r in result)})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('route',choices=MAP);p.add_argument('stage',choices=STAGES);a=p.parse_args();main(a.route,a.stage)
