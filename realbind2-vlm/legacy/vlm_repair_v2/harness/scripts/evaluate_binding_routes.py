#!/usr/bin/env python3
"""Freeze selections first, then join evaluator annotations and QA labels."""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
ROOT=Path('/mnt/d/vlm_repair_v2');sys.path.insert(0,str(ROOT))
from harness.scripts.run_high_recall_binding_repair import iou
BASE=ROOT/'outputs/high_recall_binding_repair'
POOL=BASE/'p0_qwen_independent/candidate_pool_manifest.json'
GT=ROOT/'outputs/grounding_dino_acquisition_pilot/entity_backend_comparison.jsonl'
PREP=ROOT/'outputs/minimal_decomposition_route_pilot/prepared.jsonl'
PRIOR=ROOT/'outputs/minimal_decomposition_route_pilot/route_results.jsonl'

def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def writel(p,rs):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x)+'\n' for x in rs))
def select(route):
    pool=json.loads(POOL.read_text())['referents'];out=[]
    if route=='p0':
        scores={(r['family_id'],r['logical_ref'],r['candidate_id']):r for r in rows(BASE/'p0_qwen_independent/qwen_scores.jsonl')}
        d=BASE/'p0_qwen_independent'
        expected=sum(len(r['candidates']) for r in pool)
        assert len(scores)==expected,f'incomplete P0 scores: {len(scores)}/{expected}'
    else:
        scores={(r['family_id'],r['logical_ref']):r for r in rows(BASE/'p1_objembed/objembed_scores.jsonl')}
        d=BASE/'p1_objembed'
        assert len(scores)==len(pool),f'incomplete P1 scores: {len(scores)}/{len(pool)}'
    for r in pool:
        f,ref=r['family_id'],r['logical_ref'];cs=r['candidates'];values=[]
        for idx,c in enumerate(cs):
            s=scores.get((f,ref,c['candidate_id'])) if route=='p0' else scores.get((f,ref))
            v=s.get('compatibility_score') if route=='p0' and s else s['scores'][idx] if s and len(s['scores'])==len(cs) else None
            values.append(v if v is not None and math.isfinite(v) else None)
        valid=[(v,i) for i,v in enumerate(values) if v is not None]
        if valid:i=max(valid)[1];fallback=None
        elif cs:i=0;fallback='ALL_SCORES_INVALID_OR_MODEL_FAILURE'
        else:i=None;fallback='EMPTY_CANDIDATE_POOL'
        c=cs[i] if i is not None else None
        out.append({'family_id':f,'logical_ref':ref,'referent_phrase':r['referent_phrase'],'selected_candidate_id':c['candidate_id'] if c else None,'selected_bbox_xyxy':c['bbox_xyxy'] if c else None,'selected_provenance':c['provenance'] if c else [],'selected_score':values[i] if i is not None else None,'scores':values,'fallback_reason':fallback})
    writel(d/'selected_regions.jsonl',out)
    return out
def joint(independent):
    pool={(r['family_id'],r['logical_ref']):r for r in json.loads(POOL.read_text())['referents']}
    chosen={(r['family_id'],r['logical_ref']):r for r in independent};out=[];collisions=[]
    for f in sorted({k[0] for k in pool}):
        a=pool[f,'A']['candidates'];b=pool[f,'B']['candidates'];sa=chosen[f,'A']['scores'];sb=chosen[f,'B']['scores']
        pairs=[]
        for i,ca in enumerate(a):
            if sa[i] is None:continue
            for j,cb in enumerate(b):
                if sb[j] is None or iou(ca['bbox_xyxy'],cb['bbox_xyxy'])>.7:continue
                pairs.append((sa[i]+sb[j],i,j))
        if pairs:
            _,ia,ib=max(pairs);reason=None
        else:
            ia=next((i for i,c in enumerate(a) if c['candidate_id']==chosen[f,'A']['selected_candidate_id']),None)
            ib=next((i for i,c in enumerate(b) if c['candidate_id']==chosen[f,'B']['selected_candidate_id']),None)
            reason='NO_DISTINCT_VALID_SCORE_PAIR'
        before=iou(chosen[f,'A']['selected_bbox_xyxy'],chosen[f,'B']['selected_bbox_xyxy']) if chosen[f,'A']['selected_bbox_xyxy'] and chosen[f,'B']['selected_bbox_xyxy'] else 0
        collisions.append({'family_id':f,'independent_collision':before>.7,'joint_fallback':reason})
        for ref,index,cs in [('A',ia,a),('B',ib,b)]:
            c=cs[index] if index is not None else None
            out.append({'family_id':f,'logical_ref':ref,'selected_candidate_id':c['candidate_id'] if c else None,'selected_bbox_xyxy':c['bbox_xyxy'] if c else None,'selected_provenance':c['provenance'] if c else [],'fallback_reason':reason})
    d=BASE/'p2_joint_assignment';writel(d/'joint_assignment_results.jsonl',out);write(d/'collision_analysis.json',collisions)
    return out
def evaluate(selection,route):
    # Evaluator data loaded only after selected_regions.jsonl exists.
    selected={(r['family_id'],r['logical_ref']):r for r in selection};gt={}
    for r in rows(GT):
        x=r['inference']
        if x['backend']=='grounding_dino':gt[x['family_id'],x['logical_ref']]=r['EVALUATOR_ONLY']['gt_bbox_xyxy_pixels']
    pool={(r['family_id'],r['logical_ref']):r for r in json.loads(POOL.read_text())['referents']}
    assert len(selected)==len(gt)==len(pool)==40
    entity=[]
    for k in sorted(gt):
        s=selected[k];avail=any(iou(c['bbox_xyxy'],gt[k])>=.5 for c in pool[k]['candidates']);v=iou(s['selected_bbox_xyxy'],gt[k]) if s['selected_bbox_xyxy'] else 0
        entity.append({'family_id':k[0],'logical_ref':k[1],'candidate_available':avail,'selected_iou':v,'selected_correct':v>=.5,'failure':'NO_USABLE_CANDIDATE' if not avail else 'USABLE_CANDIDATE_NOT_SELECTED' if v<.5 else None})
    em={(r['family_id'],r['logical_ref']):r for r in entity}
    both=sum(em[f,'A']['selected_correct'] and em[f,'B']['selected_correct'] for f in {r['family_id'] for r in entity})
    prior={r['qa_id']:r for r in rows(PRIOR)};qas=[]
    for q in rows(PREP):
        f=q['family_id'];a=selected[f,'A']['selected_bbox_xyxy'];b=selected[f,'B']['selected_bbox_xyxy'];true=prior[q['qa_id']]['EVALUATOR_ONLY']['true_target']
        if a and b:
            relation='left' if (a[0]+a[2])<(b[0]+b[2]) else 'right' if (a[0]+a[2])>(b[0]+b[2]) else None
            side='left' if q['qa_type'].endswith('LEFT') else 'right';target='A' if relation==side else 'B' if relation else None
        else:target=None
        logical=target==true;grounded=logical and em[f,target]['selected_correct'] if target else False
        qas.append({'qa_id':q['qa_id'],'family_id':f,'qa_type':q['qa_type'],'target':target,'logical_target_correct':logical,'grounded_target_correct':grounded})
    metrics={'candidate_available':sum(r['candidate_available'] for r in entity),'selected_region_correct':sum(r['selected_correct'] for r in entity),'selection_efficiency':sum(r['selected_correct'] for r in entity)/max(1,sum(r['candidate_available'] for r in entity)),'both_selected_regions_correct':both,'logical_target_correct':sum(r['logical_target_correct'] for r in qas),'grounded_target_correct':sum(r['grounded_target_correct'] for r in qas),'entity_denominator':40,'family_denominator':20,'qa_denominator':40}
    d=BASE/route;write(d/'binding_metrics.json' if route!='p2_joint_assignment' else d/'joint_binding_metrics.json',metrics);writel(d/'entity_evaluation.jsonl',entity);writel(d/'qa_binding_evaluation.jsonl',qas)
    r4={(r['family_id'],r['logical_ref']):r for r in rows(ROOT/'outputs/region_bottleneck_joint_repair/candidate_results.jsonl') if r['generator']=='strong'}
    same={f for f in {k[0] for k in pool} if r4[f,'A']['referent_spec']['category_constraint']['value'].casefold()==r4[f,'B']['referent_spec']['category_constraint']['value'].casefold()}
    groups={'same_category':[r for r in entity if r['family_id'] in same],'other':[r for r in entity if r['family_id'] not in same]}
    write(d/'same_category_analysis.json',{k:{'correct':sum(r['selected_correct'] for r in v),'n':len(v)} for k,v in groups.items()})
    bins={'singleton':[],'2-4':[],'5-16':[],'>16':[]}
    for r in entity:
        n=len(pool[r['family_id'],r['logical_ref']]['candidates']);name='singleton' if n==1 else '2-4' if n<=4 else '5-16' if n<=16 else '>16';bins[name].append(r)
    write(d/'candidate_count_analysis.json',{k:{'correct':sum(r['selected_correct'] for r in v),'n':len(v)} for k,v in bins.items()})
    write(d/'binding_failure_taxonomy.json',{k:sum(r['failure']==k for r in entity) for k in ('NO_USABLE_CANDIDATE','USABLE_CANDIDATE_NOT_SELECTED')})
    print(route,metrics)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('route',choices=['p0','p1','p2']);p.add_argument('--best',choices=['p0','p1']);a=p.parse_args()
    if a.route=='p2':
        assert a.best
        route='p0_qwen_independent' if a.best=='p0' else 'p1_objembed'
        indep=rows(BASE/route/'selected_regions.jsonl');out=joint(indep);evaluate(out,'p2_joint_assignment')
    else:
        route='p0_qwen_independent' if a.route=='p0' else 'p1_objembed';out=select(a.route);evaluate(out,route)
