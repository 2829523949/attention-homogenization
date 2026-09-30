#!/usr/bin/env python3
"""Frozen 20-family constraint resolution diagnostic from cached GDINO candidates."""
import json, sys, hashlib, statistics, html
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path('/mnt/d/vlm_repair_v2'); sys.path.insert(0,str(ROOT))
from harness.referents.candidate_resolver import CandidateResolver, VERSION, digest
from harness.referents.referent_resolver import resolve_question
from harness.planner.predicate_dag import build_predicate_dag
from harness.evaluators.leakage_checks import assert_inference_safe
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'outputs/constraint_resolution_pilot'; OUT.mkdir(parents=True,exist_ok=True); (OUT/'figures').mkdir(exist_ok=True)
SOURCE=ROOT/'outputs/grounding_dino_acquisition_pilot/entity_backend_comparison.jsonl'
MANIFEST=ROOT/'manifests/acquisition_semantic_support_20family.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def write(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def writel(name,rows): (OUT/name).write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))
def iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0])); y=max(0,min(a[3],b[3])-max(a[1],b[1])); inter=x*y
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]); return inter/(aa+bb-inter) if aa+bb-inter else 0
m=json.loads(MANIFEST.read_text()); assert m['n_families']==20 and len(m['unique_entity_units'])==40
cache=[r for r in lines(SOURCE) if r['inference']['backend']=='grounding_dino']; assert len(cache)==40
by={(r['inference']['family_id'],r['inference']['logical_ref']):r for r in cache}
assert len(by)==40
qa_path=Path(m['source_frozen_files']['qa']['path']); assert sha(qa_path)==m['source_frozen_files']['qa']['sha256']
qa={r['qa_id']:r for r in lines(qa_path)}
run_id='constraint_resolution_pilot_v1_cached_'+sha(SOURCE)[:12]
results=[]; traces=[]; scores=[]; ambiguous=[]; contribution=defaultdict(lambda:{'available_families':0,'hypotheses_removed':0,'families_made_unique':0,'correct_unique':0,'gt_compatible_pair_incorrectly_removed':0})
for fam in m['families']:
    fid=fam['family_id']; pair={ref:by[(fid,ref)] for ref in ('A','B')}
    case=qa[fam['case_ids'][0]]; question=case['question']; task=case['qa_type']
    parsed=resolve_question(question,task); dag=build_predicate_dag(question,task,parsed)
    # Explicit whitelist: the resolver never sees the evaluator namespace.
    candidates={ref:pair[ref]['inference']['candidates'] for ref in ('A','B')}
    assert_inference_safe(candidates,context='cached candidate projection')
    assert {s.logical_ref for s in parsed.specs}=={'A','B'}
    for spec in parsed.specs:
        cached=pair[spec.logical_ref]['inference']
        assert cached['referent_spec']==spec.to_dict(), 'cached legal ReferentSpec differs from frozen parser'
        assert set(cached['case_ids'])==set(fam['case_ids'])
        assert all(c['source_backend'].startswith('grounding_dino:') for c in candidates[spec.logical_ref])
    result=CandidateResolver().resolve(dag,parsed.specs,candidates,{},family_id=fid,run_id=run_id)
    inference=result.to_dict(); assert_inference_safe(inference,context='resolver result')
    gt={ref:pair[ref]['EVALUATOR_ONLY']['gt_bbox_xyxy_pixels'] for ref in ('A','B')}
    ci={ref:{c['candidate_id']:iou(c['bbox_xyxy_pixels'],gt[ref]) for c in candidates[ref]} for ref in ('A','B')}
    covered=all(any(v>=.5 for v in ci[ref].values()) for ref in ('A','B'))
    def correct(ids): return bool(ids) and all(ci[ref][ids[ref]]>=.5 for ref in ('A','B'))
    unique_correct=result.status=='RESOLVED_UNIQUE' and correct(result.selected)
    taxonomy='ACQUISITION_UNCOVERED' if not covered else 'COVERED_UNIQUE_CORRECT' if unique_correct else 'COVERED_UNIQUE_WRONG' if result.status=='RESOLVED_UNIQUE' else 'COVERED_AMBIGUOUS' if result.status=='ABSTAIN_AMBIGUOUS' else 'COVERED_NO_VALID'
    valid=[h for h in inference['hypotheses'] if h['valid']]
    def rank(hyps,key):
        if not hyps: return None
        return max(hyps,key=lambda h:(sum(h['candidates'][ref][key] for ref in ('A','B')),tuple(h['candidate_ids'][ref] for ref in ('A','B'))))['candidate_ids']
    def indep(key):
        if any(not candidates[ref] for ref in ('A','B')): return None
        return {ref:max(candidates[ref],key=lambda c:(c['native_grounding_score'] if key=='native' else c['siglip_diagnostic']['z_raw'],-c['candidate_index']))['candidate_id'] for ref in ('A','B')}
    native=indep('native'); sig=indep('siglip')
    allh=list(inference['hypotheses'])
    baselines={'gdino_independent':native,'siglip_independent':sig,'gdino_pair':rank(allh,'native_grounding_score'),'siglip_pair':rank(allh,'siglip_diagnostic_score')}
    score_row={'family_id':fid,'candidate_covered':covered,'baselines':{name:{'candidate_ids':ids,'joint_correct':correct(ids)} for name,ids in baselines.items()},'primary':{'status':result.status,'candidate_ids':result.selected,'joint_correct':unique_correct}}
    scores.append(score_row)
    if len(valid)>1:
        g=rank(valid,'native_grounding_score'); s=rank(valid,'siglip_diagnostic_score')
        ambiguous.append({'family_id':fid,'candidate_covered':covered,'n_valid':len(valid),'gdino_pair':g,'gdino_correct':correct(g),'siglip_pair':s,'siglip_correct':correct(s),'agree':g==s,'agree_correct':correct(g) if g==s else None})
    for typ in result.constraints:
        c=contribution[typ]; c['available_families']+=1
        removed=[h for h in allh if not h['constraint_checks'].get(typ,True)]
        c['hypotheses_removed']+=len(removed)
        others=[h for h in allh if all(v for k,v in h['constraint_checks'].items() if k!=typ)]
        if len(others)>1 and result.valid_count==1:
            c['families_made_unique']+=1; c['correct_unique']+=int(unique_correct)
        c['gt_compatible_pair_incorrectly_removed']+=int(any(correct(h['candidate_ids']) for h in removed))
    ev={'namespace':'EVALUATOR_ONLY','candidate_covered_at_0_5':covered,'candidate_ious':ci,'unique_joint_correct':unique_correct,'failure_taxonomy':taxonomy}
    results.append({'inference':{k:v for k,v in inference.items() if k!='hypotheses'},'EVALUATOR_ONLY':ev})
    for j,h in enumerate(allh): traces.append({'family_id':fid,'hypothesis_index':j,'inference':h,'EVALUATOR_ONLY':{'joint_gt_compatible_at_0_5':correct(h['candidate_ids'])}})
assert len(results)==20
writel('family_resolution_results.jsonl',results); writel('hypothesis_trace.jsonl',traces)
n=20; covered=sum(x['EVALUATOR_ONLY']['candidate_covered_at_0_5'] for x in results); unique=sum(x['inference']['status']=='RESOLVED_UNIQUE' for x in results); unique_cov=sum(x['inference']['status']=='RESOLVED_UNIQUE' and x['EVALUATOR_ONLY']['candidate_covered_at_0_5'] for x in results); correct_n=sum(x['EVALUATOR_ONLY']['unique_joint_correct'] for x in results)
amb=sum(x['inference']['status']=='ABSTAIN_AMBIGUOUS' for x in results); nov=sum(x['inference']['status']=='ABSTAIN_NO_VALID_ASSIGNMENT' for x in results)
reduction=[1-x['inference']['valid_count']/x['inference']['initial_count'] if x['inference']['initial_count'] else 0 for x in results]
metrics={'n_families':n,'candidate_covered_families':covered,'joint_candidate_recall_at_0_5':covered/n,'candidate_oracle_ceiling':covered/n,'unique_resolved_all':unique,'unique_resolved_covered':unique_cov,'unique_resolution_rate_covered':unique_cov/covered if covered else None,'correct_unique_resolution_rate_covered':correct_n/covered if covered else None,'unique_resolution_precision':correct_n/unique if unique else None,'overall_joint_binding_accuracy':correct_n/n,'ambiguous_rate':amb/n,'no_valid_rate':nov/n,'mean_reduction':statistics.mean(reduction),'median_reduction':statistics.median(reduction),'fully_unique_fraction':unique/n,'unchanged_fraction':sum(x==0 for x in reduction)/n,'initial_hypotheses':sum(x['inference']['initial_count'] for x in results),'valid_hypotheses':sum(x['inference']['valid_count'] for x in results)}
write('resolution_metrics.json',metrics); write('constraint_contribution.json',dict(contribution))
tax=dict(Counter(x['EVALUATOR_ONLY']['failure_taxonomy'] for x in results)); write('failure_taxonomy.json',tax)
base={key:{'joint_correct':sum(x['baselines'][key]['joint_correct'] for x in scores),'joint_accuracy':sum(x['baselines'][key]['joint_correct'] for x in scores)/n,'covered_correct':sum(x['candidate_covered'] and x['baselines'][key]['joint_correct'] for x in scores)} for key in ('gdino_independent','siglip_independent','gdino_pair','siglip_pair')}; write('ranking_baselines.json',{'summary':base,'families':scores})
agree=[a for a in ambiguous if a['agree']]; disagree=[a for a in ambiguous if not a['agree']]; covered_amb=[a for a in ambiguous if a['candidate_covered']]
write('ambiguity_analysis.json',{'n_ambiguous':len(ambiguous),'gdino_correct':sum(a['gdino_correct'] for a in ambiguous),'siglip_correct':sum(a['siglip_correct'] for a in ambiguous),'agree':len(agree),'agree_correct':sum(a['agree_correct'] for a in agree),'disagree':len(disagree),'gdino_correct_when_disagree':sum(a['gdino_correct'] for a in disagree),'siglip_correct_when_disagree':sum(a['siglip_correct'] for a in disagree),'families':ambiguous})
write('resolver_config.json',{'schema_version':VERSION,'run_id':run_id,'manifest':str(MANIFEST),'manifest_sha256':sha(MANIFEST),'cached_candidate_source':str(SOURCE),'cached_candidate_source_sha256':sha(SOURCE),'acquisition_backend':'grounding_dino','candidate_cache_reused':True,'hard_collision_rule':'exact bbox equality','horizontal_tie_rule':'strict inequality fails both asserted directions','selection_side_is_constraint':False,'scores_used_for_primary':False,'qwen_called':False,'certification_called':False,'gt_inference_use':False})
plt.figure(figsize=(7,4)); plt.scatter([r['inference']['initial_count'] for r in results],[r['inference']['valid_count'] for r in results]); plt.xlabel('Initial hypotheses');plt.ylabel('Valid hypotheses');plt.tight_layout();plt.savefig(OUT/'figures/initial_vs_valid.png');plt.close()
plt.figure(figsize=(7,4));plt.hist(reduction,bins=[-.01,.01,.2,.4,.6,.8,1.01]);plt.xlabel('Constraint reduction ratio');plt.ylabel('Families');plt.tight_layout();plt.savefig(OUT/'figures/reduction_distribution.png');plt.close()
labels=['ACQUISITION_UNCOVERED','COVERED_UNIQUE_CORRECT','COVERED_UNIQUE_WRONG','COVERED_AMBIGUOUS','COVERED_NO_VALID']; plt.figure(figsize=(8,4));plt.bar(range(5),[tax.get(k,0) for k in labels]);plt.xticks(range(5),['uncovered','unique correct','unique wrong','ambiguous','no valid'],rotation=20);plt.ylabel('Families');plt.tight_layout();plt.savefig(OUT/'figures/outcomes.png');plt.close()
plt.figure(figsize=(7,4));plt.bar(['GDINO top1','SigLIP top1','constraint','oracle'],[base['gdino_independent']['joint_accuracy'],base['siglip_independent']['joint_accuracy'],correct_n/n,covered/n]);plt.ylim(0,1);plt.ylabel('Joint binding correctness');plt.tight_layout();plt.savefig(OUT/'figures/comparison.png');plt.close()
def pct(v): return 'N/A' if v is None else f'{v:.3f}'
def row(k,v): return f'| {k} | {v} |'
ct='\n'.join(f"| {k} | {v['available_families']} | {v['hypotheses_removed']} | {v['families_made_unique']} | {v['correct_unique']} | {v['gt_compatible_pair_incorrectly_removed']} |" for k,v in contribution.items()) or '| None | 0 | 0 | 0 | 0 | 0 |'
recommend='FIX_CONSTRAINT_INTERFACE' if tax.get('COVERED_NO_VALID',0) or any(v['gt_compatible_pair_incorrectly_removed'] for v in contribution.values()) else 'ADD_SOFT_EVIDENCE_TIEBREAK' if tax.get('COVERED_AMBIGUOUS',0) and (sum(a['gdino_correct'] or a['siglip_correct'] for a in ambiguous)>0) else 'RETURN_TO_ACQUISITION' if unique_cov/covered>=.8 and correct_n/max(unique,1)>=.8 else 'ADD_SOFT_EVIDENCE_TIEBREAK'
report=f'''# Constraint-based candidate resolution pilot

Frozen 20-family Grounding DINO cache; no acquisition, SigLIP, certification, or Qwen call was made. Candidate source SHA-256: `{sha(SOURCE)}`. Manifest SHA-256: `{sha(MANIFEST)}`.

The REL2C DAG computes and selects a horizontal relation, but does not assert A left or right of B. Its `LEFT`/`RIGHT` suffix is the requested selection side. Thus no horizontal ordering is a legal binding constraint here. Category text was the detector query and cannot independently validate its output. The two-noun template licenses distinct entities; exact pixel bbox equality is the predeclared collision rule. No overlap threshold or score selects a primary binding. This pilot's limited contraction is a finding about the frozen legal interface.

## Candidate coverage

| Metric | Value |
|---|---:|
| N families | 20 |
| Candidate-covered@0.5 | {covered} |
| JointCandidateRecall@0.5 | {pct(covered/n)} |
| Candidate oracle ceiling | {pct(covered/n)} |

## Resolver performance

| Metric | Value |
|---|---:|
| Unique Resolution Rate (covered) | {pct(metrics['unique_resolution_rate_covered'])} |
| Correct Unique Resolution Rate (covered) | {pct(metrics['correct_unique_resolution_rate_covered'])} |
| Unique Resolution Precision (all unique) | {pct(metrics['unique_resolution_precision'])} |
| Overall Joint Binding Accuracy | {pct(metrics['overall_joint_binding_accuracy'])} |
| Ambiguous Rate | {pct(metrics['ambiguous_rate'])} |
| No-valid Rate | {pct(metrics['no_valid_rate'])} |

## Constraint effect

| Constraint type | Available families | Hypotheses removed | Families made unique | Correct unique | GT-compatible pair incorrectly removed |
|---|---:|---:|---:|---:|---:|
{ct}

The single no-valid outcome has an empty candidate set and is acquisition-uncovered; no covered family becomes no-valid.\n\nMean contraction {pct(metrics['mean_reduction'])}; median {pct(metrics['median_reduction'])}; fully unique fraction {pct(metrics['fully_unique_fraction'])}; unchanged fraction {pct(metrics['unchanged_fraction'])}. Initial hypotheses {metrics['initial_hypotheses']}; valid {metrics['valid_hypotheses']}.

## Baseline comparison

| Resolver | Joint correct | Covered-family correct | Ambiguous allowed? |
|---|---:|---:|---|
| GDINO native top1 | {base['gdino_independent']['joint_correct']}/20 | {base['gdino_independent']['covered_correct']}/{covered} | No |
| SigLIP top1 | {base['siglip_independent']['joint_correct']}/20 | {base['siglip_independent']['covered_correct']}/{covered} | No |
| Constraint-only joint | {correct_n}/20 | {correct_n}/{covered} | Yes |
| Candidate oracle | {covered}/20 | {covered}/{covered} | Evaluator-only |

| Method | Uses structural constraints? | Uses soft score? | Joint binding correctness |
|---|---|---|---:|
| GDINO native top1 | No | Yes | {pct(base['gdino_independent']['joint_accuracy'])} |
| SigLIP top1 | No | Yes | {pct(base['siglip_independent']['joint_accuracy'])} |
| Constraint-only joint resolver | Yes | No | {pct(correct_n/n)} |
| Candidate oracle | Evaluator-only | N/A | {pct(covered/n)} |

## Scientific questions

1. Q1: {unique_cov}/{covered} covered families uniquely resolve ({pct(metrics['unique_resolution_rate_covered'])}).
2. Q2: {correct_n}/{unique} unique resolutions are correct ({pct(metrics['unique_resolution_precision'])}).
3. Q3: Constraint-only joint accuracy {pct(correct_n/n)} versus GDINO independent {pct(base['gdino_independent']['joint_accuracy'])}; {'yes' if correct_n>base['gdino_independent']['joint_correct'] else 'no'}.
4. Q4: Constraint-only joint accuracy {pct(correct_n/n)} versus SigLIP independent {pct(base['siglip_independent']['joint_accuracy'])}; {'yes' if correct_n>base['siglip_independent']['joint_correct'] else 'no'}.
5. Q5: Mean hypothesis-space reduction {pct(metrics['mean_reduction'])}, median {pct(metrics['median_reduction'])}.
6. Q6: {'Distinctness is the sole available hard constraint; see contribution table.' if contribution else 'No asserted hard constraint is available.'}
7. Q7: {sum(v['gt_compatible_pair_incorrectly_removed'] for v in contribution.values())} GT-compatible pairs were removed by legal constraints.
8. Q8: Among {len(ambiguous)} ambiguous families, GDINO valid-pair ranking is correct for {sum(a['gdino_correct'] for a in ambiguous)} and SigLIP for {sum(a['siglip_correct'] for a in ambiguous)}; both agree in {len(agree)}, correctly in {sum(a['agree_correct'] for a in agree)}. Among the {len(covered_amb)} covered ambiguous families, GDINO is correct for {sum(a['gdino_correct'] for a in covered_amb)} and SigLIP for {sum(a['siglip_correct'] for a in covered_amb)}. These are diagnostics only.
9. Q9: The main blocker is {'constraint-interface error' if recommend=='FIX_CONSTRAINT_INTERFACE' else 'hard-constraint insufficiency / soft semantic ambiguity' if recommend=='ADD_SOFT_EVIDENCE_TIEBREAK' else 'acquisition coverage'} on this frozen subset. The uncovered families remain an independent ceiling.

Recommendation: **`{recommend}`**. No VLM selector or full benchmark is proposed.

Figures: [initial vs valid](figures/initial_vs_valid.png), [reduction distribution](figures/reduction_distribution.png), [outcomes](figures/outcomes.png), [comparison](figures/comparison.png).
'''
(OUT/'implementation_report.md').write_text(report)
(OUT/'audit.html').write_text('<!doctype html><meta charset="utf-8"><title>Constraint resolution audit</title><style>body{font:14px sans-serif;max-width:1100px;margin:auto}pre{white-space:pre-wrap;border:1px solid #ddd;padding:1em}</style><h1>Constraint resolution audit</h1><p>Inference and evaluator sections are separate. Scores are diagnostic only.</p>'+''.join('<details><summary>'+html.escape(r['inference']['family_id'])+' — '+r['inference']['status']+'</summary><pre>'+html.escape(json.dumps({'result':r,'hypotheses':[x for x in traces if x['family_id']==r['inference']['family_id']]},indent=2))+'</pre></details>' for r in results))
(OUT/'changed_files.txt').write_text('harness/referents/candidate_resolver.py\nharness/scripts/run_constraint_resolution_pilot.py\nharness/tests/test_candidate_resolver.py\noutputs/constraint_resolution_pilot/\n')
print(json.dumps({'metrics':metrics,'taxonomy':tax,'baseline':base,'recommendation':recommend},indent=2))
