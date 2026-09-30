#!/usr/bin/env python3
"""Build a self-contained development-set comparison from completed artifacts."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path('/mnt/d/vlm_repair_v2');sys.path.insert(0,str(ROOT))
from harness.scripts.run_high_recall_binding_repair import iou
BASE=ROOT/'outputs/high_recall_binding_repair';OUT=BASE/'final_comparison';OUT.mkdir(exist_ok=True,parents=True)
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def j(p):return json.loads(p.read_text())
def metric(route):
    d=BASE/route
    if not(d/'downstream_results.jsonl').exists():return None
    b=j(d/('joint_binding_metrics.json' if route=='p2_joint_assignment' else 'binding_metrics.json')) if route!='p3_direct_grounding' else j(d/'binding_metrics.json')
    q=rows(d/'downstream_results.jsonl')
    return {'candidate_recall':b.get('candidate_available'),'selected_region_correct':b.get('selected_region_correct',b.get('region_correct')),'selection_efficiency':b.get('selection_efficiency'),'grounded_target_correct':sum(x['grounded_target_correct'] for x in q),'minimal_state_sufficient':sum(x['minimal_state_sufficient'] for x in q),'clean_final':sum(x['clean_final_correct'] for x in q),'raw_final':sum(x['final_correct'] for x in q),'logical_target_correct':sum(x['logical_target_correct'] for x in q)}
def main():
    names={'p0_qwen_independent':'P0 Qwen independent','p1_objembed':'P1 ObjEmbed','p2_joint_assignment':'P2 Joint assignment','p3_direct_grounding':'P3 Direct grounding'}
    data={k:metric(k) for k in names};(OUT/'route_metrics.json').write_text(json.dumps(data,indent=2)+'\n')
    oracle=j(OUT/'candidate_oracle.json')['depths'];gt={}
    for r in rows(ROOT/'outputs/grounding_dino_acquisition_pilot/entity_backend_comparison.jsonl'):
        x=r['inference']
        if x['backend']=='grounding_dino':gt[x['family_id'],x['logical_ref']]=r['EVALUATOR_ONLY']['gt_bbox_xyxy_pixels']
    hist=rows(ROOT/'outputs/strong_integrated_repair/binding_results.jsonl');hist_correct=sum(iou(r['selected_bbox_xyxy'],gt[r['family_id'],r['logical_ref']])>=.5 for r in hist)
    lines=['# High-recall referent–region binding repair','', '**Development pilot only:** the locked 20-family/40-REL2C-QA subset has been repeatedly inspected. These are not held-out paper results.','',
    '## Route comparison','','| Route | Candidate recall | Selected region correct | Selection efficiency | Grounded target correct | Minimal state sufficient | Clean final | Raw final |','|---|---:|---:|---:|---:|---:|---:|---:|',
    f'| Historical Integrated | R4 {oracle["R4"]["entity"]}/40; SAM3 fallback | {hist_correct}/40 | — | 26/40 | 20/40 | 20/40 | 25/40 |']
    for k,label in names.items():
        m=data[k]
        if m is None:lines.append(f'| {label} | {"—" if k=="p3_direct_grounding" else "pending"} | — | — | pending | pending | pending | pending |');continue
        cr='—' if m['candidate_recall'] is None else f'{m["candidate_recall"]}/40'
        sc='—' if m['selected_region_correct'] is None else f'{m["selected_region_correct"]}/40'
        eff='—' if m['selection_efficiency'] is None else f'{m["selection_efficiency"]:.3f}'
        lines.append(f'| {label} | {cr} | {sc} | {eff} | {m["grounded_target_correct"]}/40 | {m["minimal_state_sufficient"]}/40 | {m["clean_final"]}/40 | {m["raw_final"]}/40 |')
    lines+=['','## Recovery provenance','',
        'Frozen candidate pool, P0, P1, P2, and historical baselines: `RECOVERED_EXISTING`. P3 direct predictions and downstream results: `RESUMED_AFTER_CRASH`. The incomplete EGM download chunk: `EXCLUDED_PARTIAL`; it was not used as a checkpoint shard.','']
    lines+=['','## Candidate depth (evaluator only)','','| Candidate pool | Entity oracle recall | Both-entity family recall |','|---|---:|---:|']
    for k in ['R4','SAM3 top16','SAM3 top32','SAM3 top64','SAM3 top128','SAM3 top200','Union top64']:
        v=oracle[k];lines.append(f'| {k} | {v["entity"]}/40 | {v["both_entities"]}/20 |')
    lines+=['','## Selector comparison','','| Matcher | Selected correct | Selection efficiency | Grounded target correct |','|---|---:|---:|---:|','| Historical SAM3 Qwen grid | ≈16/40 reference | ≈16/31 | — |']
    for k,label in list(names.items())[:3]:
        m=data[k];lines.append(f'| {label} | {m["selected_region_correct"]}/40 | {m["selection_efficiency"]:.3f} | {m["grounded_target_correct"]}/40 |')
    lines+=['','## Same-category stress test','','No A/B pair in this locked subset shares an exact frozen ReferentSpec category. The evaluator-side VG annotations also show no second named instance with the same exact referent category in these 20 images. This subset has denominator zero; a same-category effect cannot be estimated.','', '| Matcher | Same-category selected correct | Other selected correct |','|---|---:|---:|',f'| Historical Integrated | 0/0 | {hist_correct}/40 |']
    for k,label in list(names.items())[:3]:
        v=j(BASE/k/'same_category_analysis.json');lines.append(f'| {label} | {v["same_category"]["correct"]}/{v["same_category"]["n"]} | {v["other"]["correct"]}/{v["other"]["n"]} |')
    lines+=['','## Pool-size strata','','Every deduplicated primary pool contains more than 16 candidates (range 52–66), so singleton, 2–4, and 5–16 strata are empty. The pilot cannot estimate degradation across those bins.','']
    best=data['p2_joint_assignment'];q=rows(BASE/'p2_joint_assignment/downstream_results.jsonl');a=[r for r in q if r['logical_target_correct']];b=[r for r in a if r['grounded_target_correct']];c=[r for r in b if r['selected_target_bbox_xyxy'] is not None];d=[r for r in c if r['target_attribute_correct']];e=[r for r in d if r['minimal_state_sufficient']];f=[r for r in e if r['clean_final_correct']]
    lines+=['## P2 monotonic survival funnel','', 'IsolationValid here means a valid target-local view from either a validated SAM3 mask or the frozen exact-bbox fallback. Mask use alone is reported separately and is not required by the frozen state-sufficiency criterion.','',f'40 → logical {len(a)} → grounded {len(b)} → local view {len(c)} → attribute correct {len(d)} → state sufficient {len(e)} → clean final {len(f)}.','',f'Unconditional component counts: mask isolation used {sum(r["target_isolated"] for r in q)}/40; attribute correct {sum(r["target_attribute_correct"] for r in q)}/40.','']
    if data['p3_direct_grounding'] is not None:
        q3=rows(BASE/'p3_direct_grounding/downstream_results.jsonl')
        a3=[r for r in q3 if r['logical_target_correct']]
        b3=[r for r in a3 if r['grounded_target_correct']]
        c3=[r for r in b3 if r['selected_target_bbox_xyxy'] is not None]
        d3=[r for r in c3 if r['target_attribute_correct']]
        e3=[r for r in d3 if r['minimal_state_sufficient']]
        f3=[r for r in e3 if r['clean_final_correct']]
        lines+=['## P3 monotonic survival funnel','',f'40 → logical {len(a3)} → grounded {len(b3)} → local view {len(c3)} → attribute correct {len(d3)} → state sufficient {len(e3)} → clean final {len(f3)}.','']
    lines+=['## Paired changes','',f'P0→P1 entity: {j(BASE/"p1_objembed/paired_vs_qwen.json")["entity_region"]}.',f'P1→P2 grounded target: {j(BASE/"p2_joint_assignment/paired_vs_best_independent.json")["grounded_target"]}.',f'P1→P2 minimal state: {j(BASE/"p2_joint_assignment/paired_vs_best_independent.json")["minimal_state"]}.','', '## Current interpretation','']
    if data['p3_direct_grounding'] is None:
        lines+=['The top-64 union contains usable candidates for 38/40 referents, but the best proposal binder selects 26/40 correct regions. ObjEmbed materially outperforms independent Qwen on identical proposals; the one-to-one constraint resolves one collision and rescues one grounded target without improving entity-level selected-region count. Binding remains the dominant proposal-route bottleneck. P3 direct grounding is gated on and remains pending until its official checkpoint and evaluation complete.','']
    else:
        lines+=['The top-64 union contains usable candidates for 38/40 referents. ObjEmbed selected 26/40 correct regions; the one-to-one assignment improved grounded targets from 24/40 to 25/40 and clean final answers from 17/40 to 18/40. The gated P3 EGM direct grounding route selected 18/40 correct regions and reached 15/40 grounded targets and 13/40 clean final answers. It did not improve on P2. The highest grounded-target and clean-final counts in this inspected development pilot remain the historical integrated route at 26/40 and 20/40. These are descriptive pilot results, not held-out generalization evidence.','']
    (OUT/'comparison_report.md').write_text('\n'.join(lines)+'\n')
    print('wrote',OUT/'comparison_report.md')
if __name__=='__main__':main()
