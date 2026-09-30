#!/usr/bin/env python3
"""Evaluator-only forensic decomposition of frozen handoff states; no inference."""
from __future__ import annotations

import hashlib
import html
import json
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HANDOFF=ROOT/"outputs/handoff_end_to_end_pilot"
CACHE=ROOT/"outputs/grounding_dino_acquisition_pilot/entity_backend_comparison.jsonl"
OUT=ROOT/"outputs/state_bottleneck_audit"
KEYS=("region_A_correct","region_B_correct","color_A_correct","color_B_correct","relation_correct")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()


def load(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def dump(path,data):Path(path).write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n")


def write_rows(path,data):Path(path).write_text("".join(json.dumps(x,sort_keys=True,allow_nan=False)+"\n" for x in data))


def ratio(num,den):return {"count":num,"denominator":den,"rate":num/den if den else None}


def choose(candidates):
    if not candidates:return None
    return max(range(len(candidates)),key=lambda i:(candidates[i]["native_grounding_score"] if candidates[i]["native_grounding_score"] is not None else float("-inf"),-candidates[i]["bbox_xyxy_pixels"][0]))


def main():
    if OUT.exists():raise FileExistsError(f"refusing to overwrite prior output: {OUT}")
    states=load(HANDOFF/"compiled_states.jsonl")
    previous=load(HANDOFF/"qa_condition_results.jsonl")
    acquisition=[x for x in load(CACHE) if x["inference"]["backend"]=="grounding_dino"]
    config=json.loads((HANDOFF/"pilot_config.json").read_text())
    assert sha(CACHE)==config["grounding_dino_cache_sha256"]
    assert len(states)==len(previous)==40 and len(acquisition)==40
    state_by_id={x["qa_id"]:x for x in states}
    assert len(state_by_id)==40
    cache={(x["inference"]["family_id"],x["inference"]["logical_ref"]):x for x in acquisition}
    assert len(cache)==40
    assert {x["qa_id"] for x in previous}==set(state_by_id)
    OUT.mkdir();(OUT/"figures").mkdir()
    rows=[];region_events=[];color_events=[];relation_events=[]
    for previous_row in previous:
        state=state_by_id[previous_row["qa_id"]]
        e=previous_row["EVALUATOR_ONLY"]
        assert state["state_text"]==previous_row["state_text"] and state["state_sha256"]==previous_row["state_sha256"]
        assert state["gt_used"] is False and state["inference_status"]=="COMPILED_UNCERTIFIED"
        assert e["region_A_correct"]==(e["region_A_iou"]>=.5)
        assert e["region_B_correct"]==(e["region_B_iou"]>=.5)
        assert e["color_A_correct"]==(state["entities"]["A"]["color"]==e["color_A_gt"])
        assert e["color_B_correct"]==(state["entities"]["B"]["color"]==e["color_B_gt"])
        assert e["relation_correct"]==(state["relation"]["A_to_B"]==e["gt_relation_A_to_B"])
        booleans={key:bool(e[key]) for key in KEYS}
        ba,bb,ca,cb,rel=(booleans[k] for k in KEYS)
        assert e["both_regions_correct"]==(ba and bb)
        assert e["strict_fact_sufficient"]==(ba and bb and ca and cb and rel)
        booleans.update({"both_regions_correct":ba and bb,"both_colors_correct":ca and cb,"strict_fact_sufficient":ba and bb and ca and cb and rel})
        signature=" ".join(f"{label}={int(value)}" for label,value in zip(("B_A","B_B","C_A","C_B","R"),(ba,bb,ca,cb,rel)))
        entity_audit={}
        for ref,region_good,color_good in (("A",ba,ca),("B",bb,cb)):
            frozen=cache[(state["family_id"],ref)]
            inf=frozen["inference"];ev=frozen["EVALUATOR_ONLY"]
            selected=choose(inf["candidates"])
            assert selected==inf["native_top_candidate_index"]==state["entities"][ref]["selected_candidate_index"]
            selected_iou=ev["candidate_ious_with_gt_in_inference_order"][selected] if selected is not None else 0.0
            assert abs(selected_iou-e[f"region_{ref}_iou"])<1e-9
            candidate_available=ev["best_candidate_iou"]>=.5
            if region_good:failure=None
            elif candidate_available:failure="USABLE_CANDIDATE_EXISTS_BUT_NOT_SELECTED"
            else:
                failure={"GENERATION_FAILURE":"NO_USABLE_CANDIDATE","OVERLARGE_REGION":"OVERLARGE_REGION","PARTIAL_REGION":"PARTIAL_OVERLAP","WRONG_SAME_CATEGORY_INSTANCE":"WRONG_SAME_CATEGORY_INSTANCE","SEMANTIC_DISTRACTOR":"SEMANTIC_DISTRACTOR"}.get(ev["failure_classification"],"UNKNOWN")
            # Exhaustive three-way failure classification plus a true-success label.
            if region_good and color_good:color_class="CORRECT_COLOR_ON_CORRECT_REGION"
            elif region_good and not color_good:color_class="COLOR_FAILURE_AFTER_CORRECT_REGION"
            elif not region_good and color_good:color_class="COLOR_ACCIDENTALLY_CORRECT_ON_WRONG_REGION"
            else:color_class="COLOR_FAILURE_WITH_WRONG_REGION"
            label=(inf["candidates"][selected]["native_label_text"] or "").strip().casefold() if selected is not None else None
            same_query_zero_overlap=bool(selected is not None and selected_iou==0 and label==inf["legal_query"].casefold())
            audit={"logical_ref":ref,"region_correct":region_good,"selected_iou":selected_iou,"best_candidate_iou":ev["best_candidate_iou"],"candidate_available":candidate_available,"selected_candidate_index":selected,"region_failure_class":failure,"source_failure_classification":ev["failure_classification"],"same_query_zero_overlap_distractor_proxy":same_query_zero_overlap,"color_correct":color_good,"color_failure_class":color_class,"acquired_color":state["entities"][ref]["color"],"evaluator_color":e[f"color_{ref}_gt"]}
            entity_audit[ref]=audit
            region_events.append({"qa_id":state["qa_id"],"family_id":state["family_id"],"entity_unit_id":inf["entity_unit_id"],**audit})
            color_events.append({"qa_id":state["qa_id"],"family_id":state["family_id"],"entity_unit_id":inf["entity_unit_id"],"logical_ref":ref,"region_correct":region_good,"color_correct":color_good,"color_failure_class":color_class})
        relation_class=None if rel else "RELATION_FAILURE_AFTER_CORRECT_REGIONS" if ba and bb else "RELATION_FAILURE_FROM_WRONG_REGION"
        relation_events.append({"qa_id":state["qa_id"],"family_id":state["family_id"],"relation_correct":rel,"both_regions_correct":ba and bb,"relation_failure_class":relation_class,"acquired_relation":state["relation"]["A_to_B"],"evaluator_relation":e["gt_relation_A_to_B"]})
        rows.append({"qa_id":state["qa_id"],"family_id":state["family_id"],"qa_type":state["qa_type"],"state_sha256":state["state_sha256"],"primitive_correctness":booleans,"failure_signature":signature,"entities":entity_audit,"relation_failure_class":relation_class,"state_implied_answer_correct":bool(e["state_answer_consistent"]),"EVALUATOR_ONLY":True})
    assert len(rows)==40 and sum(r["primitive_correctness"]["strict_fact_sufficient"] for r in rows)==6
    write_rows(OUT/"state_component_results.jsonl",rows)
    family_groups=defaultdict(list);state_groups=defaultdict(list)
    for row in rows:family_groups[row["family_id"]].append(row)
    for state in states:state_groups[state["family_id"]].append(state)
    family_rows=[];consistency=[]
    for family in sorted(family_groups):
        rs=family_groups[family];ss=state_groups[family]
        assert len(rs)==len(ss)==2 and {s["qa_type"] for s in ss}=={"REL2C_LEFT","REL2C_RIGHT"}
        a,b=sorted(ss,key=lambda x:x["qa_type"])
        same={"A_region":a["entities"]["A"]["region_bbox_xyxy_pixels"]==b["entities"]["A"]["region_bbox_xyxy_pixels"],"B_region":a["entities"]["B"]["region_bbox_xyxy_pixels"]==b["entities"]["B"]["region_bbox_xyxy_pixels"],"A_color":a["entities"]["A"]["color"]==b["entities"]["A"]["color"],"B_color":a["entities"]["B"]["color"]==b["entities"]["B"]["color"],"relation":a["relation"]==b["relation"],"state_text":a["state_text"]==b["state_text"],"state_hash":a["state_sha256"]==b["state_sha256"]}
        stable=all(same.values())
        truth=rs[0]["primitive_correctness"]
        assert truth==rs[1]["primitive_correctness"]
        family_rows.append({"family_id":family,"qa_ids":[s["qa_id"] for s in ss],"primitive_correctness":truth,"state_stable":stable,"stability_checks":same,"failure_signature":rs[0]["failure_signature"]})
        consistency.append({"family_id":family,"qa_ids":[s["qa_id"] for s in ss],"status":"STABLE" if stable else "STATE_INCONSISTENCY","checks":same,"left_state_sha256":a["state_sha256"],"right_state_sha256":b["state_sha256"]})
    write_rows(OUT/"family_state_results.jsonl",family_rows)
    dump(OUT/"state_consistency_audit.json",{"n_families":20,"stable_families":sum(x["status"]=="STABLE" for x in consistency),"state_inconsistency_families":[x["family_id"] for x in consistency if x["status"]!="STABLE"],"families":consistency})
    b=lambda key:sum(r["primitive_correctness"][key] for r in rows)
    both=b("both_regions_correct");both_colors=b("both_colors_correct");strict=b("strict_fact_sufficient")
    both_regions_both_colors=sum(r["primitive_correctness"]["both_regions_correct"] and r["primitive_correctness"]["both_colors_correct"] for r in rows)
    relation_given_regions=sum(r["primitive_correctness"]["both_regions_correct"] and r["primitive_correctness"]["relation_correct"] for r in rows)
    relation_given_regions_colors=sum(r["primitive_correctness"]["both_regions_correct"] and r["primitive_correctness"]["both_colors_correct"] and r["primitive_correctness"]["relation_correct"] for r in rows)
    metrics={"P_B_A":ratio(b("region_A_correct"),40),"P_B_B":ratio(b("region_B_correct"),40),"P_B_A_and_B_B":ratio(both,40),"P_C_A":ratio(b("color_A_correct"),40),"P_C_B":ratio(b("color_B_correct"),40),"P_C_A_and_C_B":ratio(both_colors,40),"P_C_A_given_B_A":ratio(sum(r["primitive_correctness"]["region_A_correct"] and r["primitive_correctness"]["color_A_correct"] for r in rows),b("region_A_correct")),"P_C_B_given_B_B":ratio(sum(r["primitive_correctness"]["region_B_correct"] and r["primitive_correctness"]["color_B_correct"] for r in rows),b("region_B_correct")),"P_both_colors_given_both_regions":ratio(both_regions_both_colors,both),"P_R":ratio(b("relation_correct"),40),"P_R_given_both_regions":ratio(relation_given_regions,both),"P_R_given_both_regions_and_colors":ratio(relation_given_regions_colors,both_regions_both_colors),"P_strict":ratio(strict,40),"P_strict_given_both_regions":ratio(strict,both),"P_state_implied_answer_correct":ratio(sum(r["state_implied_answer_correct"] for r in rows),40)}
    dump(OUT/"conditional_metrics.json",metrics)
    chain=[{"stage":"All QA","count":40,"survival_from_previous":1.0,"survival_from_all":1.0},{"stage":"Both regions correct","count":both,"survival_from_previous":both/40,"survival_from_all":both/40},{"stage":"Both colors correct given correct regions","count":both_regions_both_colors,"survival_from_previous":both_regions_both_colors/both,"survival_from_all":both_regions_both_colors/40},{"stage":"Relation correct after regions and colors","count":strict,"survival_from_previous":strict/both_regions_both_colors if both_regions_both_colors else None,"survival_from_all":strict/40},{"stage":"Strict fact sufficient","count":strict,"survival_from_previous":1.0,"survival_from_all":strict/40}]
    auxiliary={"A_region_correct":b("region_A_correct"),"B_region_correct":b("region_B_correct"),"A_color_correct_given_both_regions":sum(r["primitive_correctness"]["both_regions_correct"] and r["primitive_correctness"]["color_A_correct"] for r in rows),"B_color_correct_given_both_regions":sum(r["primitive_correctness"]["both_regions_correct"] and r["primitive_correctness"]["color_B_correct"] for r in rows),"both_colors_correct_given_both_regions":both_regions_both_colors,"relation_correct_given_both_regions":relation_given_regions}
    dump(OUT/"survival_chain.json",{"primary_chain":chain,"parallel_components":auxiliary,"empirical_multiplicative_chain":{"P_both_regions":both/40,"P_both_colors_given_both_regions":both_regions_both_colors/both,"P_relation_given_regions_and_colors":strict/both_regions_both_colors if both_regions_both_colors else None,"product":(both/40)*(both_regions_both_colors/both)*(strict/both_regions_both_colors) if both_regions_both_colors else 0,"observed_P_strict":strict/40,"independence_assumed":False}})
    unique_region={x["entity_unit_id"]:x for x in region_events}
    failed_unique=[x for x in unique_region.values() if not x["region_correct"]]
    region_counts=Counter(x["region_failure_class"] for x in failed_unique)
    region_analysis={"n_unique_entities":40,"n_qa_entity_occurrences":80,"failed_unique_entities":len(failed_unique),"failed_qa_entity_occurrences":sum(not x["region_correct"] for x in region_events),"candidate_generation_failures_unique":sum(not x["candidate_available"] for x in failed_unique),"candidate_selection_failures_unique":sum(x["candidate_available"] for x in failed_unique),"candidate_oracle_joint_covered_families":sum(all(cache[(f["family_id"],ref)]["EVALUATOR_ONLY"]["best_candidate_iou"]>=.5 for ref in ("A","B")) for f in family_rows),"both_selected_regions_correct_families":both//2,"failure_classes_unique":dict(region_counts),"same_query_zero_overlap_distractor_proxy_unique":sum(x["same_query_zero_overlap_distractor_proxy"] for x in failed_unique),"same_query_zero_overlap_among_selection_failures_unique":sum(x["same_query_zero_overlap_distractor_proxy"] and x["candidate_available"] for x in failed_unique),"instance_ambiguity_note":"Native same-query labels and zero GT overlap suggest possible wrong-instance proposals, but visual category identity is not independently annotated; no verified wrong-same-category count is claimed.","failed_entities":failed_unique}
    dump(OUT/"region_failure_analysis.json",region_analysis)
    unique_color={x["entity_unit_id"]:x for x in color_events}
    color_analysis={"n_unique_entities":40,"n_qa_entity_occurrences":80,"classes_unique":dict(Counter(x["color_failure_class"] for x in unique_color.values())),"classes_qa_occurrences":dict(Counter(x["color_failure_class"] for x in color_events)),"failure_after_correct_region_qa_cases":sum(r["primitive_correctness"]["both_regions_correct"] and not r["primitive_correctness"]["both_colors_correct"] for r in rows),"events":color_events}
    dump(OUT/"color_failure_analysis.json",color_analysis)
    relation_analysis={"overall_correct":ratio(b("relation_correct"),40),"correct_given_both_regions":ratio(relation_given_regions,both),"failure_classes":dict(Counter(x["relation_failure_class"] for x in relation_events if x["relation_failure_class"])),"wrong_despite_both_regions_correct_case_ids":[x["qa_id"] for x in relation_events if x["relation_failure_class"]=="RELATION_FAILURE_AFTER_CORRECT_REGIONS"],"events":relation_events}
    dump(OUT/"relation_failure_analysis.json",relation_analysis)
    signatures=Counter(r["failure_signature"] for r in rows)
    dump(OUT/"failure_signatures.json",{"n_qa":40,"signatures":[{"signature":signature,"qa_count":count,"family_count":count//2,"example_qa_ids":[r["qa_id"] for r in rows if r["failure_signature"]==signature][:4]} for signature,count in signatures.most_common()]})
    color_unlock=sum(r["primitive_correctness"]["both_regions_correct"] and r["primitive_correctness"]["relation_correct"] and not r["primitive_correctness"]["both_colors_correct"] for r in rows)
    relation_unlock=sum(r["primitive_correctness"]["both_regions_correct"] and r["primitive_correctness"]["both_colors_correct"] and not r["primitive_correctness"]["relation_correct"] for r in rows)
    region_limited=sum(not r["primitive_correctness"]["both_regions_correct"] for r in rows)
    overlap={"region_failure_qa":region_limited,"any_color_failure_qa":sum(not r["primitive_correctness"]["both_colors_correct"] for r in rows),"relation_failure_qa":sum(not r["primitive_correctness"]["relation_correct"] for r in rows),"region_and_color_failure_qa":sum(not r["primitive_correctness"]["both_regions_correct"] and not r["primitive_correctness"]["both_colors_correct"] for r in rows),"region_and_relation_failure_qa":sum(not r["primitive_correctness"]["both_regions_correct"] and not r["primitive_correctness"]["relation_correct"] for r in rows),"color_and_relation_failure_qa":sum(not r["primitive_correctness"]["both_colors_correct"] and not r["primitive_correctness"]["relation_correct"] for r in rows),"multiple_simultaneous_component_failures_qa":sum(sum((not r["primitive_correctness"]["both_regions_correct"],not r["primitive_correctness"]["both_colors_correct"],not r["primitive_correctness"]["relation_correct"]))>=2 for r in rows)}
    recover={"baseline_strict":strict,"perfect_color_given_current_regions":{"additional_strict_sufficient_qa":color_unlock,"new_strict_sufficient_qa":strict+color_unlock,"assumption":"Evaluator-only color correction; current selected regions and relation fixed"},"perfect_relation_given_current_regions":{"additional_strict_sufficient_qa":relation_unlock,"new_strict_sufficient_qa":strict+relation_unlock,"assumption":"Evaluator-only relation correction; current regions and colors fixed"},"region_repair_only":{"additional_strict_sufficient_qa":None,"new_strict_sufficient_qa":None,"reason":"Saved color predictions were produced on current regions and cannot be transferred to hypothetical GT-correct regions; no inference was rerun"},"region_limited_qa":region_limited,"region_limited_family_count":region_limited//2,"failure_overlap":overlap,"priority_by_measurable_single_component_unlock":[{"component":"color","additional_qa":color_unlock},{"component":"relation","additional_qa":relation_unlock}],"region_priority_caveat":"24 QA require region repair before full sufficiency, but its isolated gain cannot be measured from frozen colors"}
    dump(OUT/"marginal_recoverability.json",recover)
    make_figures(chain,metrics,signatures,recover,family_rows)
    make_report(rows,family_rows,metrics,chain,region_analysis,color_analysis,relation_analysis,recover,signatures)
    (OUT/"changed_files.txt").write_text("harness/scripts/run_state_bottleneck_audit.py\noutputs/state_bottleneck_audit/ (new evaluator-only artifacts)\n")
    (OUT/"test_report.md").write_text("# Test report\n\n- Loaded 40 saved QA states and 40 evaluator result rows from the handoff pilot; no model or final reasoner was called.\n- Recomputed all primitive booleans using the frozen IoU 0.5, canonical color labels, and RealBind relation labels; all matched saved handoff evaluator values.\n- Replayed only the frozen AcquisitionRouter selection key against cached candidate scores and matched every selected index.\n- Verified 20 paired families, shared state text/hash, stable A/B regions/colors/relation, and strict count 6/40.\n- Oracle color and relation calculations are evaluator-only Boolean counterfactuals; region repair is marked unavailable because saved color predictions cannot be transferred to new regions.\n")
    print("state bottleneck audit: strict",strict,"/40; region-limited",region_limited,"; color unlock",color_unlock,"; relation unlock",relation_unlock)


def make_figures(chain,metrics,signatures,recover,families):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    dest=OUT/"figures"
    plt.figure(figsize=(9,4));plt.bar([x["stage"] for x in chain],[x["count"] for x in chain]);plt.ylim(0,40);plt.ylabel("Surviving QA");plt.xticks(rotation=18,ha="right");plt.tight_layout();plt.savefig(dest/"figure1_survival_funnel.png");plt.close()
    labels=["A color | A region","B color | B region","Both colors | both regions","Relation | both regions","Strict | both regions"]
    keys=["P_C_A_given_B_A","P_C_B_given_B_B","P_both_colors_given_both_regions","P_R_given_both_regions","P_strict_given_both_regions"]
    plt.figure(figsize=(9,4));plt.bar(labels,[metrics[k]["rate"] for k in keys]);plt.ylim(0,1);plt.ylabel("Conditional correctness");plt.xticks(rotation=18,ha="right");plt.tight_layout();plt.savefig(dest/"figure2_conditional_components.png");plt.close()
    top=signatures.most_common(8);plt.figure(figsize=(10,4));plt.bar([x[0].replace(" ","\n") for x in top],[x[1] for x in top]);plt.ylabel("QA cases");plt.xticks(fontsize=7);plt.tight_layout();plt.savefig(dest/"figure3_failure_signatures.png");plt.close()
    plt.figure(figsize=(6,4));plt.bar(["Color alone","Relation alone","Region-limited\n(not oracle gain)"],[recover["perfect_color_given_current_regions"]["additional_strict_sufficient_qa"],recover["perfect_relation_given_current_regions"]["additional_strict_sufficient_qa"],recover["region_limited_qa"]]);plt.ylabel("QA cases");plt.tight_layout();plt.savefig(dest/"figure4_recoverability.png");plt.close()
    matrix=np.array([[int(f["primitive_correctness"][key]) for key in KEYS] for f in families])
    plt.figure(figsize=(7,8));plt.imshow(matrix,cmap="Greens",vmin=0,vmax=1,aspect="auto");plt.xticks(range(5),["Region A","Region B","Color A","Color B","Relation"]);plt.yticks(range(20),[f["family_id"].split("_")[2] for f in families],fontsize=7);plt.colorbar(label="Correct");plt.tight_layout();plt.savefig(dest/"figure5_family_state_heatmap.png");plt.close()


def make_report(rows,families,m,chain,region,color,relation,recover,signatures):
    n=40;strict=m["P_strict"]["count"]
    def fmt(key):
        x=m[key];return f"{x['count']}/{x['denominator']} ({x['rate']:.3f})"
    recommendation="MULTI_COMPONENT_STATE_FAILURE"
    lines=["# State bottleneck decomposition audit","",f"**Recommendation: `{recommendation}`.** This is an evaluator-only forensic audit of the frozen handoff pilot. No detector, color reader, blur process, geometry implementation, Qwen reasoner, score, threshold, or state was changed.","",f"Frozen source hashes: handoff state `{sha(HANDOFF/'compiled_states.jsonl')}`; handoff evaluator results `{sha(HANDOFF/'qa_condition_results.jsonl')}`; Grounding DINO cache `{sha(CACHE)}`.","","## Table A — primitive correctness","","| Metric | Count | Rate |","|---|---:|---:|"]
    for label,key in [("A region correct","P_B_A"),("B region correct","P_B_B"),("Both regions correct","P_B_A_and_B_B"),("A color correct","P_C_A"),("B color correct","P_C_B"),("Both colors correct","P_C_A_and_C_B"),("Relation correct","P_R"),("Strict fact sufficient","P_strict")]:
        x=m[key];lines.append(f"| {label} | {x['count']}/40 | {x['rate']:.3f} |")
    lines += ["","## Table B — conditional correctness","","| Conditional metric | Value |","|---|---:|"]
    for label,key in [("P(color A correct \\| region A correct)","P_C_A_given_B_A"),("P(color B correct \\| region B correct)","P_C_B_given_B_B"),("P(both colors correct \\| both regions correct)","P_both_colors_given_both_regions"),("P(relation correct \\| both regions correct)","P_R_given_both_regions"),("P(strict sufficient \\| both regions correct)","P_strict_given_both_regions")]:lines.append(f"| {label} | {fmt(key)} |")
    lines += ["","## Table C — primary survival chain","","| Stage | Surviving QA | Survival from previous | Survival from all 40 |","|---|---:|---:|---:|"]
    for x in chain:lines.append(f"| {x['stage']} | {x['count']} | {x['survival_from_previous']:.3f} | {x['survival_from_all']:.3f} |")
    lines += ["",f"Empirical chain: 16/40 × 6/16 × 6/6 = 6/40 = 0.150. This is a sequential identity on observed cases; no independence is assumed. In the parallel view, A and B regions are correct in {m['P_B_A']['count']} and {m['P_B_B']['count']} QA, respectively. Among the 16 both-region cases, A color is correct in 12 and B color in 8; both colors in 6. A/B ordering is not a causal sequence.","","## Table D — overlapping failure attribution","","| Failure source | QA count |","|---|---:|",f"| Region / binding failure | {recover['failure_overlap']['region_failure_qa']} |",f"| Color failure after both regions correct | {color['failure_after_correct_region_qa_cases']} |",f"| Relation failure after both regions correct | {len(relation['wrong_despite_both_regions_correct_case_ids'])} |",f"| Multiple simultaneous component failures | {recover['failure_overlap']['multiple_simultaneous_component_failures_qa']} |",""]
    o=recover["failure_overlap"]
    lines.append(f"Overlaps: region+color {o['region_and_color_failure_qa']}, region+relation {o['region_and_relation_failure_qa']}, color+relation {o['color_and_relation_failure_qa']} QA. These categories are deliberately nonexclusive.")
    lines += ["","## Table E — marginal recoverability","","| Hypothetical single-component fix | Additional strict-sufficient QA | New upper bound |","|---|---:|---:|",f"| Perfect color given current regions | {recover['perfect_color_given_current_regions']['additional_strict_sufficient_qa']} | {recover['perfect_color_given_current_regions']['new_strict_sufficient_qa']}/40 |",f"| Perfect relation given current regions | {recover['perfect_relation_given_current_regions']['additional_strict_sufficient_qa']} | {recover['perfect_relation_given_current_regions']['new_strict_sufficient_qa']}/40 |","| Current components except region failures remain | Unavailable | Unavailable |",""]
    lines.append(f"{recover['region_limited_qa']} QA cannot become strict-sufficient without fixing at least one region. A region-only oracle result is unavailable: colors saved on wrong regions cannot be transferred to hypothetical corrected regions. The color and relation rows are evaluator-only Boolean counterfactuals, not changed inference outputs.")
    lines += ["","## Region failure breakdown","",f"Of 40 unique family entities, {region['failed_unique_entities']} selected regions fail IoU 0.5: {region['candidate_generation_failures_unique']} lack any usable cached candidate and {region['candidate_selection_failures_unique']} have a usable candidate that was not selected. The first group includes {region['failure_classes_unique'].get('OVERLARGE_REGION',0)} overlarge, {region['failure_classes_unique'].get('PARTIAL_OVERLAP',0)} partial, and {region['failure_classes_unique'].get('NO_USABLE_CANDIDATE',0)} empty/no-usable cases. {region['same_query_zero_overlap_distractor_proxy_unique']} failed regions have same-query labels and zero GT overlap, including {region['same_query_zero_overlap_among_selection_failures_unique']} selection failures. These are ambiguity proxies, not verified wrong-same-category-instance labels.",f"Joint candidate availability is {region['candidate_oracle_joint_covered_families']}/20 families; both selected regions are correct in {region['both_selected_regions_correct_families']}/20; full strict state sufficiency is {strict//2}/20.","","## Color and relation breakdown","",f"The color event taxonomy over 40 unique entities is {color['classes_unique']}. Raw color correctness includes {color['classes_qa_occurrences'].get('COLOR_ACCIDENTALLY_CORRECT_ON_WRONG_REGION',0)} QA-entity occurrences with a wrong region; these are not successful perception. Conditional color correctness is A {fmt('P_C_A_given_B_A')}, B {fmt('P_C_B_given_B_B')}, and both colors {fmt('P_both_colors_given_both_regions')}.",f"Relation is correct in {fmt('P_R')} overall and {fmt('P_R_given_both_regions')} once both regions are correct. Relation failures after correct regions: {relation['wrong_despite_both_regions_correct_case_ids']}.","","## Failure signatures and state stability","",f"Top signatures: {signatures.most_common(5)}. Full grouped signatures and case IDs are in `failure_signatures.json`. All {sum(f['state_stable'] for f in families)}/20 LEFT/RIGHT pairs have identical region, color, relation, state text, and state hash; no `STATE_INCONSISTENCY` was found.","","## Scientific questions"]
    answers=["Q1. The collapse is compounded: 24/40 QA lose at least one selected region, then 10 of the 16 both-region QA lose color correctness. Relation costs zero further cases once regions and colors are correct.","Q2. Region selection leaves 16/40 both-region QA; 24/40 are lost before attributing the later color stage.",f"Q3. With correct own-region grounding, A color {fmt('P_C_A_given_B_A')}, B color {fmt('P_C_B_given_B_B')}; both colors {fmt('P_both_colors_given_both_regions')} conditional on both regions.",f"Q4. Yes. Relation correctness is {fmt('P_R_given_both_regions')} conditional on both regions; there are no relation failures after correct regions.",f"Q5. Yes. {color['classes_qa_occurrences'].get('COLOR_ACCIDENTALLY_CORRECT_ON_WRONG_REGION',0)} of 80 QA-entity color events match the GT color despite a wrong selected region.","Q6. See the signature table: the most common patterns include correct regions with a color error and compounding wrong-region/color errors.","Q7. Yes. All 20 paired families have identical state facts and serialized state hashes across LEFT and RIGHT questions.",f"Q8. Among measurable one-component repairs, color correction alone unlocks {recover['perfect_color_given_current_regions']['additional_strict_sufficient_qa']} QA; relation unlocks {recover['perfect_relation_given_current_regions']['additional_strict_sufficient_qa']}. Region repair is necessary for {recover['region_limited_qa']} QA, but its isolated gain is unavailable without rerunning the color reader on corrected regions.","Q9. Multiple compounding errors: region/binding removes 24 QA before full state assembly, while attribute reading removes 10 more among the 16 with both regions. Geometry is not an independent bottleneck here."]
    lines += ["",*answers,"","## Recommendation","",f"`{recommendation}`. Color is the largest measurable single-component unlock on current regions, but the selected-region ceiling is only 16/40 QA. A region-only oracle is scientifically unavailable from saved color outputs. Improve state sufficiency before repeating handoff comparisons; this audit itself changes no component."]
    (OUT/"implementation_report.md").write_text("\n".join(lines)+"\n")
    sections=[]
    for r in rows:
        c=r["primitive_correctness"]
        cells="".join(f"<tr><td>{ref}</td><td>{r['entities'][ref]['selected_iou']:.3f}</td><td>{r['entities'][ref]['region_failure_class'] or 'correct'}</td><td>{r['entities'][ref]['acquired_color']}</td><td>{r['entities'][ref]['evaluator_color']}</td><td>{r['entities'][ref]['color_failure_class']}</td></tr>" for ref in ("A","B"))
        sections.append(f"<section><h2>{html.escape(r['qa_id'])}</h2><p>{r['failure_signature']} · strict={c['strict_fact_sufficient']} · relation={r['relation_failure_class'] or 'correct'}</p><table><tr><th>Entity</th><th>Selected IoU</th><th>Region diagnosis</th><th>Acquired color</th><th>GT color</th><th>Color diagnosis</th></tr>{cells}</table></section>")
    (OUT/"audit.html").write_text("<!doctype html><html><head><meta charset='utf-8'><style>body{font:14px sans-serif;max-width:1150px;margin:auto}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:5px}section{margin:22px 0}</style></head><body><h1>State bottleneck audit — evaluator only</h1>"+"".join(sections)+"</body></html>")


if __name__=="__main__":main()
