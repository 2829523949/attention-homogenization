#!/usr/bin/env python3
"""Read-only agreement analysis of frozen semantic and contextual-blur logits."""
from __future__ import annotations

import hashlib
import html
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "outputs/contextual_blur_binding_pilot"
ACQUISITION = ROOT / "outputs/grounding_dino_acquisition_pilot/entity_backend_comparison.jsonl"
OUT = ROOT / "outputs/semantic_blur_agreement_pilot"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))


def top(candidates, field):
    if not candidates:
        return None
    # Preserve frozen candidate ordering as tie break; no score threshold.
    return max(range(len(candidates)), key=lambda i: (candidates[i][field], -i))


def box_iou(a,b):
    overlap=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda box:max(0.,box[2]-box[0])*max(0.,box[3]-box[1])
    return overlap/max(1e-12,area(a)+area(b)-overlap)


def wilson(success, total):
    if not total:
        return None
    z = 1.959963984540054
    p = success / total
    den = 1 + z*z/total
    middle = (p + z*z/(2*total))/den
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return [middle-half, middle+half]


def fraction(success, total):
    return {"numerator": success, "denominator": total, "fraction": success/total if total else None, "wilson_95_ci": wilson(success,total)}


def summarize(rows, *, gt_best=False):
    scored = [r for r in rows if r["semantic_top_index"] is not None]
    agree = [r for r in scored if r["agreement"]]
    disagree = [r for r in scored if r["agreement"] is False]
    field = "semantic_gt_best" if gt_best else "semantic_gt_compatible"
    blur_field = "blur_gt_best" if gt_best else "blur_gt_compatible"
    sem_good = sum(r["EVALUATOR_ONLY"][field] for r in scored)
    blur_good = sum(r["EVALUATOR_ONLY"][blur_field] for r in scored)
    agree_good = sum(r["EVALUATOR_ONLY"][field] for r in agree)
    disagree_good = sum(r["EVALUATOR_ONLY"][field] for r in disagree)
    return {
        "n_referent_sets":len(rows),"n_scoreable":len(scored),"n_agree":len(agree),"n_disagree":len(disagree),"n_empty":len(rows)-len(scored),
        "semantic_top1_precision":fraction(sem_good,len(scored)),
        "blur_top1_precision":fraction(blur_good,len(scored)),
        "agreement_acceptance_rate":fraction(len(agree),len(rows)),
        "accepted_binding_precision":fraction(agree_good,len(agree)),
        "accepted_correct_binding_yield":fraction(agree_good,len(rows)),
        "disagreement_semantic_error_rate":fraction(len(disagree)-disagree_good,len(disagree)),
        "agreement_semantic_error_rate":fraction(len(agree)-agree_good,len(agree)),
        "disagreement_error_enrichment_risk_difference":((len(disagree)-disagree_good)/len(disagree)-(len(agree)-agree_good)/len(agree)) if disagree and agree else None,
        "disagreement_semantic_correct":disagree_good,
        "disagreement_blur_correct":sum(r["EVALUATOR_ONLY"][blur_field] for r in disagree),
    }


def build_referents(source, acquisition):
    assert len(source)==40
    gd=[r for r in acquisition if r["inference"]["backend"]=="grounding_dino"]
    assert len(gd)==40
    assert [r["entity_unit_id"] for r in source]==[r["inference"]["entity_unit_id"] for r in gd]
    rows=[]
    for s,g in zip(source,gd):
        candidates=s["candidates"]
        frozen=g["inference"]["candidates"]
        assert len(candidates)==len(frozen)==s["candidate_count"]
        for c,f in zip(candidates,frozen):
            assert c["candidate_id"]==f["candidate_id"] and c["bbox_xyxy_pixels"]==f["bbox_xyxy_pixels"]
            assert c["crop_siglip_score"]==f["siglip_diagnostic"]["z_raw"]
            assert c["native_grounding_score"]==f["native_grounding_score"]
        sem=top(candidates,"crop_siglip_score")
        blur=top(candidates,"N")
        native=top(candidates,"native_grounding_score")
        if s["EVALUATOR_ONLY"]["candidate_covered"]:
            ordered=sorted(range(len(candidates)),key=lambda i:(-candidates[i]["crop_siglip_score"],i))
            assert ordered.index(s["EVALUATOR_ONLY"]["gt_best_candidate_index"])+1==s["EVALUATOR_ONLY"]["ranks"]["crop_siglip_score"]
        agreement=sem==blur if sem is not None else None
        covered=s["EVALUATOR_ONLY"]["candidate_covered"]
        gt_best=s["EVALUATOR_ONLY"]["gt_best_candidate_index"]
        def compatible(index):return bool(index is not None and candidates[index]["EVALUATOR_ONLY"]["gt_iou"]>=.5)
        sem_good=compatible(sem);blur_good=compatible(blur)
        disagreement_type=("BOTH_GT_COMPATIBLE_DIFFERENT_PROPOSALS" if sem_good and blur_good else "SEMANTIC_ONLY_GT_COMPATIBLE" if sem_good else "BLUR_ONLY_GT_COMPATIBLE" if blur_good else "NEITHER_GT_COMPATIBLE") if agreement is False else None
        row={
            "entity_unit_id":s["entity_unit_id"],"family_id":s["family_id"],"logical_ref":s["logical_ref"],
            "candidate_count":len(candidates),"candidate_set_type":"MULTI_CANDIDATE" if len(candidates)>1 else "SINGLETON_AUTOMATIC_AGREEMENT" if len(candidates)==1 else "EMPTY_NO_PROPOSAL",
            "semantic_top_index":sem,"blur_top_index":blur,"grounding_dino_top_index":native,
            "agreement":agreement,"diagnostic_action":"ACCEPT_PROPOSAL" if agreement else "ABSTAIN_DISAGREEMENT" if agreement is False else "ABSTAIN_EMPTY",
            "semantic_blur_top_box_iou":box_iou(candidates[sem]["bbox_xyxy_pixels"],candidates[blur]["bbox_xyxy_pixels"]) if sem is not None else None,
            "selection_policy":"argmax raw logit; frozen candidate order breaks exact ties",
            "semantic_top_score":candidates[sem]["crop_siglip_score"] if sem is not None else None,
            "blur_top_N":candidates[blur]["N"] if blur is not None else None,
            "candidate_scores":[{"candidate_index":c["candidate_index"],"candidate_id":c["candidate_id"],"bbox_xyxy_pixels":c["bbox_xyxy_pixels"],"native_grounding_score":c["native_grounding_score"],"crop_siglip_score":c["crop_siglip_score"],"blur_N":c["N"],"EVALUATOR_ONLY":{"gt_iou":c["EVALUATOR_ONLY"]["gt_iou"]}} for c in candidates],
            "status":"DIAGNOSTIC_UNCALIBRATED","qwen_called":False,
            "EVALUATOR_ONLY":{"candidate_covered":covered,"gt_best_candidate_index":gt_best,"semantic_gt_compatible":sem_good,"blur_gt_compatible":blur_good,"native_gt_compatible":compatible(native),"semantic_gt_best":bool(covered and sem==gt_best),"blur_gt_best":bool(covered and blur==gt_best),"semantic_top_gt_iou":candidates[sem]["EVALUATOR_ONLY"]["gt_iou"] if sem is not None else None,"blur_top_gt_iou":candidates[blur]["EVALUATOR_ONLY"]["gt_iou"] if blur is not None else None,"disagreement_type":disagreement_type},
        }
        rows.append(row)
    return rows


def build_families(rows):
    grouped=defaultdict(dict)
    for r in rows:grouped[r["family_id"]][r["logical_ref"]]=r
    assert len(grouped)==20 and all(set(v)=={"A","B"} for v in grouped.values())
    result=[]
    for family,refs in grouped.items():
        a,b=refs["A"],refs["B"]
        accept=bool(a["agreement"] and b["agreement"])
        sem_correct=all(r["EVALUATOR_ONLY"]["semantic_gt_compatible"] for r in (a,b))
        blur_correct=all(r["EVALUATOR_ONLY"]["blur_gt_compatible"] for r in (a,b))
        result.append({"family_id":family,"referent_agreement":{"A":a["agreement"],"B":b["agreement"]},"referent_actions":{"A":a["diagnostic_action"],"B":b["diagnostic_action"]},"both_agree":accept,"diagnostic_action":"ACCEPT_BOTH_PROPOSALS" if accept else "ABSTAIN_FAMILY","status":"DIAGNOSTIC_UNCALIBRATED","qwen_called":False,"EVALUATOR_ONLY":{"joint_candidate_covered":all(r["EVALUATOR_ONLY"]["candidate_covered"] for r in (a,b)),"semantic_joint_correct":sem_correct,"blur_joint_correct":blur_correct,"accepted_joint_correct":bool(accept and sem_correct)}})
    return result


def family_summary(families):
    accepted=[f for f in families if f["both_agree"]]
    rejected=[f for f in families if not f["both_agree"]]
    covered=[f for f in families if f["EVALUATOR_ONLY"]["joint_candidate_covered"]]
    accepted_covered=[f for f in covered if f["both_agree"]]
    return {"n_families":len(families),"candidate_oracle_covered_families":len(covered),"semantic_joint_accuracy":fraction(sum(f["EVALUATOR_ONLY"]["semantic_joint_correct"] for f in families),len(families)),"blur_joint_accuracy":fraction(sum(f["EVALUATOR_ONLY"]["blur_joint_correct"] for f in families),len(families)),"both_agree_families":len(accepted),"accepted_joint_precision":fraction(sum(f["EVALUATOR_ONLY"]["accepted_joint_correct"] for f in accepted),len(accepted)),"accepted_joint_correct_yield":fraction(sum(f["EVALUATOR_ONLY"]["accepted_joint_correct"] for f in families),len(families)),"abstained_families":len(rejected),"abstained_semantic_joint_correct":sum(f["EVALUATOR_ONLY"]["semantic_joint_correct"] for f in rejected),"covered_family_accepted_precision":fraction(sum(f["EVALUATOR_ONLY"]["accepted_joint_correct"] for f in accepted_covered),len(accepted_covered)),"covered_family_semantic_accuracy":fraction(sum(f["EVALUATOR_ONLY"]["semantic_joint_correct"] for f in covered),len(covered))}


def report(stats,gt_best,fstats,rows,source_hash,acq_hash):
    all_=stats["all_scoreable"]; multi=stats["multi_candidate"]; covered=stats["candidate_covered"];mc=stats["multi_candidate_covered"]
    def fmt(x):
        return f"{x['numerator']}/{x['denominator']} ({x['fraction']:.3f})" if x["fraction"] is not None else "undefined"
    lines=["# Semantic–blur agreement verification pilot","","**Result: agreement is not a high-precision binding verification signal on this frozen set.** Outputs are `DIAGNOSTIC_UNCALIBRATED`; no certificate or persistent binding was changed.","",f"Frozen contextual-blur result SHA-256: `{source_hash}`. Frozen Grounding DINO cache SHA-256: `{acq_hash}`. No detector, SigLIP, blur, or Qwen call was made.","","## Protocol","","For each frozen candidate set, semantic proposal is argmax crop SigLIP raw logit and blur proposal is argmax contextual necessity N. Exact ties use frozen candidate order. Agreement means equal candidate indices. Correctness means selected candidate IoU ≥ 0.5 against evaluator-only GT. The 0.5 IoU criterion is inherited from the frozen pilots; no score or agreement threshold was tuned. GT-best ranking is reported separately in `agreement_metrics.json`. Empty sets abstain. All singleton sets agree by construction, so multi-candidate results are essential.","","## Referent-level results","","| Stratum | Semantic top1 correct | Agreement accepted | Correct among accepted | Error among disagreement |","|---|---:|---:|---:|---:|"]
    for label,key in [("All scoreable (39)","all_scoreable"),("Multi-candidate (21)","multi_candidate"),("Candidate-covered (29; evaluator-only)","candidate_covered"),("Multi-candidate and covered (18; evaluator-only)","multi_candidate_covered")]:
        s=stats[key];lines.append(f"| {label} | {fmt(s['semantic_top1_precision'])} | {s['n_agree']}/{s['n_scoreable']} | {fmt(s['accepted_binding_precision'])} | {fmt(s['disagreement_semantic_error_rate'])} |")
    disagreement=[r for r in rows if r["agreement"] is False]
    types={name:sum(r["EVALUATOR_ONLY"]["disagreement_type"]==name for r in disagreement) for name in ("SEMANTIC_ONLY_GT_COMPATIBLE","BLUR_ONLY_GT_COMPATIBLE","BOTH_GT_COMPATIBLE_DIFFERENT_PROPOSALS","NEITHER_GT_COMPATIBLE")}
    lines += ["",f"All scoreable agreement precision {fmt(all_['accepted_binding_precision'])} versus semantic baseline {fmt(all_['semantic_top1_precision'])}; accepted yield {fmt(all_['accepted_correct_binding_yield'])} using all 40 referent sets as denominator. Its Wilson 95% interval is {all_['accepted_binding_precision']['wilson_95_ci']}.",f"Among multi-candidate sets, agreement precision {fmt(multi['accepted_binding_precision'])} versus semantic baseline {fmt(multi['semantic_top1_precision'])}; the difference is too small to support high-precision verification.",f"GT-best ranking on the 29 covered sets reproduces crop SigLIP {fmt(gt_best['candidate_covered']['semantic_top1_precision'])}; agreement accepts 23 sets with GT-best precision {fmt(gt_best['candidate_covered']['accepted_binding_precision'])}. These GT-best numbers are ranking diagnostics; binding correctness uses IoU ≥ 0.5.",f"All {stats['singleton']['n_scoreable']} singleton sets agree automatically. {stats['uncovered']['n_agree']} of {stats['uncovered']['n_scoreable']} scoreable acquisition-uncovered sets also agree; none can be a correct binding at IoU 0.5. Agreement therefore cannot detect acquisition failure.","","Disagreement is weakly enriched for semantic errors overall: "+fmt(all_["disagreement_semantic_error_rate"])+" error among disagreements versus "+fmt(all_["agreement_semantic_error_rate"])+" among agreements. With only six disagreements, the intervals are wide. Of those six, semantic alone is GT-compatible in "+str(types["SEMANTIC_ONLY_GT_COMPATIBLE"])+", blur alone in "+str(types["BLUR_ONLY_GT_COMPATIBLE"])+", both in "+str(types["BOTH_GT_COMPATIBLE_DIFFERENT_PROPOSALS"])+", and neither in "+str(types["NEITHER_GT_COMPATIBLE"])+". Both-compatible disagreements reflect different proposals for a usable region, so disagreement is not always a binding failure.","","## Family-level accept/abstain simulation","",f"Both A and B agree in {fstats['both_agree_families']}/20 families; {fmt(fstats['accepted_joint_precision'])} of accepted families are jointly correct. Semantic selection without abstention is {fmt(fstats['semantic_joint_accuracy'])}; agreement abstains on {fstats['abstained_families']} families, including {fstats['abstained_semantic_joint_correct']} that semantic ranking got jointly correct. Among evaluator-only candidate-covered families, accepted precision is {fmt(fstats['covered_family_accepted_precision'])} versus semantic joint accuracy {fmt(fstats['covered_family_semantic_accuracy'])}. Candidate oracle coverage remains {fstats['candidate_oracle_covered_families']}/20.","","## Interpretation","","Agreement between crop semantics and contextual blur is an observable event, but it is common when the candidate set has one item or misses the target. The observed conditional precision gain is small overall and does not justify using agreement as a binding certificate. Disagreement abstention removes some errors and some correct bindings. The channels had nonredundant scores in the prior pilot; this pilot shows that nonredundancy alone does not guarantee a reliable agreement verifier.","","This is evaluator-side analysis of frozen scores. It is not a production accept/abstain rule. Persistent logical identity, atomic fact acquisition, proof-state construction, handoff, and final reasoning were untouched. `qwen_called=false`."]
    (OUT/"implementation_report.md").write_text("\n".join(lines)+"\n")
    (OUT/"audit.html").write_text("<!doctype html><html><head><meta charset='utf-8'><style>body{font:14px sans-serif;max-width:1100px;margin:auto}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:5px}section{margin:24px 0}</style></head><body><h1>Semantic–blur agreement audit</h1>"+"".join(f"<section><h2>{html.escape(r['entity_unit_id'])}</h2><p>{r['candidate_set_type']} · {r['diagnostic_action']} · agreement={r['agreement']} · GT-covered={r['EVALUATOR_ONLY']['candidate_covered']}</p><table><tr><th>Candidate</th><th>Crop SigLIP</th><th>Blur N</th><th>GT IoU (evaluator only)</th></tr>"+"".join(f"<tr><td>{c['candidate_index']}</td><td>{c['crop_siglip_score']:.4f}</td><td>{c['blur_N']:.4f}</td><td>{c['EVALUATOR_ONLY']['gt_iou']:.4f}</td></tr>" for c in r['candidate_scores'])+"</table></section>" for r in rows)+"</body></html>")


def main():
    if OUT.exists():raise FileExistsError(f"Refusing to overwrite prior output: {OUT}")
    source_path=SOURCE_DIR/"candidate_intervention_results.jsonl"
    source=load_jsonl(source_path); acquisition=load_jsonl(ACQUISITION)
    config=json.loads((SOURCE_DIR/"blur_config.json").read_text())
    assert sha(ACQUISITION)==config["source_sha256"]
    assert all(not r["qwen_called"] and r["status"]=="DIAGNOSTIC_UNCALIBRATED" for r in source)
    rows=build_referents(source,acquisition)
    families=build_families(rows)
    assert len(rows)==40 and sum(r["EVALUATOR_ONLY"]["candidate_covered"] for r in rows)==29
    assert sum(f["EVALUATOR_ONLY"]["joint_candidate_covered"] for f in families)==12
    assert sum(f["EVALUATOR_ONLY"]["semantic_joint_correct"] for f in families)==8
    strata={"all_scoreable":rows,"singleton":[r for r in rows if r["candidate_count"]==1],"multi_candidate":[r for r in rows if r["candidate_count"]>1],"candidate_covered":[r for r in rows if r["EVALUATOR_ONLY"]["candidate_covered"]],"multi_candidate_covered":[r for r in rows if r["candidate_count"]>1 and r["EVALUATOR_ONLY"]["candidate_covered"]],"uncovered":[r for r in rows if not r["EVALUATOR_ONLY"]["candidate_covered"]]}
    stats={k:summarize(v) for k,v in strata.items()}
    gt_best={k:summarize(v,gt_best=True) for k,v in strata.items()}
    fstats=family_summary(families)
    OUT.mkdir(parents=True)
    write_jsonl(OUT/"referent_agreement_results.jsonl",rows)
    write_jsonl(OUT/"family_agreement_results.jsonl",families)
    write_json(OUT/"agreement_metrics.json",{"correctness":"selected candidate GT IoU >= 0.5; evaluator only","strata":stats,"gt_best_ranking_diagnostic":gt_best,"family":fstats,"qwen_called":False,"status":"DIAGNOSTIC_UNCALIBRATED"})
    write_json(OUT/"disagreement_taxonomy.json",{"cases":[{"entity_unit_id":r["entity_unit_id"],"semantic_top_index":r["semantic_top_index"],"blur_top_index":r["blur_top_index"],"top_box_iou":r["semantic_blur_top_box_iou"],"EVALUATOR_ONLY":r["EVALUATOR_ONLY"]} for r in rows if r["agreement"] is False],"counts":{name:sum(r["EVALUATOR_ONLY"]["disagreement_type"]==name for r in rows if r["agreement"] is False) for name in ("SEMANTIC_ONLY_GT_COMPATIBLE","BLUR_ONLY_GT_COMPATIBLE","BOTH_GT_COMPATIBLE_DIFFERENT_PROPOSALS","NEITHER_GT_COMPATIBLE")}})
    write_json(OUT/"input_manifest.json",{"contextual_candidate_results":{"path":str(source_path),"sha256":sha(source_path)},"grounding_dino_cache":{"path":str(ACQUISITION),"sha256":sha(ACQUISITION)},"score_modification":False,"model_called":False,"blur_generated":False,"qwen_called":False})
    report(stats,gt_best,fstats,rows,sha(source_path),sha(ACQUISITION))
    (OUT/"changed_files.txt").write_text("harness/scripts/run_semantic_blur_agreement_pilot.py\noutputs/semantic_blur_agreement_pilot/ (new analysis artifacts)\n")
    (OUT/"test_report.md").write_text("# Test report\n\n- Reused the frozen 40 referent sets, 74 candidates, and 20 families.\n- Verified each candidate ID, box, Grounding DINO score, and crop SigLIP score against the original acquisition cache.\n- Verified frozen input hashes, 29 candidate-covered sets, 12 candidate-covered families, and semantic joint correctness 8/20.\n- Used saved blur N values without model calls or view regeneration.\n- Preserved evaluator-only GT labels and recorded `qwen_called=false`.\n- No score thresholds or persistent bindings were changed.\n")
    print("Agreement:",stats["all_scoreable"]["n_agree"],"/39; precision:",stats["all_scoreable"]["accepted_binding_precision"],"family:",fstats["accepted_joint_precision"])


if __name__=="__main__":main()
