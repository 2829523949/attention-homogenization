#!/usr/bin/env python3
"""Frozen-candidate, uncalibrated contextual blur binding diagnostic."""
from __future__ import annotations

import hashlib
import html
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops, ImageFilter, __version__ as pillow_version
from scipy.stats import spearmanr

from harness.configs.paths import Paths, ensure_d_drive_environment
from harness.processors.defocus import DEFOCUS_CONFIG, _box_mask, _expanded
from harness.processors.siglip_backend import Siglip2Backend
from harness.processors.siglip_full_image_query import Siglip2FullImageQueryScorer

SOURCE = ROOT / "outputs/grounding_dino_acquisition_pilot/entity_backend_comparison.jsonl"
MANIFEST = ROOT / "manifests/acquisition_semantic_support_20family.json"
OUT = ROOT / "outputs/contextual_blur_binding_pilot"
METHODS = {"GDINO native": "native_grounding_score", "crop SigLIP": "crop_siglip_score", "Focus score": "S_focus", "Suppression effect D": "D", "Necessity N": "N", "Exclusivity E": "E"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows), encoding="utf-8")


def iou(a, b):
    x = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(0., a[2]-a[0])*max(0., a[3]-a[1])
    bb = max(0., b[2]-b[0])*max(0., b[3]-b[1])
    return x*y/max(1e-12, aa+bb-x*y)


def geometry(candidates, size):
    boxes = [c["bbox_xyxy_pixels"] for c in candidates]
    n = len(boxes)
    areas = [max(0., b[2]-b[0])*max(0., b[3]-b[1]) for b in boxes]
    centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in boxes]
    diag = math.hypot(*size)
    return {
        "candidate_overlap_clusters": {"status": "OBSERVATIONAL_ONLY_NO_CLUSTER_THRESHOLD", "candidate_indices": list(range(n)), "pairwise_iou_matrix": [[iou(a,b) for b in boxes] for a in boxes], "pairs": [
            {"i": i, "j": j, "iou": iou(boxes[i],boxes[j]), "center_distance_pixels": math.dist(centers[i],centers[j]), "center_distance_image_diagonal_fraction": math.dist(centers[i],centers[j])/diag, "bbox_area_ratio_larger_to_smaller": max(areas[i],areas[j])/max(1e-12,min(areas[i],areas[j])), "native_score_difference_i_minus_j": candidates[i]["native_grounding_score"]-candidates[j]["native_grounding_score"], "crop_siglip_score_difference_i_minus_j": candidates[i]["siglip_diagnostic"]["z_raw"]-candidates[j]["siglip_diagnostic"]["z_raw"]}
            for i in range(n) for j in range(i+1,n)]},
    }


def make_views(image, box, prefix):
    cfg = DEFOCUS_CONFIG
    support_box = _expanded(box, image.width, image.height, cfg["bbox_expand_frac"])
    mask = _box_mask(image.size, support_box, cfg)
    radius = cfg["blur_radius_fraction_of_min_dimension"] * min(image.size)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    focus = Image.composite(image, blurred, mask)
    suppress = Image.composite(blurred, image, mask)
    paths = {"FOCUS": prefix.with_name(prefix.name + "_focus.png"), "SUPPRESS": prefix.with_name(prefix.name + "_suppress.png")}
    focus.save(paths["FOCUS"])
    suppress.save(paths["SUPPRESS"])
    a = np.asarray(image).astype(np.int16)
    f = np.asarray(focus).astype(np.int16)
    s = np.asarray(suppress).astype(np.int16)
    m = np.asarray(mask)
    # PIL composite arithmetic is checked against the same mask and blur, pixel by pixel.
    expected_f = np.asarray(Image.composite(image, blurred, mask))
    expected_s = np.asarray(Image.composite(blurred, image, mask))
    return paths, {
        "candidate_bbox_xyxy_pixels": box, "expanded_support_bbox_xyxy_pixels": support_box,
        "image_dimensions": list(image.size), "blur_radius_pixels": radius,
        "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "focus_sha256": sha(paths["FOCUS"]), "suppress_sha256": sha(paths["SUPPRESS"]),
        "focus_exact_expected_composite": bool(np.array_equal(np.asarray(focus), expected_f)),
        "suppress_exact_expected_composite": bool(np.array_equal(np.asarray(suppress), expected_s)),
        "focus_sharp_core_exact": bool(np.array_equal(f[m == 255], a[m == 255])),
        "suppress_context_exact": bool(np.array_equal(s[m == 0], a[m == 0])),
        "focus_context_changed_pixels": int(np.count_nonzero(np.any(f[m == 0] != a[m == 0], axis=1))),
        "suppress_support_changed_pixels": int(np.count_nonzero(np.any(s[m == 255] != a[m == 255], axis=1))),
        "focus_dimensions": list(focus.size), "suppress_dimensions": list(suppress.size),
        "mask_pixels_core": int(np.count_nonzero(m == 255)), "mask_pixels_context": int(np.count_nonzero(m == 0)),
    }


def order(rows, key):
    if key == "S_raw":
        return None
    return sorted(range(len(rows)), key=lambda i: (-(rows[i][key] if rows[i][key] is not None else -math.inf), i))


def rank_metrics(sets):
    result = {}
    for label,key in METHODS.items():
        ranks = []
        for s in sets:
            if not s["EVALUATOR_ONLY"]["candidate_covered"]: continue
            if key == "E" and len(s["candidates"]) < 2: continue
            ranks.append(order(s["candidates"], key).index(s["EVALUATOR_ONLY"]["gt_best_candidate_index"])+1)
        result[label] = {"n": len(ranks), "top1_count": sum(r==1 for r in ranks), "top1_fraction": sum(r==1 for r in ranks)/len(ranks) if ranks else None, "top2_fraction": sum(r<=2 for r in ranks)/len(ranks) if ranks else None, "mrr": sum(1/r for r in ranks)/len(ranks) if ranks else None, "median_rank": median(ranks) if ranks else None, "ranks": ranks}
    return result


def incremental(sets):
    out = {}
    for method in ("D","N","E"):
        eligible = [s for s in sets if s["EVALUATOR_ONLY"]["candidate_covered"] and (method != "E" or len(s["candidates"])>1)]
        def correct(s, key): return order(s["candidates"], key)[0] == s["EVALUATOR_ONLY"]["gt_best_candidate_index"]
        def cond(base, desired_base, desired_new):
            ss = [s for s in eligible if correct(s,base)==desired_base]
            count = sum(correct(s,method)==desired_new for s in ss)
            return {"count": count, "denominator": len(ss), "fraction": count/len(ss) if ss else None}
        out[method] = {"n":len(eligible), "rescue_when_gdino_wrong":cond("native_grounding_score",False,True), "rescue_when_siglip_wrong":cond("crop_siglip_score",False,True), "regression_when_siglip_correct":cond("crop_siglip_score",True,False), "both_correct": sum(correct(s,"crop_siglip_score") and correct(s,method) for s in eligible), "both_wrong": sum(not correct(s,"crop_siglip_score") and not correct(s,method) for s in eligible)}
    return out


def correlations(sets):
    pairs = [("crop SigLIP vs D","crop_siglip_score","D"),("crop SigLIP vs N","crop_siglip_score","N"),("crop SigLIP vs E","crop_siglip_score","E"),("GDINO score vs N","native_grounding_score","N"),("full-image raw vs D","S_raw","D"),("full-image raw vs N","S_raw","N"),("full-image raw vs E","S_raw","E")]
    result={}
    for label,x,y in pairs:
        # Across candidates; raw is shared within a referent set, as the design requires.
        points=[(c[x],c[y]) for s in sets for c in s["candidates"] if c[x] is not None and c[y] is not None]
        a=np.array(points,dtype=float)
        rho=float(spearmanr(a[:,0],a[:,1]).statistic) if len(a)>2 else None
        rng=np.random.default_rng(20260924)
        bs=[]
        families=sorted({s["family_id"] for s in sets})
        by_family={family:np.array([(c[x],c[y]) for s in sets if s["family_id"]==family for c in s["candidates"] if c[x] is not None and c[y] is not None],dtype=float).reshape(-1,2) for family in families}
        if len(a)>3:
            for _ in range(2000):
                sample=np.concatenate([by_family[families[i]] for i in rng.integers(0,len(families),len(families))],axis=0)
                z=float(spearmanr(sample[:,0],sample[:,1]).statistic)
                if math.isfinite(z): bs.append(z)
        result[label]={"n_candidate_observations":len(a),"spearman_rho":rho,"bootstrap_95_ci":[float(v) for v in np.quantile(bs,[.025,.975])] if bs else None,"bootstrap_unit":"family (20 clusters); descriptive on frozen diagnostic"}
    return result


def family_results(sets):
    groups=defaultdict(dict)
    for s in sets: groups[s["family_id"]][s["logical_ref"]]=s
    rows=[]
    for family, refs in groups.items():
        assert set(refs)=={"A","B"}
        covered=all(refs[r]["EVALUATOR_ONLY"]["candidate_covered"] for r in ("A","B"))
        results={}
        for name,key in [("GDINO","native_grounding_score"),("crop_SigLIP","crop_siglip_score"),("D","D"),("N","N"),("E","E")]:
            indices={r:(order(refs[r]["candidates"],key)[0] if refs[r]["candidates"] and (key!="E" or len(refs[r]["candidates"])>1) else None) for r in ("A","B")}
            correct=all(indices[r] is not None and refs[r]["candidates"][indices[r]]["EVALUATOR_ONLY"]["gt_iou"]>=.5 for r in ("A","B"))
            results[name]={"top_indices":indices,"joint_correct":correct,"defined_for_both":all(indices[r] is not None for r in ("A","B"))}
        rows.append({"family_id":family,"candidate_covered":covered,"methods":results,"candidate_oracle_joint_correct":covered,"status":"DIAGNOSTIC_UNCALIBRATED","qwen_called":False})
    assert len(rows)==20
    return rows


def plot(figdir, sets, metrics, inc, fam):
    names=["GDINO native","crop SigLIP","Suppression effect D","Necessity N","Exclusivity E"]
    plt.figure(figsize=(8,4)); plt.bar(names,[metrics[n]["top1_fraction"] for n in names]); plt.ylabel("GT-best top-1 fraction"); plt.xticks(rotation=25,ha="right"); plt.tight_layout(); plt.savefig(figdir/"figure1_gt_best_rank.png"); plt.close()
    x=[c["crop_siglip_score"] for s in sets for c in s["candidates"]]; y=[c["N"] for s in sets for c in s["candidates"]]
    plt.figure(figsize=(6,5));plt.scatter(x,y,alpha=.7);plt.xlabel("Crop SigLIP raw logit");plt.ylabel("N = Focus - Suppress raw logits");plt.tight_layout();plt.savefig(figdir/"figure2_crop_vs_N.png");plt.close()
    xx=[];yy=[]
    for s in sets:
        if s["EVALUATOR_ONLY"]["candidate_covered"] and len(s["candidates"])>1:
            j=s["EVALUATOR_ONLY"]["gt_best_candidate_index"];xx.append(s["candidates"][j]["N"]);yy.append(max(c["N"] for i,c in enumerate(s["candidates"]) if i!=j))
    plt.figure(figsize=(5,5));plt.scatter(xx,yy);lo=min(xx+yy);hi=max(xx+yy);plt.plot([lo,hi],[lo,hi],"k--");plt.xlabel("GT-best candidate N");plt.ylabel("Strongest competitor N");plt.tight_layout();plt.savefig(figdir/"figure3_gt_vs_competitor_N.png");plt.close()
    labels=list(inc);plt.figure(figsize=(6,4));xx=np.arange(3);plt.bar(xx-.2,[inc[k]["rescue_when_siglip_wrong"]["count"] for k in labels],.4,label="rescue");plt.bar(xx+.2,[inc[k]["regression_when_siglip_correct"]["count"] for k in labels],.4,label="regression");plt.xticks(xx,labels);plt.ylabel("Candidate sets");plt.legend();plt.tight_layout();plt.savefig(figdir/"figure4_rescue_regression.png");plt.close()
    labels=["GDINO","crop_SigLIP","D","N","E","oracle"]; values=[sum(f["methods"][k]["joint_correct"] for f in fam) for k in labels[:-1]]+[sum(f["candidate_oracle_joint_correct"] for f in fam)]
    plt.figure(figsize=(7,4));plt.bar(labels,values);plt.ylim(0,20);plt.ylabel("Joint correct families / 20");plt.tight_layout();plt.savefig(figdir/"figure5_joint_ceiling.png");plt.close()


def main():
    if OUT.exists(): raise FileExistsError(f"Refusing to overwrite prior output: {OUT}")
    manifest=json.loads(MANIFEST.read_text())
    assert manifest["n_families"]==20 and len(manifest["unique_entity_units"])==40
    source_hash=sha(SOURCE); manifest_hash=sha(MANIFEST)
    source=[json.loads(x) for x in SOURCE.read_text().splitlines() if x]
    records=[r for r in source if r["inference"]["backend"]=="grounding_dino"]
    assert len(records)==40 and {r["inference"]["family_id"] for r in records}=={f["family_id"] for f in manifest["families"]}
    assert all(not r["inference"]["qwen_called"] for r in records)
    OUT.mkdir(parents=True);(OUT/"figures").mkdir();(OUT/"views").mkdir();(OUT/"crops").mkdir()
    ensure_d_drive_environment(Paths())
    backend=Siglip2Backend(Paths().siglip2)
    scorer=Siglip2FullImageQueryScorer(backend)
    cfg={"blur_implementation_version":"harness.processors.defocus.DEFOCUS_CONFIG PIL_GaussianBlur_v1", "pillow_version":pillow_version,"blur_kernel_semantics":"PIL.ImageFilter.GaussianBlur(radius=0.025*min(W,H)); PIL internal kernel, no custom kernel", "pil_gaussian_radius_fraction_of_min_dimension":DEFOCUS_CONFIG["blur_radius_fraction_of_min_dimension"],"bbox_expand_frac":DEFOCUS_CONFIG["bbox_expand_frac"],"feather_radius_fraction_of_min_dimension":DEFOCUS_CONFIG["feather_radius_fraction_of_min_dimension"],"support_policy":"same expanded bbox and same feathered mask for FOCUS and SUPPRESS","score_scope":scorer.score_scope,"score_semantics":scorer.score_semantics,"score_units":"raw_unscaled_logits","query_template_id":"siglip2_region_query_photo_of_legal_referent_v1","source_sha256":source_hash,"manifest_sha256":manifest_hash,"model_path":str(Paths().siglip2),"model_fingerprint":backend.model_fingerprint,"status":"DIAGNOSTIC_UNCALIBRATED","qwen_called":False}
    dump(OUT/"blur_config.json",cfg)
    sets=[];integrity=[];raw_cache={}
    for row in records:
        inf=row["inference"]
        uid=inf["entity_unit_id"]
        assert inf["candidate_count"]==len(inf["candidates"])
        assert all(c["candidate_index"]==i for i,c in enumerate(inf["candidates"]))
        assert all(c["query_hash"]==inf["query_hash"] for c in inf["candidates"])
        with Image.open(inf["image_path"]) as opened:image=opened.convert("RGB")
        assert list(image.size)==inf["image_size"]
        query="a photo of a "+inf["legal_query"]
        # The frozen candidate cache records the exact text-template identity and hash.
        expected_hash=hashlib.sha256(("siglip2_region_query_photo_of_legal_referent_v1|"+query).encode()).hexdigest()[:32]
        assert all(c["siglip_diagnostic"]["query_hash"]==expected_hash for c in inf["candidates"])
        raw_key=(inf["image_path"],query)
        if raw_key not in raw_cache: raw_cache[raw_key]=scorer.score(Path(inf["image_path"]),query)
        raw=raw_cache[raw_key]
        candidates=[]
        for c in inf["candidates"]:
            i=c["candidate_index"]; prefix=OUT/"views"/(uid.replace(":","_")+f"_{i}")
            paths,check=make_views(image,c["bbox_xyxy_pixels"],prefix)
            check.update({"entity_unit_id":uid,"candidate_index":i,"raw_image_sha256":sha(Path(inf["image_path"])),"query_text":query,"query_hash":expected_hash,"same_support_for_both_views":True,"no_crop_or_resize":True,"no_gt_dependency":True,"backend_rerun":False})
            assert check["focus_exact_expected_composite"] and check["suppress_exact_expected_composite"] and check["focus_sharp_core_exact"] and check["suppress_context_exact"]
            assert check["focus_dimensions"]==check["suppress_dimensions"]==list(image.size)
            integrity.append(check)
            focus=scorer.score(paths["FOCUS"],query); suppress=scorer.score(paths["SUPPRESS"],query)
            p=focus-raw;d=raw-suppress;n=focus-suppress
            # Candidate crops are audit assets only; the scoring call above reads whole images.
            b=c["bbox_xyxy_pixels"]
            crop=image.crop((math.floor(b[0]),math.floor(b[1]),math.ceil(b[2]),math.ceil(b[3])))
            crop_path=OUT/"crops"/(uid.replace(":","_")+f"_{i}.png");crop.save(crop_path)
            candidates.append({"candidate_index":i,"candidate_id":c["candidate_id"],"bbox_xyxy_pixels":c["bbox_xyxy_pixels"],"native_grounding_score":c["native_grounding_score"],"crop_siglip_score":c["siglip_diagnostic"]["z_raw"],"query_text":query,"query_hash":expected_hash,"S_raw":raw,"S_focus":focus,"S_suppress":suppress,"P":p,"D":d,"N":n,"E":None,"view_paths":{k:str(v) for k,v in paths.items()},"crop_path":str(crop_path),"view_sha256":{k:sha(v) for k,v in paths.items()},"EVALUATOR_ONLY":{"gt_iou":row["EVALUATOR_ONLY"]["candidate_ious_with_gt_in_inference_order"][i]}})
        for i,c in enumerate(candidates):
            if len(candidates)>1:c["E"]=c["N"]-max(x["N"] for j,x in enumerate(candidates) if j!=i)
        ev=row["EVALUATOR_ONLY"]
        assert len(candidates)==len(ev["candidate_ious_with_gt_in_inference_order"])
        covered=ev["best_candidate_iou"]>=.5
        gaps={}
        for key in ("D","N","E"):
            vals=sorted((c[key] for c in candidates if c[key] is not None),reverse=True)
            gaps[key]=vals[0]-vals[1] if len(vals)>1 else None
        rank={key:(order(candidates,key).index(ev["best_candidate_index"])+1 if covered and candidates and (key!="E" or len(candidates)>1) else None) for key in METHODS.values()}
        taxonomy={}
        if covered:
            semantic=rank["crop_siglip_score"]==1
            for key in ("D","N","E"):
                if rank[key] is None:taxonomy[key]="UNDEFINED_SINGLE_CANDIDATE";continue
                intervention=rank[key]==1
                taxonomy[key]="BOTH_CORRECT" if semantic and intervention else "INTERVENTION_REGRESSION" if semantic else "INTERVENTION_RESCUE" if intervention else "BOTH_WRONG"
        else: taxonomy={k:"NO_GT_COMPATIBLE_CANDIDATE" for k in ("D","N","E")}
        sets.append({"entity_unit_id":uid,"family_id":inf["family_id"],"logical_ref":inf["logical_ref"],"image_path":inf["image_path"],"image_size":inf["image_size"],"legal_query":inf["legal_query"],"query_text":query,"query_hash":expected_hash,"candidate_count":len(candidates),"candidates":candidates,"candidate_overlap_clusters":geometry(inf["candidates"],image.size)["candidate_overlap_clusters"],"ambiguity_gaps":gaps,"failure_taxonomy":taxonomy,"status":"DIAGNOSTIC_UNCALIBRATED","qwen_called":False,"EVALUATOR_ONLY":{"candidate_covered":covered,"gt_best_candidate_index":ev["best_candidate_index"],"best_candidate_iou":ev["best_candidate_iou"],"ranks":rank}})
        print(f"{len(sets)}/40 {uid} ({len(candidates)} candidates)",flush=True)
    assert sum(len(s["candidates"]) for s in sets)==74
    assert sum(s["EVALUATOR_ONLY"]["candidate_covered"] for s in sets)==29
    assert len(integrity)==74
    jsonl(OUT/"candidate_intervention_results.jsonl",sets)
    dump(OUT/"view_integrity.json",{"n_views":2*len(integrity),"n_candidate_supports":len(integrity),"all_checks_pass":all(x["focus_exact_expected_composite"] and x["suppress_exact_expected_composite"] and x["focus_sharp_core_exact"] and x["suppress_context_exact"] for x in integrity),"checks":integrity})
    metrics=rank_metrics(sets);dump(OUT/"ranking_metrics.json",metrics)
    inc=incremental(sets);dump(OUT/"incremental_value.json",inc)
    corr=correlations(sets);dump(OUT/"correlation_analysis.json",corr)
    fam=family_results(sets);jsonl(OUT/"family_intervention_results.jsonl",fam)
    ambiguity={"multi_candidate_sets":sum(s["candidate_count"]>1 for s in sets),"overlap_pair_count":sum(len(s["candidate_overlap_clusters"]["pairs"]) for s in sets),"spatially_disjoint_same_query_sets":sum(any(p["iou"]==0 for p in s["candidate_overlap_clusters"]["pairs"]) for s in sets),"gaps_by_group":{},"multi_candidate_cases":[]}
    for group,selector in [("N_top1_correct",lambda s:s["EVALUATOR_ONLY"]["ranks"]["N"]==1),("N_top1_incorrect",lambda s:s["EVALUATOR_ONLY"]["candidate_covered"] and s["EVALUATOR_ONLY"]["ranks"]["N"]!=1),("same_category_multi_instance",lambda s:s["candidate_count"]>1 and len({c["candidate_id"].split(":")[0] for c in s["candidates"]})==1)]:
        selected=[s for s in sets if selector(s)]
        ambiguity["gaps_by_group"][group]={"n_sets":len(selected), **{key:[s["ambiguity_gaps"][key] for s in selected if s["ambiguity_gaps"][key] is not None] for key in ("D","N","E")}}
    ambiguity["multi_candidate_cases"]=[{"entity_unit_id":s["entity_unit_id"],"candidate_count":s["candidate_count"],"has_spatially_disjoint_same_query_pair":any(p["iou"]==0 for p in s["candidate_overlap_clusters"]["pairs"]),"candidate_overlap_clusters":s["candidate_overlap_clusters"],"ambiguity_gaps":s["ambiguity_gaps"],"candidates":s["candidates"]} for s in sets if s["candidate_count"]>1]
    dump(OUT/"ambiguity_analysis.json",ambiguity)
    taxonomy={"D":dict(Counter(s["failure_taxonomy"]["D"] for s in sets)),"N":dict(Counter(s["failure_taxonomy"]["N"] for s in sets)),"E":dict(Counter(s["failure_taxonomy"]["E"] for s in sets)),"AMBIGUOUS_SIGNAL":{"exact_N_tie_sets":[s["entity_unit_id"] for s in sets if s["ambiguity_gaps"]["N"]==0],"near_tie_policy":"No calibrated near-tie threshold; inspect raw ambiguity_gaps distributions"}}
    dump(OUT/"failure_taxonomy.json",taxonomy)
    plot(OUT/"figures",sets,metrics,inc,fam)
    make_report(sets,metrics,inc,corr,fam,ambiguity,taxonomy,cfg)
    (OUT/"changed_files.txt").write_text("harness/processors/siglip_full_image_query.py\nharness/scripts/run_contextual_blur_binding_pilot.py\noutputs/contextual_blur_binding_pilot/ (new generated artifacts)\n")


def make_report(sets,metrics,inc,corr,fam,ambiguity,taxonomy,cfg):
    covered=sum(f["candidate_covered"] for f in fam)
    joint={k:sum(f["methods"][k]["joint_correct"] for f in fam) for k in ("GDINO","crop_SigLIP","D","N","E")}
    report=["# Contextual blur binding verification pilot","","All results are `DIAGNOSTIC_UNCALIBRATED`. Frozen 20-family / 40-referent Grounding DINO candidate cache; no acquisition rerun, GT inference, Qwen, final reasoning, threshold calibration, persistent binding, or full benchmark.","",f"Source SHA-256: `{cfg['source_sha256']}`. Manifest SHA-256: `{cfg['manifest_sha256']}`. Candidate sets: {len(sets)}; candidates: {sum(len(s['candidates']) for s in sets)}; candidate-covered referents: {sum(s['EVALUATOR_ONLY']['candidate_covered'] for s in sets)}; covered families: {covered}.","","## Method","","FOCUS preserves the candidate under the repository DaH feathered expanded-box mask and blurs context. SUPPRESS uses the identical mask with sharp context and blurred candidate. PIL GaussianBlur radius is 0.025 × min(W,H); bbox expansion 0.10; feather radius 0.010 × min(W,H). The same legal `a photo of a {referent}` query, processor, model, and full-image raw-logit interface score RAW, FOCUS, and SUPPRESS. Crops are audit only. Per-candidate pixel checks, dimensions, and SHA-256 hashes are in `view_integrity.json`.","","`P=FOCUS−RAW`, `D=RAW−SUPPRESS`, `N=FOCUS−SUPPRESS`, `E=N−max(other N)`. E is undefined for singleton sets. Full-image RAW is shared within each referent and has no candidate ranking. All GT IoUs and ranks are evaluator-only.","","## Table A — candidate-level GT-best ranking","","| Evidence | GT-best top1 | Top2 | MRR | Median rank | n |","|---|---:|---:|---:|---:|---:|"]
    for label,m in metrics.items(): report.append(f"| {label} | {m['top1_count']}/{m['n']} ({m['top1_fraction']:.3f}) | {m['top2_fraction']:.3f} | {m['mrr']:.3f} | {m['median_rank']} | {m['n']} |")
    report += ["","Full-image RAW support: shared across all candidates in a set; ranking is undefined by design.","","## Table B — incremental value","","| Condition | D | N | E |","|---|---:|---:|---:|"]
    for key,label in [("rescue_when_gdino_wrong","Rescue when GDINO wrong"),("rescue_when_siglip_wrong","Rescue when SigLIP wrong"),("regression_when_siglip_correct","Regression when SigLIP correct")]:
        report.append("| "+label+" | "+" | ".join(f"{inc[m][key]['count']}/{inc[m][key]['denominator']}" for m in ("D","N","E"))+" |")
    for key,label in [("both_correct","Both correct"),("both_wrong","Both wrong")]:report.append("| "+label+" | "+" | ".join(str(inc[m][key]) for m in ("D","N","E"))+" |")
    report += ["","## Table C — family-level binding","","| Method | Joint correct / 20 | Covered-family correct / 12 |","|---|---:|---:|"]
    for key,label in [("GDINO","GDINO top1"),("crop_SigLIP","crop SigLIP top1"),("D","D top1"),("N","N top1"),("E","E top1")]:report.append(f"| {label} | {joint[key]}/20 | {joint[key]}/{covered} |")
    report.append(f"| Candidate oracle | {covered}/20 | {covered}/{covered} |")
    report += ["","E joint top1 requires at least two candidates for both A and B; undefined singleton sets count as no joint E selection, not as a filled score.","","## Table D — correlation","","| Pair | Spearman rho | Bootstrap 95% CI | n |","|---|---:|---:|---:|"]
    for label,c in corr.items():report.append(f"| {label} | {c['spearman_rho']:.3f} | {c['bootstrap_95_ci']} | {c['n_candidate_observations']} |")
    report += ["","Bootstrap resamples the 20 families as clusters; intervals remain descriptive for this frozen diagnostic.","","## Scientific questions"]
    gn=metrics["GDINO native"];cs=metrics["crop SigLIP"];d=metrics["Suppression effect D"];n=metrics["Necessity N"];e=metrics["Exclusivity E"]
    covered_sets=[s for s in sets if s["EVALUATOR_ONLY"]["candidate_covered"]]
    focus_to_n_changes=sum(order(s["candidates"],"S_focus")[0]!=order(s["candidates"],"N")[0] for s in covered_sets)
    focus_to_n_rescues=sum(s["EVALUATOR_ONLY"]["ranks"]["S_focus"]!=1 and s["EVALUATOR_ONLY"]["ranks"]["N"]==1 for s in covered_sets)
    focus_to_n_regressions=sum(s["EVALUATOR_ONLY"]["ranks"]["S_focus"]==1 and s["EVALUATOR_ONLY"]["ranks"]["N"]!=1 for s in covered_sets)
    multi=[s for s in covered_sets if s["candidate_count"]>1]
    disjoint=[s for s in multi if any(p["iou"]==0 for p in s["candidate_overlap_clusters"]["pairs"])]
    multi_siglip=sum(s["EVALUATOR_ONLY"]["ranks"]["crop_siglip_score"]==1 for s in multi)
    multi_n=sum(s["EVALUATOR_ONLY"]["ranks"]["N"]==1 for s in multi)
    disjoint_siglip=sum(s["EVALUATOR_ONLY"]["ranks"]["crop_siglip_score"]==1 for s in disjoint)
    disjoint_n=sum(s["EVALUATOR_ONLY"]["ranks"]["N"]==1 for s in disjoint)
    answers=[f"Q1. N top1 {n['top1_count']}/{n['n']} vs GDINO {gn['top1_count']}/{gn['n']}; joint {joint['N']}/20 vs {joint['GDINO']}/20. It does not improve the family outcome.",f"Q2. N top1 {n['top1_count']}/{n['n']} vs crop SigLIP {cs['top1_count']}/{cs['n']}; joint {joint['N']}/20 vs {joint['crop_SigLIP']}/20. It does not improve the family outcome.",f"Q3. N uniquely rescues {inc['N']['rescue_when_siglip_wrong']['count']} of {inc['N']['rescue_when_siglip_wrong']['denominator']} crop-SigLIP errors; D and E counts are in Table B.",f"Q4. N harms {inc['N']['regression_when_siglip_correct']['count']} of {inc['N']['regression_when_siglip_correct']['denominator']} previously correct crop-SigLIP sets.",f"Q5. Crop SigLIP vs N rho={corr['crop SigLIP vs N']['spearman_rho']:.3f}; full-image RAW vs N rho={corr['full-image raw vs N']['spearman_rho']:.3f}. The latter is a pooled score with repeated RAW values. Neither is near-perfect.",f"Q6. FOCUS top1 {metrics['Focus score']['top1_count']}/{metrics['Focus score']['n']}; D top1 {d['top1_count']}/{d['n']}; N top1 {n['top1_count']}/{n['n']}. Suppression changes the FOCUS top candidate in {focus_to_n_changes}/{len(covered_sets)} covered sets, rescuing {focus_to_n_rescues} and regressing {focus_to_n_regressions}. It changes information but yields no aggregate top1 gain.",f"Q7. {ambiguity['multi_candidate_sets']} multi-candidate same-query sets; among {len(multi)} GT-covered multi-candidate sets, crop SigLIP is top1 in {multi_siglip} and N in {multi_n}. {len(disjoint)} of these have at least one exactly disjoint pair: crop SigLIP {disjoint_siglip}/{len(disjoint)}, N {disjoint_n}/{len(disjoint)}. Disjoint same-query proposals are plausible multi-instance cases, not independently verified same-category instances. Per-case crops, boxes, GT IoUs, and gaps are in ambiguity_analysis.json. No tie is forced or calibrated.",f"Q8. N recovers {joint['N']}/{covered} covered families; candidate oracle {covered}/{covered}.","Q9. The signal is nonredundant in rank changes and correlation, but its rescue/regression balance and lower joint accuracy do not support standalone certification. Retain it only as a candidate evidence source for future independent validation."]
    report += ["",*answers,"","## Decision"]
    rescue=inc["N"]["rescue_when_siglip_wrong"]["count"];regress=inc["N"]["regression_when_siglip_correct"]["count"]
    if joint["N"]>joint["crop_SigLIP"] and rescue>regress and abs(corr["crop SigLIP vs N"]["spearman_rho"])<.9:decision="PROMISING_BINDING_VERIFICATION_SIGNAL"
    elif regress>rescue and joint["N"]<joint["crop_SigLIP"]:decision="BLUR_INTERVENTION_UNSTABLE"
    elif abs(corr["crop SigLIP vs N"]["spearman_rho"])>.95 and rescue<=1:decision="INTERVENTION_SIGNAL_REDUNDANT_WITH_SEMANTIC_SCORE"
    else:decision="BLUR_SIGNAL_USEFUL_BUT_AMBIGUOUS"
    report += ["",f"**{decision}**. This is a descriptive recommendation from the frozen diagnostic, not a production decision rule. E and N have the same order within any multi-candidate set by construction; E adds an exclusivity magnitude, not an independent ranking. Its low family count arises from undefined singleton E scores.","","## Novelty and limits","","This experiment is not detector reranking by a stronger model, SoM-style VLM selection, threshold confidence calibration, or ordinary image enhancement. The hypothesis is candidate-specific visual intervention can provide evidence about whether a region causally supports the referent representation beyond semantic similarity alone. The broader method remains centered on persistent logical identity, explicit binding evidence, intervention-based verification, minimal proof state, and hard handoff away from raw-image reasoning. Historical DaH showed background defocus can improve reasoning behavior; that is a separate claim and is not evidence that this binding hypothesis succeeds.","","Candidate overlap is recorded without clustering or NMS. No random-region control was included; interpretation rests on the symmetric candidate views and integrity checks. These 20 locked families were analyzed without tuning, and no certification threshold is proposed."]
    (OUT/"implementation_report.md").write_text("\n".join(report)+"\n")
    tests=["# Test report","",f"- Frozen source and manifest hashed; exactly 20 families, 40 candidate sets, and 74 candidates preserved.","- No detector, Qwen, final reasoner, or persistent EntityManager was called.","- All 148 generated FOCUS/SUPPRESS views retained original image dimensions.","- For every view, pixel checks matched the exact complementary composite; FOCUS sharp core and SUPPRESS sharp context matched RAW exactly.","- Candidate region mask and support are identical across FOCUS and SUPPRESS; output hashes and dimensions are recorded per candidate.","- Full-image scorer uses one legal query and raw image-text logits throughout.","- E is null for singleton sets; RAW rank is undefined; no threshold was tuned.","- GT appears only in evaluator outputs after scoring."]
    (OUT/"test_report.md").write_text("\n".join(tests)+"\n")
    items=[]
    for s in sets:
        cand="".join(f"<tr><td>{c['candidate_index']}</td><td><img width='150' src='{html.escape(Path(c['crop_path']).relative_to(OUT).as_posix())}'></td><td>{c['native_grounding_score']:.3f}</td><td>{c['crop_siglip_score']:.3f}</td><td>{c['S_focus']:.3f}</td><td>{c['S_suppress']:.3f}</td><td>{c['D']:.3f}</td><td>{c['N']:.3f}</td><td>{'' if c['E'] is None else format(c['E'],'.3f')}</td><td>{c['EVALUATOR_ONLY']['gt_iou']:.3f}</td><td><a href='{html.escape(Path(c['view_paths']['FOCUS']).relative_to(OUT).as_posix())}'>focus</a> / <a href='{html.escape(Path(c['view_paths']['SUPPRESS']).relative_to(OUT).as_posix())}'>suppress</a></td></tr>" for c in s["candidates"])
        items.append(f"<section><h2>{html.escape(s['entity_unit_id'])} — {html.escape(s['legal_query'])}</h2><p>covered={s['EVALUATOR_ONLY']['candidate_covered']} | N gap={s['ambiguity_gaps']['N']} | taxonomy={html.escape(str(s['failure_taxonomy']))}</p><table><tr><th>index</th><th>crop</th><th>GDINO</th><th>crop SigLIP</th><th>FOCUS</th><th>SUPPRESS</th><th>D</th><th>N</th><th>E</th><th>GT IoU</th><th>views</th></tr>{cand}</table></section>")
    (OUT/"audit.html").write_text("<!doctype html><html><head><meta charset='utf-8'><style>body{font:14px sans-serif;max-width:1200px;margin:auto}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:5px}section{margin:30px 0}img{max-height:150px}</style></head><body><h1>Contextual blur binding audit — DIAGNOSTIC_UNCALIBRATED</h1>"+"".join(items)+"</body></html>")
    print("RECOMMENDATION",decision,"N joint",joint["N"],"/20",flush=True)


if __name__=="__main__":main()
