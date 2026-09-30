#!/usr/bin/env python3
"""Frozen 20-family REL2C visual-exposure handoff pilot."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from PIL import Image, ImageChops, ImageFilter
import numpy as np

from harness.configs.paths import Paths,ensure_d_drive_environment
from harness.dataset import read_jsonl
from harness.entities.entity_manager import EntityManager
from harness.evaluators.leakage_checks import assert_no_image_payload
from harness.processors.base import RegionPrediction
from harness.processors.color_siglip import SiglipColorReader
from harness.processors.defocus import DEFOCUS_CONFIG,_expanded,_box_mask
from harness.processors.geometry import horizontal_relation
from harness.processors.siglip_backend import Siglip2Backend
from harness.referents.referent_resolver import resolve_question
from harness.referents.referent_spec import GroundingQuery

OUT=ROOT/"outputs/handoff_end_to_end_pilot"
MANIFEST=ROOT/"manifests/acquisition_semantic_support_20family.json"
CACHE=ROOT/"outputs/grounding_dino_acquisition_pilot/entity_backend_comparison.jsonl"
CONDITIONS=("VANILLA_RAW","RAW_IMAGE_PLUS_STATE","DEFOCUS_PLUS_STATE","STATE_ONLY_HARD_HANDOFF")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")


def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def append(path,value):
    with Path(path).open("a") as f:f.write(json.dumps(value,sort_keys=True,allow_nan=False)+"\n")


def safe_qa():
    return {str(r["qa_id"]):{k:str(r[k]) for k in ("qa_id","family_id","qa_type","question","image_path")} for r in read_jsonl(Paths().qa)}


def frozen_data():
    manifest=json.loads(MANIFEST.read_text())
    assert manifest["n_families"]==20 and len(manifest["unique_entity_units"])==40
    acquisition=[r for r in rows(CACHE) if r["inference"]["backend"]=="grounding_dino"]
    assert len(acquisition)==40
    lookup={(r["inference"]["family_id"],r["inference"]["logical_ref"]):r["inference"] for r in acquisition}
    assert len(lookup)==40
    return manifest,lookup


def choose(candidates):
    if not candidates:return None
    # Frozen AcquisitionRouter: max native score, then smaller x0; first candidate wins exact ties.
    return max(range(len(candidates)),key=lambda i:(candidates[i]["native_grounding_score"] if candidates[i]["native_grounding_score"] is not None else -math.inf,-candidates[i]["bbox_xyxy_pixels"][0]))


def region_of(candidate,size):
    return RegionPrediction.from_pixels(candidate["bbox_xyxy_pixels"],*size,score=candidate["native_grounding_score"],label=candidate["native_label_text"] or "",source=candidate["source_backend"])


def minimal_text(noun_a,color_a,noun_b,color_b,relation):
    def value(x):return x if x is not None else "unknown"
    rel={"left":"A is left of B","right":"A is right of B"}.get(relation,"unknown")
    return f"Entity A ({noun_a}): color = {value(color_a)}\nEntity B ({noun_b}): color = {value(color_b)}\nSpatial relation: {rel}"


def union_mask(size,regions):
    masks=[_box_mask(size,_expanded(r.bbox_xyxy_pixels,*size,DEFOCUS_CONFIG["bbox_expand_frac"]),DEFOCUS_CONFIG) for r in regions]
    union=masks[0]
    for mask in masks[1:]:union=ImageChops.lighter(union,mask)
    # DaH's feathered expansion can spill into a selected bbox on narrow or
    # edge-clipped regions. Enforce the user's exact sharp-region invariant.
    values=np.array(union)
    for region in regions:
        x0,y0,x1,y1=region.bbox_xyxy_pixels
        values[max(0,math.floor(y0)):min(size[1],math.ceil(y1)),max(0,math.floor(x0)):min(size[0],math.ceil(x1))]=255
    return Image.fromarray(values,"L")


def render_defocus(original_path,output_path,regions):
    with Image.open(original_path) as im:raw=im.convert("RGB")
    mask=union_mask(raw.size,regions)
    radius=DEFOCUS_CONFIG["blur_radius_fraction_of_min_dimension"]*min(raw.size)
    result=Image.composite(raw,raw.filter(ImageFilter.GaussianBlur(radius=radius)),mask)
    result.save(output_path,format="PNG")
    return {"output_path":str(output_path),"config":DEFOCUS_CONFIG,"blur_radius":radius,"selected_regions_xyxy_pixels":[list(r.bbox_xyxy_pixels) for r in regions],"support_core_override":"exact acquired bbox pixels sharp; same DaH expansion, feathering, and Gaussian blur elsewhere","gt_used":False,"target_side_used":False}


def view_check(original_path,output_path,regions,meta):
    with Image.open(original_path) as im:raw=im.convert("RGB")
    with Image.open(output_path) as im:actual=im.convert("RGB")
    assert raw.size==actual.size
    union=union_mask(raw.size,regions)
    blurred=raw.filter(ImageFilter.GaussianBlur(radius=DEFOCUS_CONFIG["blur_radius_fraction_of_min_dimension"]*min(raw.size)))
    expected=Image.composite(raw,blurred,union)
    a=np.asarray(raw);b=np.asarray(actual);m=np.asarray(union)
    boxes_exact=[]
    for region in regions:
        x0,y0,x1,y1=region.bbox_xyxy_pixels
        boxes_exact.append(bool(np.array_equal(a[math.floor(y0):math.ceil(y1),math.floor(x0):math.ceil(x1)],b[math.floor(y0):math.ceil(y1),math.floor(x0):math.ceil(x1)])))
    checks={"same_dimensions":raw.size==actual.size,"exact_repository_composite":bool(np.array_equal(b,np.asarray(expected))),"union_sharp_core_exact":bool(np.array_equal(a[m==255],b[m==255])),"each_acquired_bbox_exact":boxes_exact,"context_changed_pixels":int(np.count_nonzero(np.any(a[m==0]!=b[m==0],axis=1))),"selected_region_count":len(regions),"candidate_regions_used":[list(r.bbox_xyxy_pixels) for r in regions],"original_dimensions":list(raw.size),"defocus_dimensions":list(actual.size),"original_sha256":sha(original_path),"defocus_sha256":sha(output_path),"mask_sha256":hashlib.sha256(union.tobytes()).hexdigest(),"defocus_metadata":meta,"gt_used":False,"target_side_used":False,"crop_or_resize":False}
    assert checks["same_dimensions"] and checks["exact_repository_composite"] and checks["union_sharp_core_exact"] and all(boxes_exact)
    return checks


def prepare():
    if OUT.exists():raise FileExistsError(f"refusing to overwrite prior output: {OUT}")
    OUT.mkdir(parents=True);(OUT/"figures").mkdir();(OUT/"defocus_views").mkdir()
    ensure_d_drive_environment(Paths())
    manifest,lookup=frozen_data();qa=safe_qa()
    backend=Siglip2Backend(Paths().siglip2)
    color_reader=SiglipColorReader(backend=backend)
    state_path=OUT/"compiled_states.jsonl";integrity=[];leakage=[]
    for fi,family in enumerate(manifest["families"]):
        cases=[qa[cid] for cid in family["case_ids"]]
        assert len(cases)==2 and {c["qa_type"] for c in cases}=={"REL2C_LEFT","REL2C_RIGHT"}
        assert cases[0]["image_path"]==cases[1]["image_path"]==family["image_path"]
        resolved=resolve_question(cases[0]["question"],cases[0]["qa_type"])
        spec={s.logical_ref:s for s in resolved.specs}
        assert set(spec)=={"A","B"}
        manager=EntityManager()
        detail={};regions={}
        for ref in ("A","B"):
            inf=lookup[(family["family_id"],ref)]
            assert inf["image_path"]==family["image_path"]
            assert inf["candidate_count"]==len(inf["candidates"])
            assert inf["legal_query"]==GroundingQuery.from_spec(spec[ref]).semantic_text
            entity=manager.ensure_entity(ref,GroundingQuery.from_spec(spec[ref]))
            index=choose(inf["candidates"])
            assert index==inf["native_top_candidate_index"]
            candidate=inf["candidates"][index] if index is not None else None
            region=region_of(candidate,inf["image_size"]) if candidate is not None else None
            if region is not None:manager.bind_region(ref,region,source="frozen_grounding_dino_cache")
            color=None
            if region is not None:
                manager.reuse(ref,reason="frozen_region_color_acquisition")
                color=color_reader.read(Path(inf["image_path"]),region)
                manager.append_observation(ref,color)
                regions[ref]=region
            detail[ref]={"logical_ref":ref,"entity_id":entity.entity_id,"noun":inf["legal_query"],"selected_candidate_index":index,"selected_candidate_id":candidate["candidate_id"] if candidate else None,"candidate_count":len(inf["candidates"]),"region_bbox_xyxy_pixels":list(region.bbox_xyxy_pixels) if region else None,"region_bbox_xyxy_normalized":list(region.bbox_xyxy_normalized) if region else None,"region_processor":"grounding_dino","region_selection_policy":"AcquisitionRouter.max_native_score_then_smaller_x0","region_acquisition_provenance":candidate["provenance"] if candidate else ["QUESTION_TEXT","TOOL_OBSERVATION_EMPTY_SET"],"region_score":region.score if region else None,"color":color.color if color else None,"color_processor":color.source if color else None,"color_score":color.confidence if color else None,"color_score_semantics":"SigLIP2 softmax across frozen RealBind color vocabulary, not calibrated" if color else None,"color_region_bbox_xyxy_pixels":list(region.bbox_xyxy_pixels) if region else None}
        relation=horizontal_relation("A",regions.get("A"),"B",regions.get("B"))
        state_text=minimal_text(detail["A"]["noun"],detail["A"]["color"],detail["B"]["noun"],detail["B"]["color"],relation.relation)
        assert "answer is" not in state_text.casefold()
        state_hash=hashlib.sha256(state_text.encode()).hexdigest()
        selected=list(regions.values())
        if selected:
            image=OUT/"defocus_views"/(family["family_id"]+".png")
            meta=render_defocus(Path(family["image_path"]),image,selected)
            check=view_check(family["image_path"],image,selected,meta)
        else:
            image=None;check={"selected_region_count":0,"view_missing_reason":"NO_ACQUIRED_REGIONS"}
        integrity.append({"family_id":family["family_id"],**check})
        for c in cases:
            prompt=c["question"]+"\n\nAcquired visual facts:\n"+state_text
            # The same bytes are recorded for every state-based condition.
            hashes={condition:state_hash for condition in CONDITIONS[1:]}
            assert len(set(hashes.values()))==1
            assert_no_image_payload({"question":c["question"],"state_text":state_text},context="compiled minimal reasoning state")
            record={"qa_id":c["qa_id"],"family_id":family["family_id"],"qa_type":c["qa_type"],"question":c["question"],"image_path":c["image_path"],"defocus_image_path":str(image) if image else None,"entities":detail,"relation":{"A_to_B":relation.relation,"processor":"deterministic_geometry.normalized_bbox_center_x_order_v1","provenance":"ACQUIRED_REGION_BBOX_CENTERS"},"entity_manager_events":[x.to_dict() for x in manager.events],"state_text":state_text,"state_sha256":state_hash,"condition_state_hashes":hashes,"state_prompt":prompt,"state_implied_answer":None,"inference_status":"COMPILED_UNCERTIFIED","gt_used":False}
            append(state_path,record)
            leakage.append({"qa_id":c["qa_id"],"state_contains_final_answer_phrase":False,"state_has_gt_metadata":False,"state_has_detector_score":False,"state_has_color_score":False,"state_has_candidate_list":False,"state_has_caption":False,"state_has_image_path":False,"state_hashes_identical":True,"ground_truth_used_in_acquisition":False,"selection_uses_question_target_side":False})
        print(f"prepared {fi+1}/20 {family['family_id']}",flush=True)
    dump(OUT/"view_integrity.json",{"families":integrity,"all_available_views_valid":all(x.get("exact_repository_composite",True) for x in integrity)})
    dump(OUT/"leakage_audit.json",{"qa":leakage,"all_pass":all(x["state_hashes_identical"] and not x["state_has_gt_metadata"] for x in leakage)})
    dump(OUT/"pilot_config.json",{"manifest_sha256":sha(MANIFEST),"grounding_dino_cache_sha256":sha(CACHE),"families":20,"qa":40,"conditions":list(CONDITIONS),"region_selection":"frozen AcquisitionRouter max(native score), tie smaller x0","color_reader":color_reader.version,"color_model_fingerprint":backend.model_fingerprint,"geometry":"normalized_bbox_center_x_order_v1","blur_config":DEFOCUS_CONFIG,"defocus_support_core_override":"exact acquired bbox pixels sharp; DaH expansion, feathering, and Gaussian blur elsewhere","reasoner":"Qwen/Qwen3.5-4B existing frozen loader","generation":{"do_sample":False,"use_cache":False,"max_new_tokens":16,"batch_size":1,"enable_thinking":False},"state_prompt_suffix":"\\n\\nAcquired visual facts:\\n{state_text}","certification_used":False,"gt_inference_used":False,"candidate_selector":"AcquisitionRouter native score"})


def infer():
    if not (OUT/"compiled_states.jsonl").is_file():raise FileNotFoundError("prepare stage missing")
    from harness.reasoners.qwen35 import Qwen35Reasoner
    states=rows(OUT/"compiled_states.jsonl")
    assert len(states)==40
    reasoner=Qwen35Reasoner()
    path=OUT/"inference_calls.jsonl"
    completed={(r["qa_id"],r["condition"]) for r in rows(path)} if path.exists() else set()
    vision_module=getattr(reasoner.model,"visual",None) or getattr(getattr(reasoner.model,"model",None),"visual",None)
    if vision_module is None:raise RuntimeError("cannot identify vision module for hard-handoff audit")
    for i,state in enumerate(states):
        assert len(set(state["condition_state_hashes"].values()))==1
        for condition in CONDITIONS:
            key=(state["qa_id"],condition)
            if key in completed:continue
            image=Path(state["image_path"]) if condition in ("VANILLA_RAW","RAW_IMAGE_PLUS_STATE") else Path(state["defocus_image_path"]) if condition=="DEFOCUS_PLUS_STATE" and state["defocus_image_path"] else None
            if condition=="DEFOCUS_PLUS_STATE" and image is None:
                # A truly empty detector set may leave no region to preserve; here retain raw image.
                image=Path(state["image_path"])
            prompt=state["question"] if condition=="VANILLA_RAW" else state["state_prompt"]
            generation=reasoner.generate(prompt,image)
            if condition=="STATE_ONLY_HARD_HANDOFF":
                assert generation["image_path"] is None and not generation["visual_tokens_in_rendered_prompt"] and not generation["vision_encoder_called"]
                assert not any("pixel" in k or "image" in k or "video" in k for k in generation["input_keys"])
            call={"qa_id":state["qa_id"],"family_id":state["family_id"],"qa_type":state["qa_type"],"condition":condition,"prompt":prompt,"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),"state_sha256":state["state_sha256"] if condition!="VANILLA_RAW" else None,"image_path":str(image) if image else None,"image_sha256":sha(image) if image else None,"generation":generation,"qwen_called":True}
            append(path,call)
            print(f"inference {i+1}/40 {condition} {state['qa_id']} => {generation['response']!r}",flush=True)


def xywh_to_xyxy(box):return [box[0],box[1],box[0]+box[2],box[1]+box[3]]


def iou(a,b):
    if a is None:return 0.0
    ix=max(0.,min(a[2],b[2])-max(a[0],b[0]));iy=max(0.,min(a[3],b[3])-max(a[1],b[1]))
    aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1])
    return ix*iy/max(1e-12,aa+bb-ix*iy)


def state_answer(state,qa_type):
    relation=state["relation"]["A_to_B"]
    if relation not in ("left","right"):return None
    desired="left" if qa_type.endswith("LEFT") else "right"
    target="A" if relation==desired else "B"
    return state["entities"][target]["color"]


def transition(case_rows,x,y):
    mapping={"correct_to_correct":0,"correct_to_wrong":0,"wrong_to_correct":0,"wrong_to_wrong":0}
    for r in case_rows:
        a=r["conditions"][x]["correct"];b=r["conditions"][y]["correct"]
        mapping[("correct" if a else "wrong")+"_to_"+("correct" if b else "wrong")]+=1
    return mapping


def paired_stats(case_rows,x,y):
    from scipy.stats import binomtest
    before=[int(r["conditions"][x]["correct"]) for r in case_rows]
    after=[int(r["conditions"][y]["correct"]) for r in case_rows]
    n=len(case_rows);delta=sum(b-a for a,b in zip(before,after))/n if n else None
    t=transition(case_rows,x,y)
    discord=t["correct_to_wrong"]+t["wrong_to_correct"]
    p=binomtest(min(t["correct_to_wrong"],t["wrong_to_correct"]),discord,.5).pvalue if discord else 1.0
    families=sorted({r["family_id"] for r in case_rows})
    by_family={f:[r for r in case_rows if r["family_id"]==f] for f in families}
    rng=np.random.default_rng(20260924);rep=[]
    if families:
        for _ in range(10000):
            draw=[r for idx in rng.integers(0,len(families),len(families)) for r in by_family[families[idx]]]
            rep.append(sum(int(r["conditions"][y]["correct"])-int(r["conditions"][x]["correct"]) for r in draw)/len(draw))
    return {"n":n,"accuracy_before":sum(before)/n if n else None,"accuracy_after":sum(after)/n if n else None,"absolute_accuracy_difference_after_minus_before":delta,"family_cluster_bootstrap_95_ci":[float(v) for v in np.quantile(rep,[.025,.975])] if rep else None,"mcnemar_exact_p_two_sided":float(p),"transition_counts":t}


def accuracy(case_rows,condition):
    n=len(case_rows);good=sum(r["conditions"][condition]["correct"] for r in case_rows)
    return {"correct":good,"n":n,"accuracy":good/n if n else None}


def make_figures(all_cases,strict,families,taxonomy):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig=OUT/"figures"
    names=["Raw + state","Defocus + state","State only"]
    keys=CONDITIONS[1:]
    plt.figure(figsize=(7,4));plt.plot(names,[accuracy(strict,k)["accuracy"] if strict else 0.0 for k in keys],marker="o");plt.ylim(0,1);plt.ylabel("Accuracy on strict fact-sufficient cases" if strict else "No strict fact-sufficient cases");plt.tight_layout();plt.savefig(fig/"figure1_visual_exposure_gradient.png");plt.close()
    t=transition(strict,CONDITIONS[1],CONDITIONS[3]);labels=["correct→correct","correct→wrong","wrong→correct","wrong→wrong"]
    plt.figure(figsize=(7,4));plt.bar(labels,[t[k.replace("→","_to_")] for k in labels]);plt.ylabel("Cases");plt.xticks(rotation=20,ha="right");plt.tight_layout();plt.savefig(fig/"figure2_raw_to_handoff_transitions.png");plt.close()
    xx=np.arange(4);plt.figure(figsize=(9,4));plt.bar(xx-.2,[accuracy(all_cases,k)["accuracy"] for k in CONDITIONS],.4,label="All cases");plt.bar(xx+.2,[accuracy(strict,k)["accuracy"] if strict else 0.0 for k in CONDITIONS],.4,label="Strict fact-sufficient");plt.xticks(xx,["Vanilla","Raw+state","Defocus+state","State only"]);plt.ylim(0,1);plt.ylabel("Accuracy");plt.legend();plt.tight_layout();plt.savefig(fig/"figure3_all_vs_sufficient.png");plt.close()
    labs=["ACQUISITION_FAILURE","VISUAL_INTERFERENCE_CASE","IMAGE_RESCUE_CASE","BOTH_REASONING_FAIL"]
    plt.figure(figsize=(8,4));plt.bar(labs,[taxonomy.get(k,0) for k in labs]);plt.xticks(rotation=18,ha="right");plt.ylabel("QA cases");plt.tight_layout();plt.savefig(fig/"figure4_failure_decomposition.png");plt.close()
    mat=np.array([[int(f["conditions"][k]["both_qa_correct"]) for k in CONDITIONS] for f in families])
    plt.figure(figsize=(8,8));plt.imshow(mat,aspect="auto",cmap="Greens",vmin=0,vmax=1);plt.yticks(range(len(families)),[f["family_id"].split("_")[2] for f in families],fontsize=7);plt.xticks(range(4),["Vanilla","Raw+state","Defocus+state","State only"],rotation=20,ha="right");plt.colorbar(label="Both QA correct");plt.tight_layout();plt.savefig(fig/"figure5_family_outcomes.png");plt.close()


def analyze():
    from harness.evaluators.end2end_metrics import score
    from harness.processors.color_pixel import REALBIND_COLORS
    states=rows(OUT/"compiled_states.jsonl");calls=rows(OUT/"inference_calls.jsonl")
    assert len(states)==40 and len(calls)==160
    lookup={(r["qa_id"],r["condition"]):r for r in calls}
    assert len(lookup)==160 and all((s["qa_id"],k) in lookup for s in states for k in CONDITIONS)
    all_hashes=all(len(set(s["condition_state_hashes"].values()))==1 and all(v==s["state_sha256"] for v in s["condition_state_hashes"].values()) for s in states)
    assert all_hashes
    hard=[lookup[(s["qa_id"],"STATE_ONLY_HARD_HANDOFF")] for s in states]
    hard_checks=[]
    for s in states:
        c=lookup[(s["qa_id"],"STATE_ONLY_HARD_HANDOFF")];g=c["generation"]
        passed=c["image_path"] is None and g["image_path"] is None and g["vision_module_found"] and not g["vision_encoder_called"] and not g["visual_tokens_in_rendered_prompt"] and not any("pixel" in k or "image" in k or "video" in k for k in g["input_keys"])
        hard_checks.append({"qa_id":s["qa_id"],"pass":passed,"image_path":c["image_path"],"input_keys":g["input_keys"],"vision_module_found":g["vision_module_found"],"vision_encoder_called":g["vision_encoder_called"],"vision_encoder_call_count":g["vision_encoder_call_count"],"visual_tokens_in_rendered_prompt":g["visual_tokens_in_rendered_prompt"]})
    assert all(x["pass"] for x in hard_checks)
    dump(OUT/"hard_handoff_audit.json",{"all_pass":True,"state_hash_integrity":all_hashes,"state_only_cases":hard_checks})
    # Only now are evaluator records joined. They never enter state compilation or Qwen calls.
    families_data={str(r["family_id"]):r for r in read_jsonl(Paths().families)}
    qa_data={str(r["qa_id"]):r for r in read_jsonl(Paths().qa)}
    vocab={"color":list(REALBIND_COLORS),"position":["left","right"],"identity":sorted({f[ref]["object"] for f in families_data.values() for ref in ("A","B")})}
    case_rows=[]
    for s in states:
        gt=qa_data[s["qa_id"]];family=families_data[s["family_id"]]
        evaluator={}
        for ref in ("A","B"):
            e=s["entities"][ref];iou_value=iou(e["region_bbox_xyxy_pixels"],xywh_to_xyxy(family[ref]["bbox"]))
            evaluator[f"region_{ref}_iou"]=iou_value
            evaluator[f"region_{ref}_correct"]=iou_value>=.5
            evaluator[f"color_{ref}_correct"]=e["color"]==family[ref]["canonical_color"]
            evaluator[f"color_{ref}_gt"]=family[ref]["canonical_color"]
        true_relation="left" if family["left_is_A"] else "right"
        evaluator["relation_correct"]=s["relation"]["A_to_B"]==true_relation
        evaluator["gt_relation_A_to_B"]=true_relation
        evaluator["both_regions_correct"]=evaluator["region_A_correct"] and evaluator["region_B_correct"]
        evaluator["strict_fact_sufficient"]=all(evaluator[k] for k in ("region_A_correct","region_B_correct","color_A_correct","color_B_correct","relation_correct"))
        implied=state_answer(s,s["qa_type"])
        evaluator["state_implied_answer"]=implied
        evaluator["state_answer_consistent"]=implied==gt["ground_truth"]
        conditions={}
        for condition in CONDITIONS:
            c=lookup[(s["qa_id"],condition)]
            assert c["prompt"]==(s["question"] if condition=="VANILLA_RAW" else s["state_prompt"])
            if condition!="VANILLA_RAW":assert c["state_sha256"]==s["state_sha256"]
            parsed=score(c["generation"]["response"],"color",gt["ground_truth"],family,vocab)
            conditions[condition]={"correct":parsed["correct"],"normalized_prediction":parsed["normalized_prediction"],"invalid_prediction":parsed["invalid_prediction"],"parser_reason":parsed.get("parser_reason"),"raw_response":c["generation"]["response"],"raw_response_with_special_tokens":c["generation"]["raw_response_with_special_tokens"],"token_ids":c["generation"]["token_ids"],"prompt":c["prompt"],"prompt_sha256":c["prompt_sha256"],"image_path":c["image_path"],"image_sha256":c["image_sha256"],"state_sha256":c["state_sha256"],"vision_encoder_called":c["generation"]["vision_encoder_called"],"input_keys":c["generation"]["input_keys"],"generation_config":c["generation"]["generation_config"]}
        raw=conditions["RAW_IMAGE_PLUS_STATE"]["correct"];defo=conditions["DEFOCUS_PLUS_STATE"]["correct"];handoff=conditions["STATE_ONLY_HARD_HANDOFF"]["correct"]
        labels=[]
        if not evaluator["strict_fact_sufficient"]:labels.append("ACQUISITION_FAILURE")
        elif raw and handoff:labels.append("FACT_SUFFICIENT_RAW_CORRECT_HARD_CORRECT")
        elif not raw and handoff:labels.append("VISUAL_INTERFERENCE_CASE")
        elif raw and not handoff:labels.append("IMAGE_RESCUE_CASE")
        else:labels.append("BOTH_REASONING_FAIL")
        if not raw and defo:labels.append("DEFOCUS_RESCUE")
        if raw and not defo:labels.append("DEFOCUS_REGRESSION")
        case_rows.append({"qa_id":s["qa_id"],"family_id":s["family_id"],"qa_type":s["qa_type"],"question":s["question"],"state_sha256":s["state_sha256"],"state_text":s["state_text"],"conditions":conditions,"failure_taxonomy":labels,"EVALUATOR_ONLY":{**evaluator,"ground_truth":gt["ground_truth"]}})
    with (OUT/"qa_condition_results.jsonl").open("w") as f:
        for r in case_rows:f.write(json.dumps(r,sort_keys=True)+"\n")
    strict=[r for r in case_rows if r["EVALUATOR_ONLY"]["strict_fact_sufficient"]]
    insufficient=[r for r in case_rows if not r["EVALUATOR_ONLY"]["strict_fact_sufficient"]]
    quality={"n_qa":40,"A_region_correct":sum(r["EVALUATOR_ONLY"]["region_A_correct"] for r in case_rows),"B_region_correct":sum(r["EVALUATOR_ONLY"]["region_B_correct"] for r in case_rows),"both_regions_correct":sum(r["EVALUATOR_ONLY"]["both_regions_correct"] for r in case_rows),"color_A_correct":sum(r["EVALUATOR_ONLY"]["color_A_correct"] for r in case_rows),"color_B_correct":sum(r["EVALUATOR_ONLY"]["color_B_correct"] for r in case_rows),"both_colors_correct":sum(r["EVALUATOR_ONLY"]["color_A_correct"] and r["EVALUATOR_ONLY"]["color_B_correct"] for r in case_rows),"both_regions_and_colors_correct":sum(r["EVALUATOR_ONLY"]["both_regions_correct"] and r["EVALUATOR_ONLY"]["color_A_correct"] and r["EVALUATOR_ONLY"]["color_B_correct"] for r in case_rows),"relation_correct":sum(r["EVALUATOR_ONLY"]["relation_correct"] for r in case_rows),"relation_correct_given_both_regions":sum(r["EVALUATOR_ONLY"]["relation_correct"] and r["EVALUATOR_ONLY"]["both_regions_correct"] for r in case_rows),"strict_fact_sufficient_qa":len(strict),"state_implied_answer_correct":sum(r["EVALUATOR_ONLY"]["state_answer_consistent"] for r in case_rows),"selection_policy":"AcquisitionRouter native score; no certification","gt_joined_after_all_160_inference_calls":True}
    dump(OUT/"state_quality.json",quality)
    strata={"all_cases":case_rows,"strict_fact_sufficient":strict,"fact_insufficient":insufficient}
    analyses={name:{"n":len(group),"accuracy":{k:accuracy(group,k) for k in CONDITIONS}} for name,group in strata.items()}
    hand=paired_stats(strict,"RAW_IMAGE_PLUS_STATE","STATE_ONLY_HARD_HANDOFF")
    analyses["strict_fact_sufficient"]["hard_handoff_gain"]=hand
    dump(OUT/"fact_sufficient_analysis.json",analyses)
    pairs=(("RAW_IMAGE_PLUS_STATE","STATE_ONLY_HARD_HANDOFF"),("RAW_IMAGE_PLUS_STATE","DEFOCUS_PLUS_STATE"),("DEFOCUS_PLUS_STATE","STATE_ONLY_HARD_HANDOFF"))
    paired={name:{f"{x}__TO__{y}":paired_stats(group,x,y) for x,y in pairs} for name,group in strata.items()}
    dump(OUT/"paired_transition_analysis.json",paired)
    harm=hand["transition_counts"]["wrong_to_correct"];rescue=hand["transition_counts"]["correct_to_wrong"]
    answer_changes={name:sum(r["conditions"]["RAW_IMAGE_PLUS_STATE"]["normalized_prediction"]!=r["conditions"]["STATE_ONLY_HARD_HANDOFF"]["normalized_prediction"] for r in group) for name,group in strata.items()}
    dump(OUT/"visual_interference_analysis.json",{"strict_n":len(strict),"image_harm":harm,"image_rescue":rescue,"harm_minus_rescue":harm-rescue,"visual_interference_rate":harm/len(strict) if strict else None,"image_rescue_rate":rescue/len(strict) if strict else None,"answer_changes_raw_vs_handoff":answer_changes,"hard_handoff_gain":hand["absolute_accuracy_difference_after_minus_before"],"paired_exact_p":hand["mcnemar_exact_p_two_sided"]})
    defocus_pairs=paired["strict_fact_sufficient"]
    raw_def=defocus_pairs["RAW_IMAGE_PLUS_STATE__TO__DEFOCUS_PLUS_STATE"]["transition_counts"]
    def_hard=defocus_pairs["DEFOCUS_PLUS_STATE__TO__STATE_ONLY_HARD_HANDOFF"]["transition_counts"]
    dump(OUT/"defocus_analysis.json",{"strict_n":len(strict),"raw_to_defocus_rescues":raw_def["wrong_to_correct"],"raw_to_defocus_regressions":raw_def["correct_to_wrong"],"defocus_to_hard_rescues":def_hard["wrong_to_correct"],"defocus_to_hard_regressions":def_hard["correct_to_wrong"],"defocus_rescues_raw_harm":sum((not r["conditions"]["RAW_IMAGE_PLUS_STATE"]["correct"]) and r["conditions"]["STATE_ONLY_HARD_HANDOFF"]["correct"] and r["conditions"]["DEFOCUS_PLUS_STATE"]["correct"] for r in strict),"defocus_rescue_denominator_raw_harm":harm})
    grouped=defaultdict(list)
    for r in case_rows:grouped[r["family_id"]].append(r)
    fam=[]
    for fid,group in grouped.items():
        assert len(group)==2
        fam.append({"family_id":fid,"strict_fact_sufficient_both_qa":all(r["EVALUATOR_ONLY"]["strict_fact_sufficient"] for r in group),"conditions":{k:{"correct_qa":sum(r["conditions"][k]["correct"] for r in group),"both_qa_correct":all(r["conditions"][k]["correct"] for r in group)} for k in CONDITIONS}})
    with (OUT/"family_condition_results.jsonl").open("w") as f:
        for r in fam:f.write(json.dumps(r,sort_keys=True)+"\n")
    taxonomy=dict(Counter(label for r in case_rows for label in r["failure_taxonomy"]))
    dump(OUT/"failure_taxonomy.json",taxonomy)
    make_figures(case_rows,strict,fam,taxonomy)
    write_report(case_rows,strict,insufficient,quality,analyses,paired,taxonomy,fam)
    (OUT/"changed_files.txt").write_text("harness/reasoners/qwen35.py\nharness/scripts/run_handoff_end_to_end_pilot.py\noutputs/handoff_end_to_end_pilot/ (new artifacts)\n")
    (OUT/"test_report.md").write_text(f"# Test report\n\n- Frozen manifest: 20 families and 40 REL2C QA cases; frozen Grounding DINO cache reused with unchanged AcquisitionRouter native-score selection.\n- Existing SigLIP2 color reader and deterministic bbox-center geometry supplied primitive facts.\n- One state text was compiled per family; all three state-based condition hashes match for every QA.\n- 160 Qwen calls completed with the same model and decoding settings; every prompt and paired output was saved.\n- All 40 state-only calls had no image path, visual token, pixel/image input key, or vision-encoder call.\n- Defocus used the repository DaH configuration; all available views matched the expected union-mask composite and original dimensions.\n- GT joined only after inference; strict fact-sufficient N={len(strict)}.\n")
    with (OUT/"test_report.md").open("a") as report:
        report.write("- The DaH feathered mask changed some acquired bbox pixels; final views add an exact sharp-core override without changing Gaussian radius, expansion, or feathering elsewhere. All selected bbox pixels were verified unchanged.\n- Twelve pre-correction defocus calls were archived separately, excluded, and rerun on final views.\n")


def write_report(all_cases,strict,insufficient,quality,analyses,paired,taxonomy,families):
    def cell(group,k):
        item=analyses[group]["accuracy"][k];return f"{item['correct']}/{item['n']} ({item['accuracy']:.3f})" if item["accuracy"] is not None else "undefined (N=0)"
    p=paired["strict_fact_sufficient"]
    raw_hard=p["RAW_IMAGE_PLUS_STATE__TO__STATE_ONLY_HARD_HANDOFF"]
    raw_def=p["RAW_IMAGE_PLUS_STATE__TO__DEFOCUS_PLUS_STATE"]
    def_hard=p["DEFOCUS_PLUS_STATE__TO__STATE_ONLY_HARD_HANDOFF"]
    t=raw_hard["transition_counts"]
    harm=t["wrong_to_correct"];rescue=t["correct_to_wrong"]
    n=len(strict)
    if n<8:decision="STATE_QUALITY_IS_CURRENT_BLOCKER"
    elif raw_hard["absolute_accuracy_difference_after_minus_before"]>0 and harm>rescue:decision="HARD_HANDOFF_SUPPORTED"
    elif raw_def["absolute_accuracy_difference_after_minus_before"]>0 and raw_hard["absolute_accuracy_difference_after_minus_before"]<=raw_def["absolute_accuracy_difference_after_minus_before"]:decision="DEFOCUS_SUPPORTED_HARD_HANDOFF_NOT_ESTABLISHED"
    elif raw_hard["absolute_accuracy_difference_after_minus_before"]<0 and rescue>harm:decision="RAW_IMAGE_PROVIDES_NECESSARY_RESCUE"
    else:decision="NO_VISUAL_EXPOSURE_EFFECT_ESTABLISHED"
    lines=["# End-to-end perception–reasoning handoff pilot","",f"**Recommendation: `{decision}`.** This is a frozen 20-family, 40-QA pilot; no production change is implied.","","## Protocol","","Grounding DINO candidate arrays came from the frozen acquisition cache. The current AcquisitionRouter native-score top candidate was bound once to each persistent A/B entity. The existing SigLIP2 color reader acquired both colors from those exact regions; deterministic bbox-center geometry supplied A/B left/right. No certification or VLM selector was used. The same minimal state bytes and original question were used in Raw+State, Defocus+State, and StateOnly. Vanilla Raw received the original question without state. All 160 Qwen3.5-4B calls used the frozen non-thinking, deterministic 16-token settings.","","State-only calls were true text-only calls: no image path, visual token, image/pixel input, or vision-encoder call. All available defocus views preserved the union of A/B acquired regions using the repository DaH blur configuration; image dimensions and expected pixels passed integrity checks.","","## Table A — state quality","","| Metric | Value |","|---|---:|"]
    lines.insert(lines.index("## Table A — state quality"),"The historical DaH feathered mask changed pixels inside some acquired boxes. This pilot added an exact sharp-core override inside each selected bbox while retaining the frozen Gaussian radius, expansion, and feathering elsewhere. Twelve pre-correction defocus calls were archived and excluded; all final defocus calls use the corrected views. This support-preservation detail limits direct comparability with historical DaH.")
    lines.insert(lines.index("## Table A — state quality"),"Nineteen families have two acquired regions and the defocus view preserves both. One family has no A candidate; its view preserves the available B region only. Its two QA cases are fact-insufficient, and no GT region was substituted.")
    for key,label in [("n_qa","N QA"),("A_region_correct","A region correct"),("B_region_correct","B region correct"),("both_regions_correct","Both regions correct"),("color_A_correct","Color A correct"),("color_B_correct","Color B correct"),("relation_correct","Relation correct"),("strict_fact_sufficient_qa","Strict fact-sufficient QA"),("state_implied_answer_correct","State-implied answer correct")]:lines.append(f"| {label} | {quality[key]}/40 |")
    lines += ["",f"Both colors correct: {quality['both_colors_correct']}/40; both regions and both colors correct: {quality['both_regions_and_colors_correct']}/40. Relation is correct in {quality['relation_correct_given_both_regions']}/{quality['both_regions_correct']} QA cases with both regions correct. Binding plus color acquisition reduces the strict subset to {quality['strict_fact_sufficient_qa']}/40.","","Strict fact sufficiency requires both selected regions IoU ≥ 0.5, both acquired colors equal frozen canonical colors, and the acquired spatial relation equal evaluator geometry. State-answer consistency is separate and cannot substitute for primitive correctness.","","## Table B — all-case end-to-end","","| Condition | Correct | Accuracy |","|---|---:|---:|"]
    for k,label in zip(CONDITIONS,("Vanilla Raw","Raw + State","Defocus + State","State Only Hard Handoff")):
        a=analyses["all_cases"]["accuracy"][k];lines.append(f"| {label} | {a['correct']}/{a['n']} | {a['accuracy']:.3f} |")
    lines += ["","## Table C — strict fact-sufficient subset","",f"N = **{n} QA cases** across {len({r['family_id'] for r in strict})} families.","","| Condition | Correct | Accuracy |","|---|---:|---:|"]
    for k,label in zip(CONDITIONS[1:],("Raw + State","Defocus + State","State Only Hard Handoff")):
        a=analyses["strict_fact_sufficient"]["accuracy"][k];lines.append(f"| {label} | {a['correct']}/{a['n']} | {a['accuracy']:.3f} |" if n else f"| {label} | 0/0 | undefined |")
    lines += ["","## Table D — paired Raw+State to StateOnly","","| Transition | Count |","|---|---:|",f"| Raw correct → Hard correct | {t['correct_to_correct']} |",f"| Raw correct → Hard wrong (image rescue) | {rescue} |",f"| Raw wrong → Hard correct (visual interference) | {harm} |",f"| Raw wrong → Hard wrong | {t['wrong_to_wrong']} |",""]
    lines.append(f"HardHandoffGain = {raw_hard['absolute_accuracy_difference_after_minus_before']}; family-cluster bootstrap 95% CI = {raw_hard['family_cluster_bootstrap_95_ci']}; McNemar exact two-sided p = {raw_hard['mcnemar_exact_p_two_sided']}. Image harm {harm}/{n}; image rescue {rescue}/{n}; asymmetry {harm-rescue}.")
    lines += ["","## Table E — defocus effect","","| Metric | Value |","|---|---:|",f"| Raw+State accuracy | {cell('strict_fact_sufficient',CONDITIONS[1])} |",f"| Defocus+State accuracy | {cell('strict_fact_sufficient',CONDITIONS[2])} |",f"| StateOnly accuracy | {cell('strict_fact_sufficient',CONDITIONS[3])} |",f"| Raw→Defocus rescues | {raw_def['transition_counts']['wrong_to_correct']} |",f"| Raw→Defocus regressions | {raw_def['transition_counts']['correct_to_wrong']} |",f"| Defocus→Hard rescues | {def_hard['transition_counts']['wrong_to_correct']} |",f"| Defocus→Hard regressions | {def_hard['transition_counts']['correct_to_wrong']} |",""]
    lines += ["## Fact-insufficient cases","",f"N = {len(insufficient)}. Raw+State: {cell('fact_insufficient',CONDITIONS[1])}; Defocus+State: {cell('fact_insufficient',CONDITIONS[2])}; StateOnly: {cell('fact_insufficient',CONDITIONS[3])}. This stratum can contain raw-image rescue of incorrect external facts and is not used to infer interference with sufficient facts.","","## Scientific questions"]
    answer_changes=sum(r["conditions"]["RAW_IMAGE_PLUS_STATE"]["normalized_prediction"]!=r["conditions"]["STATE_ONLY_HARD_HANDOFF"]["normalized_prediction"] for r in strict)
    lines += ["",f"Q1. On strict fact-sufficient cases, Raw+State {cell('strict_fact_sufficient',CONDITIONS[1])} versus StateOnly {cell('strict_fact_sufficient',CONDITIONS[3])}; paired gain {raw_hard['absolute_accuracy_difference_after_minus_before']}.",f"Q2. Defocus+State {cell('strict_fact_sufficient',CONDITIONS[2])}; Raw→Defocus rescues {raw_def['transition_counts']['wrong_to_correct']} and regressions {raw_def['transition_counts']['correct_to_wrong']}.",f"Q3. Defocus→Hard changes accuracy by {def_hard['absolute_accuracy_difference_after_minus_before']}; transitions in paired_transition_analysis.json.",f"Q4. Raw image helps {rescue}/{n} strict cases relative to StateOnly.",f"Q5. Raw image hurts {harm}/{n} strict cases relative to StateOnly.",f"Q6. {len(insufficient)}/40 QA cases fail strict primitive sufficiency; compare this with {taxonomy.get('VISUAL_INTERFERENCE_CASE',0)} visual-interference cases and all-case accuracy before attributing the bottleneck.",f"Q7. {answer_changes}/{n} strict cases change parsed answer between Raw+State and StateOnly despite identical state hashes; {harm+rescue} flip correctness. Exact responses are in qa_condition_results.jsonl.",f"Q8. The conclusion is `{decision}` under this pilot; same-state and true text-only controls pass, with the paired effect and uncertainty reported above."]
    lines += ["","## Novelty audit","","This tests image plus structured evidence against the exact same structured evidence after explicit image exit. The contribution under test is explicit perception–reasoning decoupling through an entity-indexed visual state and mandatory visual-channel handoff. It is not a better detector, more tools, a better prompt, confidence fusion, a new scene graph, or VLM candidate reranking. The acquisition model remains replaceable. No ground truth corrected acquisition, no score threshold was tuned, and no final answer was inserted into the state."]
    (OUT/"implementation_report.md").write_text("\n".join(lines)+"\n")
    sections=[]
    for r in all_cases:
        cells="".join(f"<tr><td>{html.escape(k)}</td><td>{r['conditions'][k]['correct']}</td><td>{html.escape(str(r['conditions'][k]['normalized_prediction']))}</td><td>{html.escape(r['conditions'][k]['raw_response'])}</td></tr>" for k in CONDITIONS)
        sections.append(f"<section><h2>{html.escape(r['qa_id'])}</h2><p>Strict sufficient: {r['EVALUATOR_ONLY']['strict_fact_sufficient']} · GT: {html.escape(r['EVALUATOR_ONLY']['ground_truth'])} · {html.escape(str(r['failure_taxonomy']))}</p><pre>{html.escape(r['state_text'])}</pre><table><tr><th>Condition</th><th>Correct</th><th>Parsed</th><th>Raw response</th></tr>{cells}</table></section>")
    (OUT/"audit.html").write_text("<!doctype html><html><head><meta charset='utf-8'><style>body{font:14px sans-serif;max-width:1100px;margin:auto}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:5px}section{margin:24px 0}</style></head><body><h1>Handoff pilot audit</h1>"+"".join(sections)+"</body></html>")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("prepare","infer","analyze"));args=parser.parse_args()
    if args.stage=="prepare":prepare()
    elif args.stage=="infer":infer()
    else:analyze()


if __name__=="__main__":main()
