#!/usr/bin/env python3
from __future__ import annotations

import argparse,html,json,sys
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from harness.compiler.fact_compiler import compile_facts,serialize_facts
from harness.configs.paths import Paths
from harness.dataset import evaluator_records
from harness.evaluators.failure_taxonomy import classify,summarize
from harness.io_utils import write_json_new,write_new
from harness.processors.base import VisualState


def draw_boxes(image_path:Path,boxes:list[tuple[list[float],str]],output:Path,color:str)->None:
    image=Image.open(image_path).convert("RGB");draw=ImageDraw.Draw(image)
    for box,label in boxes:
        draw.rectangle(tuple(box),outline=color,width=4);draw.text((box[0]+3,box[1]+3),label,fill=color,stroke_width=2,stroke_fill="black")
    image.save(output)


def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument("--states",type=Path,required=True);ap.add_argument("--processor-results",type=Path,required=True);ap.add_argument("--pipeline-results",type=Path,required=True);ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--limit",type=int,default=20);args=ap.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);images=args.output_dir/"images";images.mkdir()
    if not 1<=args.limit<=20:raise ValueError("HTML audit limit must be 1..20")
    items=[json.loads(x) for x in args.states.read_text().splitlines() if x.strip()][:args.limit];proc={r["case_id"]:r for r in map(json.loads,args.processor_results.read_text().splitlines())};pipe={r["case_id"]:r for r in map(json.loads,args.pipeline_results.read_text().splitlines())}
    p=Paths();families,qa=evaluator_records(p);historical_path=p.realbind/"experiments/pair_direct_v1_1_qwen35_4b_baseline/inference_results.jsonl";historical={r["qa_id"]:r for r in map(json.loads,historical_path.read_text().splitlines())}
    cards=[];failure_rows=[]
    for index,item in enumerate(items):
        case=item["case"];state=VisualState.from_dict(item["visual_state"]);family=families[case["family_id"]];pr=proc[state.case_id];result=pipe.get(state.case_id,{})
        pred_boxes=[(list(e.predicted_region.bbox_xyxy_pixels),f"PRED {e.reference_id}: {e.category}") for e in state.entities if e.predicted_region]
        gt_boxes=[]
        for ref in ("A","B"):
            x,y,w,h=family[ref]["bbox"];gt_boxes.append(([x,y,x+w,y+h],f"EVALUATOR-ONLY GT {ref}"))
        pred_name=f"{index:02d}_pred.png";gt_name=f"{index:02d}_gt_evaluator_only.png";draw_boxes(Path(case["image_path"]),pred_boxes,images/pred_name,"cyan");draw_boxes(Path(case["image_path"]),gt_boxes,images/gt_name,"red")
        facts=compile_facts(state,{"A":case["object_A"],"B":case["object_B"]});selected=result.get("selected_target");score=result.get("score",{});pinfo=pr["entities"]
        failure_input={"case_id":state.case_id,"localization_ok":all(pinfo[x]["localized_iou_0_5"] for x in ("A","B")),"identity_mapping_ok":all(pinfo[x]["localized_iou_0_5"] for x in ("A","B")),"attribute_ok":all(pinfo[x]["color_correct_auto_region"] for x in ("A","B")),"geometry_ok":bool(pr["relation_accuracy_automatic_regions"]),"compilation_ok":True,"all_required_facts_ok":bool(pr["required_fact_exact_match"]),"final_correct":score.get("correct")};failure=classify(failure_input);failure_rows.append(failure_input)
        colors={e.reference_id:e.color for e in state.entities};relation=next((r.relation for r in state.relations if r.source_entity=="A"),None);orig=historical.get(state.case_id,{})
        cards.append(f"""<section><h2>{html.escape(state.case_id)}</h2><div class='grid'><figure><img src='images/{pred_name}'><figcaption>Automatic Florence regions (inference)</figcaption></figure><figure><img src='images/{gt_name}'><figcaption class='gt'>GT boxes — evaluator visualization only, never inference input</figcaption></figure></div><p><b>Mask:</b> SAM2 not run (optional module). <b>Mapping:</b> A={html.escape(case['object_A'])}, B={html.escape(case['object_B'])}. <b>Colors:</b> {html.escape(str(colors))}. <b>Relation:</b> A is {html.escape(str(relation))} of B.</p><pre>{html.escape(serialize_facts(facts))}</pre><p><b>Automatic target:</b> {html.escape(str(selected))} · <b>Historical Original:</b> {html.escape(str(orig.get('normalized_prediction')))} · <b>Automatic pipeline:</b> {html.escape(str(score.get('normalized_prediction')))} · <b>GT answer:</b> {html.escape(str(qa[state.case_id]['ground_truth']))} · <b>Failure:</b> {failure}</p></section>""")
    css="body{font:15px sans-serif;max-width:1200px;margin:auto}section{border-top:2px solid #555;padding:1rem}.grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}img{max-width:100%;height:auto}.gt{color:#b00;font-weight:bold}pre{white-space:pre-wrap;background:#eee;padding:.7rem}"
    write_new(args.output_dir/"visual_audit.html",f"<!doctype html><meta charset='utf-8'><title>RealBind automatic repair smoke audit</title><style>{css}</style><h1>RealBind automatic repair visual audit</h1><p>Red GT overlays are evaluator-only and were not provided to any automatic processor.</p>{''.join(cards)}")
    write_json_new(args.output_dir/"failure_taxonomy.json",summarize(failure_rows));print(args.output_dir/"visual_audit.html");return 0
if __name__=="__main__":raise SystemExit(main())
