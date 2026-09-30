#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
PROJECT=Path("/root/Qwen3-VL"); SCRIPT_DIR=PROJECT/"experiments/realbind_phase2/scripts"
for p in (PROJECT,SCRIPT_DIR):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from harness.compiler.fact_compiler import compile_facts,prompt_with_facts,serialize_facts
from harness.configs.paths import Paths,ensure_d_drive_environment
from harness.dataset import evaluator_records
from harness.evaluators.end2end_metrics import score
from harness.io_utils import write_jsonl_new
from harness.processors.base import VisualState
from harness.processors.defocus import auto_defocus
from harness.reasoners.qwen35 import Qwen35Reasoner
from common import CANONICAL_COLORS

VARIANTS=("original","auto_facts_only","image_plus_auto_facts","auto_defocus_plus_auto_facts","auto_facts_plus_handoff","full_automatic_pipeline")


def selected_target(state:VisualState,qa_type:str)->str|None:
    entities={e.reference_id:e for e in state.entities}
    if any(entities.get(x) is None or entities[x].center_x is None for x in ("A","B")): return None
    ordered=sorted(("A","B"),key=lambda x:(entities[x].center_x,x))
    return ordered[0] if qa_type.endswith("LEFT") else ordered[-1]


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--states",type=Path,required=True); ap.add_argument("--variant",choices=VARIANTS,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--allow-full-benchmark",action="store_true"); args=ap.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    items=[json.loads(x) for x in args.states.read_text().splitlines() if x.strip()]
    if len(items)>5 and not args.allow_full_benchmark: raise RuntimeError("safety gate: >5 cases requires --allow-full-benchmark")
    paths=Paths(); ensure_d_drive_environment(paths); families,qa_by_id=evaluator_records(paths)
    vocab={"color":list(CANONICAL_COLORS),"position":["left","right"],"identity":sorted({f[s]["object"] for f in families.values() for s in ("A","B")})}
    args.output_dir.mkdir(parents=True); defocus_dir=args.output_dir/"derived_images"; reasoner=Qwen35Reasoner(); rows=[]
    for item in items:
        case=item["case"]; state=VisualState.from_dict(item["visual_state"]); family=families[case["family_id"]]; gt=qa_by_id[case["case_id"]]
        facts=compile_facts(state,{"A":case["object_A"],"B":case["object_B"]}); handoff=selected_target(state,case["qa_type"])
        image=Path(case["image_path"]); prompt=case["question"]; defocus_meta=None
        if args.variant!="original": prompt=prompt_with_facts(case["question"],facts,handoff if args.variant in {"auto_facts_plus_handoff","full_automatic_pipeline"} else None)
        if args.variant in {"auto_facts_only","auto_facts_plus_handoff"}: image=None
        elif args.variant in {"auto_defocus_plus_auto_facts","full_automatic_pipeline"}:
            regions=[e.predicted_region for e in state.entities if e.predicted_region]
            target=defocus_dir/f"{state.case_id}.png"; defocus_meta=auto_defocus(Path(case["image_path"]),regions,target); image=target
        generated=reasoner.generate(prompt,image); parsed=score(generated["response"],"color",str(gt["ground_truth"]),family,vocab)
        rows.append({"case_id":state.case_id,"family_id":case["family_id"],"variant":args.variant,"compiled_facts":facts,"serialized_facts":serialize_facts(facts),"selected_target":handoff,"defocus":defocus_meta,"generation":generated,"ground_truth":gt["ground_truth"],"score":parsed})
        print(json.dumps({"case_id":state.case_id,"variant":args.variant,"prediction":parsed.get("normalized_prediction"),"correct":parsed["correct"]}))
    write_jsonl_new(args.output_dir/"results.jsonl",rows); return 0
if __name__=="__main__": raise SystemExit(main())
