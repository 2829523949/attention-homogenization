#!/usr/bin/env python3
from __future__ import annotations

import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
PROJECT=Path("/root/Qwen3-VL"); SCRIPT_DIR=PROJECT/"experiments/realbind_phase2/scripts"
for p in (PROJECT,SCRIPT_DIR):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from common import CANONICAL_COLORS
from harness.configs.paths import Paths,ensure_d_drive_environment
from harness.dataset import evaluator_records
from harness.evaluators.end2end_metrics import score
from harness.io_utils import write_jsonl_new
from harness.processors.base import VisualState
from harness.processors.defocus import auto_defocus
from harness.reasoners.qwen35 import Qwen35Reasoner

VARIANTS=("original","auto_blur","auto_handoff","auto_full_dah")


def select(state:VisualState,side:str):
    valid=[e for e in state.entities if e.center_x is not None and e.predicted_region]
    if len(valid)!=2:return None
    valid.sort(key=lambda e:(e.center_x,e.reference_id)); return valid[0] if side=="left" else valid[-1]


def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument("--states",type=Path,required=True);ap.add_argument("--variant",choices=VARIANTS,required=True);ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--allow-full-benchmark",action="store_true");args=ap.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    items=[json.loads(x) for x in args.states.read_text().splitlines() if x.strip()]
    if len(items)>5 and not args.allow_full_benchmark:raise RuntimeError("safety gate: >5 cases requires --allow-full-benchmark")
    p=Paths();ensure_d_drive_environment(p);families,qa=evaluator_records(p);vocab={"color":list(CANONICAL_COLORS),"position":["left","right"],"identity":sorted({f[s]["object"] for f in families.values() for s in ("A","B")})}
    args.output_dir.mkdir(parents=True);reasoner=Qwen35Reasoner();rows=[]
    for item in items:
        case=item["case"];state=VisualState.from_dict(item["visual_state"]);family=families[case["family_id"]];gt=qa[case["case_id"]];side="left" if case["qa_type"].endswith("LEFT") else "right";target=select(state,side)
        prompt=case["question"];image=Path(case["image_path"]);meta=None
        if args.variant=="auto_blur":
            meta=auto_defocus(image,[e.predicted_region for e in state.entities if e.predicted_region],args.output_dir/"derived_images"/f"{state.case_id}_pair.png");image=Path(meta["output_path"])
        elif args.variant in {"auto_handoff","auto_full_dah"}:
            if target is None: rows.append({"case_id":state.case_id,"variant":args.variant,"abstained":True,"reason":"automatic target unavailable"});continue
            prompt=next(r["question"] for r in qa.values() if r["family_id"]==case["family_id"] and r["qa_type"]==f"I2C_{target.reference_id}")
            if args.variant=="auto_full_dah":
                meta=auto_defocus(image,[target.predicted_region],args.output_dir/"derived_images"/f"{state.case_id}_target.png");image=Path(meta["output_path"])
        generated=reasoner.generate(prompt,image);parsed=score(generated["response"],"color",str(gt["ground_truth"]),family,vocab)
        rows.append({"case_id":state.case_id,"family_id":case["family_id"],"variant":args.variant,"automatic_target":target.reference_id if target else None,"prompt":prompt,"image_path":str(image),"defocus":meta,"generation":generated,"ground_truth":gt["ground_truth"],"score":parsed})
    write_jsonl_new(args.output_dir/"results.jsonl",rows);return 0
if __name__=="__main__":raise SystemExit(main())
